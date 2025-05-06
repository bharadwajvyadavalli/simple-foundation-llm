"""
Model Server Module for SimpleFoundation

This module provides a lightweight FastAPI server for serving
the trained foundational model for inference.
"""

import os
import json
import time
import logging
import torch
import asyncio
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input prompt for the model")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature (0.0 to 1.0)")
    top_p: float = Field(0.9, description="Top-p sampling parameter (0.0 to 1.0)")
    top_k: int = Field(50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, description="Repetition penalty (1.0 = no penalty)")
    do_sample: bool = Field(True, description="Whether to use sampling (True) or greedy decoding (False)")
    stream: bool = Field(False, description="Whether to stream the response token by token")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    
    generated_text: str = Field(..., description="Generated text")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of generated tokens")


class ModelInfo(BaseModel):
    """Model information."""
    
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (full or LoRA)")
    quantization: Optional[str] = Field(None, description="Quantization type (None, 8-bit, 4-bit)")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    loaded_at: str = Field(..., description="Timestamp when model was loaded")


class InferenceServer:
    """Inference server for SimpleFoundation models."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
        max_concurrent_requests: int = 1
    ):
        """
        Initialize the inference server.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on (default: auto-detect)
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
            use_flash_attention: Whether to use flash attention for faster inference
            max_concurrent_requests: Maximum number of concurrent inference requests
        """
        self.model_path = model_path
        self.use_flash_attention = use_flash_attention
        self.max_concurrent_requests = max_concurrent_requests
        
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="SimpleFoundation Inference API",
            description="API for running inference on SimpleFoundation models",
            version="0.1.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._setup_routes()
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine if this is a LoRA model
        self.is_lora = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
        
        # Quantization settings
        if self.load_in_4bit:
            logger.info("Loading model in 4-bit precision")
            quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
            self.quantization = "4-bit"
        elif self.load_in_8bit:
            logger.info("Loading model in 8-bit precision")
            quantization_config = {"load_in_8bit": True}
            self.quantization = "8-bit"
        else:
            quantization_config = None
            self.quantization = None
        
        # Flash attention settings
        attn_config = {"use_flash_attention_2": self.use_flash_attention} if self.use_flash_attention else {}
        
        # If it's a LoRA model, we need to load the base model first
        if self.is_lora:
            # Try to find the base model path in the config
            try:
                with open(os.path.join(self.model_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")
            except:
                # If we can't find it, look for a training config file
                try:
                    with open(os.path.join(self.model_path, "training_config.json"), 'r') as f:
                        training_config = json.load(f)
                    base_model_path = training_config.get("model_name_or_path")
                except:
                    raise ValueError("Could not determine base model path for LoRA model")
            
            logger.info(f"Loading base model {base_model_path} for LoRA")
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager"
            )
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            self.model_type = "LoRA"
            self.model_name = f"{os.path.basename(base_model_path)}+{os.path.basename(self.model_path)}"
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager"
            )
            self.model_type = "Full"
            self.model_name = os.path.basename(self.model_path)
        
        # Set up generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        except:
            # Use default config if not found
            self.generation_config = GenerationConfig(
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        self.max_sequence_length = self.model.config.max_position_embeddings
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def _setup_routes(self):
        """Set up API routes."""
        @self.app.get("/")
        async def root():
            """API root endpoint."""
            return {"message": "SimpleFoundation Inference API"}
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "device": self.device}
        
        @self.app.get("/model/info", response_model=ModelInfo)
        async def model_info():
            """Get model information."""
            return ModelInfo(
                model_name=self.model_name,
                model_type=self.model_type,
                quantization=self.quantization,
                max_sequence_length=self.max_sequence_length,
                loaded_at=self.loaded_at
            )
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
            """
            Generate text based on input prompt.
            
            Args:
                request: Generation request parameters
                
            Returns:
                Generated text and metadata
            """
            # Acquire semaphore to limit concurrent requests
            async with self.semaphore:
                logger.info(f"Processing generation request: {request.prompt[:50]}...")
                
                try:
                    start_time = time.time()
                    
                    # Tokenize prompt
                    inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.model.device)
                    input_token_count = inputs.input_ids.shape[1]
                    
                    # Set up generation parameters
                    generation_kwargs = {
                        "max_new_tokens": request.max_new_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                        "repetition_penalty": request.repetition_penalty,
                        "do_sample": request.do_sample,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id
                    }
                    
                    # TODO: Implement streaming if request.stream is True
                    if request.stream:
                        return HTTPException(status_code=501, detail="Streaming not yet implemented")
                    
                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            **generation_kwargs
                        )
                    
                    # Extract generated text (without the prompt)
                    generated_ids = outputs[0][input_token_count:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # Calculate metrics
                    end_time = time.time()
                    generation_time = end_time - start_time
                    output_token_count = len(generated_ids)
                    
                    logger.info(f"Generated {output_token_count} tokens in {generation_time:.2f}s")
                    
                    return GenerationResponse(
                        generated_text=generated_text,
                        generation_time=generation_time,
                        input_tokens=input_token_count,
                        output_tokens=output_token_count
                    )
                
                except Exception as e:
                    logger.error(f"Error generating text: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the inference server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        import uvicorn
        
        logger.info(f"Starting inference server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def start_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
    use_flash_attention: bool = True,
    max_concurrent_requests: int = 1
):
    """
    Start the inference server.
    
    Args:
        model_path: Path to model checkpoint
        host: Host to bind to
        port: Port to listen on
        load_in_8bit: Whether to load model in 8-bit quantization
        load_in_4bit: Whether to load model in 4-bit quantization
        use_flash_attention: Whether to use flash attention
        max_concurrent_requests: Maximum number of concurrent requests
    """
    server = InferenceServer(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        use_flash_attention=use_flash_attention,
        max_concurrent_requests=max_concurrent_requests
    )
    
    server.run(host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start an inference server for SimpleFoundation models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization")
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable flash attention")
    parser.add_argument("--max_concurrent", type=int, default=1, help="Maximum concurrent requests")
    
    args = parser.parse_args()
    
    start_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=not args.no_flash_attention,
        max_concurrent_requests=args.max_concurrent
    )

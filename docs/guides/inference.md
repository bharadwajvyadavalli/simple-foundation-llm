# Inference Documentation

## Overview

The inference module in SimpleFoundation provides tools for deploying and serving trained models. This module enables you to use your model for real-world applications, interact with it through a web interface, and integrate it into other systems.

![Inference Architecture](../images/architecture.png)

## Key Components

### Model Server

```
inference/server/model_server.py
```

The model server is a FastAPI application that provides:
- RESTful API endpoints for model inference
- Efficient model loading and serving
- Request handling and response formatting
- Health monitoring

### Web UI Demo

```
inference/demo/app.py
```

A Streamlit-based web interface that allows:
- Interactive problem solving
- Testing model capabilities
- Visualizing model outputs
- User-friendly experience without coding

## Setting Up the Model Server

### 1. Basic Setup

```bash
# Start the model server
python -m inference.server.model_server \
    --model_path training/checkpoints/final \
    --host 0.0.0.0 \
    --port 8000 \
    --load_in_4bit
```

### 2. Optimized Setup

```bash
# Start the model server with optimizations
python -m inference.server.model_server \
    --model_path training/checkpoints/final \
    --host 0.0.0.0 \
    --port 8000 \
    --load_in_4bit \
    --no_flash_attention  # Disable if causing issues
```

### 3. Using Environment Variables

```bash
# Set environment variables
export MODEL_PATH="training/checkpoints/final"
export HOST="0.0.0.0"
export PORT="8000"
export LOAD_IN_4BIT="true"

# Start the server
python -m inference.server.model_server
```

## API Endpoints

The model server provides several RESTful endpoints:

### 1. Health Check

```
GET /health
```

Returns the health status of the server.

Example response:
```json
{
  "status": "healthy"
}
```

### 2. Generate Text

```
POST /generate
```

Generates text from a prompt.

Request body:
```json
{
  "prompt": "Explain the concept of recursion in programming.",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "num_return_sequences": 1,
  "stream": false
}
```

Response:
```json
{
  "prompt": "Explain the concept of recursion in programming.",
  "generated_text": "Recursion is a programming concept where a function calls itself...",
  "generation_time": 1.25,
  "tokens_generated": 128
}
```

### 3. Solve Math Problems

```
POST /solve/math
```

Solves mathematical problems with step-by-step reasoning.

Request body:
```json
{
  "problem": "Solve for x: 2x + 3 = 9",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

Response:
```json
{
  "problem": "Solve for x: 2x + 3 = 9",
  "solution": "I'll solve this step by step...",
  "answer": "3",
  "generation_time": 0.85
}
```

### 4. Solve Coding Problems

```
POST /solve/coding
```

Solves coding problems with explanations and code.

Request body:
```json
{
  "problem": "Write a function to calculate the Fibonacci sequence.",
  "max_new_tokens": 1024,
  "temperature": 0.8,
  "stream": false
}
```

Response:
```json
{
  "problem": "Write a function to calculate the Fibonacci sequence.",
  "solution": "I'll approach this step by step...",
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "generation_time": 1.42
}
```

## Using the Web UI

### 1. Starting the UI

```bash
# Start the UI (make sure the model server is running)
export API_URL="http://localhost:8000"  # Point to your model server
streamlit run inference/demo/app.py
```

### 2. UI Features

The web UI provides:
- Math problem solver tab
- Coding problem solver tab
- Example problems for quick testing
- Temperature and token limit controls
- Server status indicator
- Response timing information

### 3. Customizing the UI

You can customize the UI by modifying `inference/demo/app.py`:
- Change the theme and layout
- Add new example problems
- Create additional tabs for different tasks
- Integrate visualization tools

## Programmatic Usage

### Python Client

```python
import requests

# Model server URL
API_URL = "http://localhost:8000"

# Solve a math problem
def solve_math_problem(problem):
    response = requests.post(
        f"{API_URL}/solve/math",
        json={
            "problem": problem,
            "temperature": 0.7,
            "max_new_tokens": 512
        }
    )
    return response.json()

# Example usage
result = solve_math_problem("Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3")
print(f"Solution: {result['solution']}")
print(f"Answer: {result['answer']}")
```

### Batch Processing

```python
import requests
import json
from concurrent.futures import ThreadPoolExecutor

# Process a batch of problems
def process_batch(problems, problem_type="math", max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if problem_type == "math":
            futures = [executor.submit(solve_math_problem, p) for p in problems]
        else:
            futures = [executor.submit(solve_coding_problem, p) for p in problems]
        
        results = [f.result() for f in futures]
    
    return results

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Streaming Responses

For longer generations, streaming provides a better user experience:

```python
import requests
import json

# Stream a response
def stream_solution(problem, problem_type="math"):
    url = f"http://localhost:8000/solve/{problem_type}"
    
    response = requests.post(
        url,
        json={
            "problem": problem,
            "stream": True,
            "temperature": 0.7
        },
        stream=True
    )
    
    # Process the streamed response
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode('utf-8'), end='', flush=True)
```

The model server also supports streaming via server-sent events (SSE).

## Performance Optimization

### 1. Model Quantization

```bash
# Use 4-bit quantization (best trade-off between speed and quality)
python -m inference.server.model_server --model_path model_path --load_in_4bit

# Use 8-bit quantization (better quality, more memory)
python -m inference.server.model_server --model_path model_path --load_in_8bit
```

### 2. Flash Attention

```bash
# Enable flash attention (default)
python -m inference.server.model_server --model_path model_path

# Disable flash attention if causing issues
python -m inference.server.model_server --model_path model_path --no_flash_attention
```

### 3. Horizontal Scaling

For high-throughput applications:
- Deploy multiple model servers
- Use a load balancer (e.g., NGINX, HAProxy)
- Implement caching for common requests

## Deployment Options

### 1. Docker

```bash
# Build Docker image
docker build -t simplefoundation:latest .

# Run container
docker run -p 8000:8000 -p 8501:8501 \
    -v /path/to/model:/app/model \
    -e MODEL_PATH=/app/model \
    simplefoundation:latest
```

### 2. Kubernetes

Create a deployment manifest (`deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: simplefoundation
spec:
  replicas: 1
  selector:
    matchLabels:
      app: simplefoundation
  template:
    metadata:
      labels:
        app: simplefoundation
    spec:
      containers:
      - name: model-server
        image: simplefoundation:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/model"
        volumeMounts:
        - name: model-volume
          mountPath: /app/model
        resources:
          limits:
            nvidia.com/gpu: 1
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

Create a service manifest (`service.yaml`):
```yaml
apiVersion: v1
kind: Service
metadata:
  name: simplefoundation-service
spec:
  selector:
    app: simplefoundation
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 3. Cloud Deployment

See the [Deployment Guide](./deployment_guide.md) for detailed instructions on deploying to:
- AWS (EC2, SageMaker)
- Google Cloud (Compute Engine, AI Platform)
- Azure (VMs, Azure ML)
- Specialized GPU providers (Lambda Labs, Vast.ai)

## Security Considerations

### 1. API Authentication

Add authentication to the FastAPI server:

```python
from fastapi.security import APIKeyHeader
from fastapi import Security, HTTPException, Depends

# Define API key header
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Valid API keys (in production, use a secure database or auth service)
API_KEYS = ["your-secret-key-1", "your-secret-key-2"]

# Dependency for endpoints
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Protected endpoint example
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, api_key: str = Depends(get_api_key)):
    # Endpoint implementation
    ...
```

### 2. Request Validation

The server already implements validation via Pydantic models, but you can add additional checks:

```python
# Add content filtering
def filter_request(request: GenerationRequest):
    # Check for prohibited content
    prohibited_terms = ["list", "of", "prohibited", "terms"]
    if any(term in request.prompt.lower() for term in prohibited_terms):
        raise HTTPException(status_code=400, detail="Prohibited content in request")
    return request

# Use in endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest = Depends(filter_request)):
    # Endpoint implementation
    ...
```

### 3. Rate Limiting

Add rate limiting to prevent abuse:

```bash
pip install fastapi-limiter redis
```

```python
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# Initialize limiter
@app.on_event("startup")
async def startup():
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    await FastAPILimiter.init(redis_client)

# Apply rate limiting to endpoint
@app.post("/generate", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def generate_text(request: GenerationRequest):
    # Endpoint implementation
    ...
```

## Monitoring and Logging

### 1. Basic Logging

The server includes basic logging setup:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example log
logger.info(f"Processing request: {request}")
```

### 2. Advanced Monitoring

For production deployments, integrate with monitoring systems:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUESTS = Counter('model_requests_total', 'Total number of requests', ['endpoint'])
LATENCY = Histogram('model_latency_seconds', 'Request latency in seconds', ['endpoint'])

# Start metrics server on port 8001
start_http_server(8001)

# Use in endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    REQUESTS.labels(endpoint='generate').inc()
    
    start_time = time.time()
    # Process request
    result = ...
    
    LATENCY.labels(endpoint='generate').observe(time.time() - start_time)
    return result
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Check model path is correct
   - Ensure enough GPU memory
   - Try lower quantization (4-bit instead of 8-bit)

2. **Slow Inference**:
   - Enable flash attention
   - Use smaller context lengths
   - Reduce temperature for faster generation
   - Check for CPU bottlenecks

3. **Memory Issues**:
   - Monitor GPU memory with `nvidia-smi`
   - Implement offloading for large models
   - Reduce max token length

4. **Connection Issues**:
   - Verify host and port settings
   - Check firewall rules
   - Ensure correct network configuration in Docker

5. **UI Not Connecting to API**:
   - Verify API_URL environment variable
   - Check CORS settings in model server
   - Test API independently with curl or Postman

## Best Practices

1. **Caching**: Implement response caching for common queries
2. **Logging**: Log requests and responses for debugging
3. **Error Handling**: Implement graceful error handling and user-friendly messages
4. **Horizontal Scaling**: Deploy multiple servers behind a load balancer for high traffic
5. **Monitoring**: Set up alerts for errors and performance issues
6. **Testing**: Create automated tests for API endpoints
7. **Documentation**: Keep API documentation updated for users

For more advanced deployment scenarios, refer to the [Deployment Guide](./deployment_guide.md).

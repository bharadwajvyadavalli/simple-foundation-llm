# SimpleFoundation Deployment Guide

This guide provides instructions for deploying SimpleFoundation on various environments, from local machines to cloud platforms.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Local Deployment](#local-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [AWS](#aws)
  - [Google Cloud](#google-cloud)
  - [Azure](#azure)
  - [Lambda Labs](#lambda-labs)
  - [Vast.ai](#vastai)
- [Docker Deployment](#docker-deployment)
- [Scaling for Production](#scaling-for-production)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

## Hardware Requirements

The hardware requirements depend on the model size and training data volume:

### Minimal Setup (Smaller Models)
- **GPU**: 1 GPU with 16GB+ VRAM (e.g., RTX 3090, A5000)
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD
- **CPU**: 8+ cores
- **Estimated Cost**: $1,000-2,000 for hardware, or $0.5-1.5/hour on cloud platforms

### Recommended Setup (Mid-size Models)
- **GPU**: 4-8 GPUs with 24GB+ VRAM each (e.g., RTX 4090, A100)
- **RAM**: 128GB+
- **Storage**: 1TB+ SSD
- **CPU**: 16+ cores
- **Estimated Cost**: $10,000-15,000 for hardware, or $5-10/hour on cloud platforms

### Production Setup (Larger Models)
- **GPU**: 8+ A100 (80GB) or H100 GPUs
- **RAM**: 256GB+
- **Storage**: 2TB+ NVMe SSD
- **CPU**: 32+ cores
- **Estimated Cost**: $100,000+ for hardware, or $20-50/hour on cloud platforms

## Local Deployment

Follow these steps to deploy SimpleFoundation on your local machine:

1. **Clone the repository and set up the environment**:
   ```bash
   git clone https://github.com/yourusername/SimpleFoundation.git
   cd SimpleFoundation
   python setup.py
   ```

2. **Run the data pipeline** (if training your own model):
   ```bash
   bash scripts/run_pipeline.sh
   ```
   
   This will:
   - Download and prepare datasets
   - Generate and filter solutions
   - Format data for training
   - Train the model

3. **OR Download a pre-trained model**:
   If you don't want to train from scratch, you can download a pre-trained model:
   ```bash
   # Create a directory for the model
   mkdir -p training/checkpoints/pretrained
   
   # Download a pre-trained LoRA adapter (example)
   # Replace the URL with the actual model URL
   wget https://huggingface.co/example/simplefoundation-model/resolve/main/adapter_model.bin -O training/checkpoints/pretrained/adapter_model.bin
   wget https://huggingface.co/example/simplefoundation-model/resolve/main/adapter_config.json -O training/checkpoints/pretrained/adapter_config.json
   ```

4. **Start the inference server**:
   ```bash
   bash scripts/serve.sh
   ```
   
   This will start the FastAPI server on `http://localhost:8000`.

5. **Launch the demo UI** (in a separate terminal):
   ```bash
   bash scripts/demo.sh
   ```
   
   The Streamlit UI will be available at `http://localhost:8501`.

## Cloud Deployment

### AWS

#### EC2 Instance

1. **Launch an EC2 instance** with appropriate GPU support:
   - Instance type: `p3.2xlarge` (1 V100 GPU), `p3.8xlarge` (4 V100 GPUs), or `p4d.24xlarge` (8 A100 GPUs)
   - AMI: Deep Learning AMI (Ubuntu)
   - Storage: At least 100GB EBS volume

2. **Connect to your instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Clone and set up**:
   ```bash
   git clone https://github.com/yourusername/SimpleFoundation.git
   cd SimpleFoundation
   python setup.py
   ```

4. **Start the server with screen** (to keep it running after disconnecting):
   ```bash
   screen -S model-server
   bash scripts/serve.sh
   ```
   
   Press `Ctrl+A` then `D` to detach from the screen.

5. **Set up a second screen for the UI**:
   ```bash
   screen -S demo-ui
   bash scripts/demo.sh
   ```

6. **Configure security groups** to allow traffic on ports 8000 (API) and 8501 (UI).

#### SageMaker

For a more managed solution:

1. **Package the model as a SageMaker model**:
   - Create a `model.tar.gz` with your model files
   - Upload to S3

2. **Create a SageMaker endpoint**:
   - Use the SageMaker SDK or console
   - Select an appropriate instance type with GPUs
   - Deploy the model

3. **Create an API Gateway** to expose your endpoint

### Google Cloud

#### Google Compute Engine

1. **Create a VM instance**:
   - Machine type: A2 (NVIDIA A100 GPUs) or N1 with T4/V100 GPUs
   - Boot disk: Deep Learning VM Image
   - Disk size: 100GB+

2. **Connect and deploy**:
   ```bash
   gcloud compute ssh your-instance-name
   git clone https://github.com/yourusername/SimpleFoundation.git
   cd SimpleFoundation
   python setup.py
   ```

3. **Use tmux for persistent sessions**:
   ```bash
   tmux new -s model-server
   bash scripts/serve.sh
   ```
   
   Press `Ctrl+B` then `D` to detach.

4. **Configure firewall rules** to allow traffic on ports 8000 and 8501.

### Azure

#### Azure VM

1. **Create an Azure VM**:
   - Size: NC-series (NVIDIA GPUs)
   - Image: Data Science Virtual Machine for Linux (Ubuntu)

2. **Connect and deploy**:
   ```bash
   ssh username@your-vm-ip
   git clone https://github.com/yourusername/SimpleFoundation.git
   cd SimpleFoundation
   python setup.py
   ```

3. **Configure network security group** to allow traffic on ports 8000 and 8501.

### Lambda Labs

Lambda Labs offers cost-effective GPU instances:

1. **Create an instance**:
   - Select an appropriate GPU configuration (A100, H100, etc.)
   - Choose Ubuntu as the operating system

2. **SSH into your instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Clone and set up**:
   ```bash
   git clone https://github.com/yourusername/SimpleFoundation.git
   cd SimpleFoundation
   python setup.py
   ```

4. **Start the services**:
   ```bash
   screen -S model-server
   bash scripts/serve.sh
   
   # In a new screen
   screen -S demo-ui
   bash scripts/demo.sh
   ```

### Vast.ai

Vast.ai offers marketplace GPU rentals:

1. **Create an instance**:
   - Filter by GPU type and memory
   - Choose Ubuntu image

2. **Connect and deploy**:
   ```bash
   ssh -p port -i your-key.pem user@your-instance-ip
   git clone https://github.com/yourusername/SimpleFoundation.git
   cd SimpleFoundation
   python setup.py
   ```

3. **Start services**:
   ```bash
   screen -S model-server
   bash scripts/serve.sh
   
   # In a new screen
   screen -S demo-ui
   bash scripts/demo.sh
   ```

4. **Set up port forwarding** to access the services remotely.

## Docker Deployment

For containerized deployment:

1. **Create a Dockerfile** in the project root:
   ```Dockerfile
   FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
   
   # Install dependencies
   RUN apt-get update && apt-get install -y \
       python3 \
       python3-pip \
       git \
       wget \
       && rm -rf /var/lib/apt/lists/*
   
   # Set working directory
   WORKDIR /app
   
   # Copy requirements and install dependencies
   COPY requirements.txt .
   RUN pip3 install --no-cache-dir -r requirements.txt
   
   # Copy project files
   COPY . .
   
   # Port for API and UI
   EXPOSE 8000 8501
   
   # Entry point script
   COPY docker-entrypoint.sh /usr/local/bin/
   RUN chmod +x /usr/local/bin/docker-entrypoint.sh
   ENTRYPOINT ["docker-entrypoint.sh"]
   ```

2. **Create an entrypoint script**:
   ```bash
   #!/bin/bash
   # docker-entrypoint.sh
   
   if [ "$1" = "api" ]; then
       # Start API server
       python -m inference.server.model_server --model_path $MODEL_PATH --host 0.0.0.0 --port 8000
   elif [ "$1" = "ui" ]; then
       # Start UI
       streamlit run inference/demo/app.py
   elif [ "$1" = "train" ]; then
       # Run training
       python -m training.trainers.model_trainer --config $CONFIG_PATH --train_file $TRAIN_FILE
   else
       # Default: run both API and UI
       python -m inference.server.model_server --model_path $MODEL_PATH --host 0.0.0.0 --port 8000 &
       streamlit run inference/demo/app.py
   fi
   ```

3. **Build the Docker image**:
   ```bash
   docker build -t simplefoundation:latest .
   ```

4. **Run the container**:
   ```bash
   # Run both API and UI
   docker run --gpus all -p 8000:8000 -p 8501:8501 \
       -e MODEL_PATH=/app/training/checkpoints/final \
       -v /path/to/model:/app/training/checkpoints \
       simplefoundation:latest
   
   # Run only API
   docker run --gpus all -p 8000:8000 \
       -e MODEL_PATH=/app/training/checkpoints/final \
       -v /path/to/model:/app/training/checkpoints \
       simplefoundation:latest api
   ```

5. **For Docker Compose**, create a `docker-compose.yml`:
   ```yaml
   version: '3'
   services:
     api:
       image: simplefoundation:latest
       command: api
       ports:
         - "8000:8000"
       environment:
         - MODEL_PATH=/app/training/checkpoints/final
       volumes:
         - ./training/checkpoints:/app/training/checkpoints
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
     
     ui:
       image: simplefoundation:latest
       command: ui
       ports:
         - "8501:8501"
       environment:
         - API_URL=http://api:8000
       depends_on:
         - api
   ```

   Run with:
   ```bash
   docker-compose up -d
   ```

## Scaling for Production

For production environments:

1. **Load balancing**:
   - Deploy multiple inference servers
   - Use NGINX or HAProxy as a load balancer
   - Consider Kubernetes for orchestration

2. **Horizontal scaling**:
   ```
   Client -> Load Balancer -> Multiple Inference Servers
                           -> Redis Cache
   ```

3. **Caching common requests**:
   - Implement Redis for response caching
   - Set appropriate TTL values

4. **Use Kubernetes** for orchestration:
   - Create deployment and service manifests
   - Use GPU node pools or node selectors
   - Implement autoscaling

5. **Advanced logging and monitoring**:
   - Integrate Prometheus and Grafana
   - Use ELK stack for log aggregation

## Monitoring and Maintenance

To keep your deployment running smoothly:

1. **Set up monitoring**:
   ```bash
   # Install Prometheus client
   pip install prometheus-client
   
   # Add instrumentation to your code
   # See inference/server/model_server.py for implementation
   ```

2. **Create health checks**:
   - Use the `/health` endpoint
   - Set up regular pings

3. **Implement logging**:
   - Log to a central location
   - Set up log rotation

4. **Update model periodically**:
   - Create a CI/CD pipeline for model updates
   - Test new models before deployment
   - Keep backups of previous models

5. **Resource monitoring**:
   - Monitor GPU usage with `nvidia-smi`
   - Set up alerts for high resource usage
   - Implement graceful degradation when resources are constrained

For any questions or issues, please open an issue on the GitHub repository or contact the maintainers.

# Amazon SageMaker Model Loading Performance Benchmark

This project benchmarks and compares two different model loading approaches on Amazon SageMaker using DJL (Deep Java Library) for HuggingFace models:

1. **Standard Model Loading** - Traditional sequential loading process
2. **Fast Model Loading** - Optimized streaming approach with pre-sharding

## What This Project Demonstrates

### Standard Model Loading Process
- Provisions new compute instances
- Downloads container image
- Downloads model artifacts from S3 to disk
- Loads model artifacts to host memory (CPU/RAM)
- Prepares model for GPU loading
- Finally loads model onto GPU

### Fast Model Loading Process
- **Weight Streaming** - Directly streams model weights from S3 to GPU memory
- **Pre-sharding** - Model is sharded in advance into uniform chunks for optimal loading
- **Parallel Loading** - Concurrent loading of model weights
- **Bypasses Sequential Steps** - Eliminates time-consuming intermediate steps

## Prerequisites

- Python 3.11+
- AWS credentials configured
- Amazon SageMaker execution role with appropriate permissions
- HuggingFace account and token (for gated models)
- S3 bucket for storing optimized model weights (fast loading only)

## Setup Instructions

### 1. Install UV Package Manager and Setup Environment

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create and activate virtual environment
uv venv && source .venv/bin/activate && uv pip sync pyproject.toml

# Set UV environment variable
export UV_PROJECT_ENVIRONMENT=.venv

# Add additional dependencies
uv add zmq

# Install Jupyter kernel (optional, for notebook usage)
python -m ipykernel install --user --name=.venv --display-name="Python (uv env)"
```

### 2. Configuration Setup

#### Edit `config.yaml`

Update the configuration file with your specific settings:

```yaml
# Standard deployment configuration
standard_deployment_info:
  hf_model_id: meta-llama/Llama-3.1-8B  # Your HuggingFace model ID
  sagemaker_exec_role: arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXXX
  image_uri: 763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128
  instance_type: ml.g5.2xlarge  # Adjust based on model size
  container_startup_health_check_timeout: 600
  use_hf_token: yes
  container_env_vars:
    "SERVING_FAIL_FAST": "true"
    "OPTION_ASYNC_MODE": "true"
    "OPTION_ROLLING_BATCH": "disable"
    "HF_MODEL_ID": meta-llama/Llama-3.1-8B  # Match hf_model_id
    "OPTION_MAX_MODEL_LEN": "4096"
    "OPTION_TENSOR_PARALLEL_DEGREE": "max"
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service"

# Fast model loading configuration
fast_model_loading_info:
  hf_model_id: meta-llama/Llama-3.1-8B  # Your HuggingFace model ID
  bucket: your-s3-bucket-name  # S3 bucket for storing optimized weights
  sagemaker_exec_role: arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXXX
  image_uri: 763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128
  instance_type: ml.g5.2xlarge  # Adjust based on model size
  container_startup_health_check_timeout: 600
  instance_count: 1
  use_hf_token: yes
  container_env_vars:
    "SERVING_FAIL_FAST": "true"
    "OPTION_ASYNC_MODE": "true"
    "OPTION_ROLLING_BATCH": "disable"
    "HF_MODEL_ID": meta-llama/Llama-3.1-8B
    "OPTION_MAX_MODEL_LEN": "4096"
    "OPTION_TENSOR_PARALLEL_DEGREE": "max"
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service"
  sample_input:
    "inputs": "What is the capital of France?"
    "parameters":
      "max_new_tokens": 100
      "temperature": 0.7
      "do_sample": True
  sample_output:
    - "generated_text": "The capital of France is Paris."
```

#### Required Configuration Updates:

1. **AWS Account Information:**
   - Replace `YOUR_ACCOUNT` with your AWS account ID in the SageMaker execution role ARN
   - Update the role name if different from the example

2. **S3 Bucket (Fast Loading Only):**
   - Replace `your-s3-bucket-name` with your S3 bucket name
   - Ensure the bucket exists and your SageMaker role has read/write access

3. **Model Selection:**
   - Update `hf_model_id` to the HuggingFace model you want to deploy
   - Ensure the model ID matches in both `hf_model_id` and `container_env_vars.HF_MODEL_ID`

4. **Instance Type:**
   - Choose appropriate instance type based on model size:
     - `ml.g5.xlarge` - For smaller models (7B parameters)
     - `ml.g5.2xlarge` - For medium models (8B-13B parameters)
     - `ml.g5.4xlarge` or larger - For large models (30B+ parameters)

#### Setup Environment Variables

Create `.env` files in both directories with your HuggingFace token:

```bash
# Create .env in root directory
echo "HUGGING_FACE_HUB_TOKEN='your_hf_token_here'" > .env

# Create .env in standard_deployment directory
echo "HUGGING_FACE_HUB_TOKEN='your_hf_token_here'" > standard_deployment/.env
```

**To get your HuggingFace token:**
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token or copy an existing one
3. Replace `your_hf_token_here` with your actual token

## Usage

### Running Standard Deployment Benchmark

```bash
python standard_deployment/standard_sm_deployment.py
```

This will:
- Deploy the model using traditional loading approach
- Measure deployment time
- Test the endpoint with a sample request
- Generate a deployment summary file

### Running Fast Model Loading Benchmark

```bash
python fast_model_loading/fast_model_loading.py
```

This will:
- Optimize the model for fast loading (pre-sharding)
- Deploy using the optimized approach
- Measure all phases of deployment
- Test the endpoint with a sample request
- Generate timing results and performance summary

### Performance Comparison

Both scripts will output detailed timing metrics:
- **Initialization time**
- **Model preparation time**
- **Deployment time**
- **Total execution time**

Expected performance improvements with Fast Model Loading:
- 2-5x faster deployment times
- Reduced cold start latency
- More efficient resource utilization

## Supported Models

This benchmark supports any HuggingFace model compatible with DJL inference, including:

- **Text Generation Models:** Llama, Mistral, CodeLlama, etc.
- **Gated Models:** Models requiring HuggingFace authentication
- **Large Language Models:** Models up to 70B parameters (with appropriate instance types)

## Instance Type Recommendations

| Model Size | Recommended Instance Type | Memory Requirements |
|-----------|---------------------------|-------------------|
| 7B-8B parameters | ml.g5.xlarge | 24GB GPU memory |
| 13B-15B parameters | ml.g5.2xlarge | 48GB GPU memory |
| 30B-34B parameters | ml.g5.4xlarge | 96GB GPU memory |
| 65B-70B parameters | ml.g5.12xlarge+ | 192GB+ GPU memory |

## Troubleshooting

### Common Issues

1. **Missing HuggingFace Token:**
   ```
   Warning: HUGGING_FACE_HUB_TOKEN not found in environment variables
   ```
   **Solution:** Create `.env` file with your HuggingFace token

2. **Insufficient Instance Capacity:**
   ```
   Cannot provide requested instance type
   ```
   **Solution:** Try a different AWS region or request quota increase

3. **Model Loading Timeout:**
   ```
   Container startup health check timeout
   ```
   **Solution:** Increase `container_startup_health_check_timeout` in config.yaml

4. **S3 Bucket Access Issues (Fast Loading):**
   ```
   Access denied to S3 bucket
   ```
   **Solution:** Verify bucket permissions and SageMaker execution role policies

### Performance Optimization Tips

1. **Use Latest Container Images:** Ensure you're using the most recent DJL inference container
2. **Optimize Tensor Parallelism:** Adjust `OPTION_TENSOR_PARALLEL_DEGREE` based on instance GPU count
3. **Monitor Resource Usage:** Use CloudWatch to monitor CPU, memory, and GPU utilization
4. **Regional Considerations:** Deploy in the same region as your S3 bucket for optimal performance

## Cleanup

**Important:** Remember to delete endpoints after testing to avoid charges:

```python
# In your Python session after running the scripts
predictor.delete_endpoint()
```

Or use AWS CLI:
```bash
aws sagemaker delete-endpoint --endpoint-name your-endpoint-name
```

## Project Structure

```
   README.md                           # This file
   config.yaml                         # Configuration for both deployment types
   pyproject.toml                      # Python dependencies
   utils.py                           # Utility functions
   .env.example                       # Environment variables template
   fast_model_loading/
      fast_model_loading.py          # Fast loading benchmark script
      .env.example                   # Environment template
   standard_deployment/
       standard_sm_deployment.py      # Standard deployment benchmark script
       .env.example                   # Environment template
```

## Links and Resources

- [Amazon SageMaker Fast Model Loading Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-fast-model-loading.html)
- [DJL (Deep Java Library) Documentation](https://docs.djl.ai/)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/)

## Performance Results

After running both benchmarks, you can compare:
- Deployment times
- Resource utilization
- Cost efficiency
- Inference latency

The fast model loading approach typically shows significant improvements in deployment speed, especially for larger models, making it ideal for production workloads requiring quick scaling and reduced cold start times.
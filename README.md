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
# this is the configuration for the standard deployment time
standard_deployment_info:
  hf_model_id: meta-llama/Llama-3.1-8B-Instruct
  sagemaker_exec_role: 
  image_uri: 763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126
  container_env_vars:
    "HF_MODEL_ID": meta-llama/Llama-3.1-8B-Instruct
    "OPTION_ROLLING_BATCH": "lmi-dist"
    "OPTION_MAX_ROLLING_BATCH_SIZE": "16"
    "OPTION_TENSOR_PARALLEL_DEGREE": "max"
    "OPTION_DTYPE": "fp16"
    "OPTION_MAX_MODEL_LEN": "6000"
  instance_type: ml.g5.2xlarge
  container_startup_health_check_timeout: 600
  use_hf_token: yes

# This is the configuration for fast model loading
fast_model_loading_info:
  hf_model_id: meta-llama/Llama-3.1-8B-Instruct
  bucket: 
  sagemaker_exec_role: 
  image_uri: 763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126
  container_env_vars:
    "HF_MODEL_ID": meta-llama/Llama-3.1-8B-Instruct
    "OPTION_ROLLING_BATCH": "lmi-dist"
    "OPTION_MAX_ROLLING_BATCH_SIZE": "16"
    "OPTION_TENSOR_PARALLEL_DEGREE": "max"
    "OPTION_DTYPE": "fp16"
    "OPTION_MAX_MODEL_LEN": "6000"
  instance_type: ml.g5.2xlarge
  container_startup_health_check_timeout: 600
  sample_input: 
    "inputs": "What is the capital of France?"
    "parameters": 
        "max_new_tokens": 100
        "temperature": 0.7
        "do_sample": True
  sample_output:
  - "generated_text": "The capital of France is Paris."
  instance_count: 1
  use_hf_token: yes
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

## Results
Running both deployment formats will generate a `txt` and `json` files that will contain information on time taken to optimize and deploy models and the difference in the two. Using the standard `llama3.1 8b instruct` model deployed on a `ml.g5.2xlarge`, the deployment time dropped from 10 minutes 32 seconds to 7 minutes 30 seconds. View the results below:

### Using Standard Deployment

```txt

============================================================
🎉 DEPLOYMENT SUMMARY
============================================================
📊 TIMING METRICS:
  • Setup time:      0s
  • Deployment time: 10m 32s
  • Total time:      10m 33s

📋 DEPLOYMENT DETAILS:
  • Endpoint name:   
  • Instance type:   ml.g5.2xlarge
  • Model:           meta-llama/Llama-3.1-8B-Instruct
  • Container:       763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126
  • Completed at:    2025-08-27 22:01:06

==================================================
EXAMPLE USAGE FOR VISION-LANGUAGE TASKS:
==================================================
# For text + image inputs, use this format:
payload = {
    "inputs": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YOUR_BASE64_IMAGE"}}
            ]
        }
    ],
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.7
    }
}
response = predictor.predict(payload)
```

### Using Fast Model Loading

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "instance_type": "ml.g5.2xlarge",
  "initialization_time": 0.014258765993872657,
  "builder_creation_time": 0.0004756310081575066,
  "optimization_time": 1116.8918553480034,
  "build_time": 1.3685317259951262,
  "deployment_time": 450.0898740720004,
  "testing_time": 0.14585125500161666,
  "total_execution_time": 1568.5111365079938,
  "timestamp": "2025-08-27 21:41:41"
}
```

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
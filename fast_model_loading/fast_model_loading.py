"""
Amazon SageMaker fast model loader represents a significant advancement
in deploying large language models (LLMs) for inference. As LLMs continue to grow
in size and complexity, with some models requiring hundreds of gigabytes of memory, 
the traditional model loading process is a major bottleneck in deploying and scaling
these inference workloads.
This python file demonstrates how you can leverage the fast model loader to dramatically
improve the model loading times. This feature works by directly streaming the model weights from
Amazon S3 to GPU accelerators, bypassing the typical sequential loading steps that contribute
to the deployment latency. We will look at the following steps:
1. Optimizing the model for streaming using the ModelBuilder
2. Configure tensor parallelism for distributed inference
3. Deploy the optimized model endpoint to SageMaker
4. Test the deployment and measure the time taken to deploy the optimized model
In this case, the Fast Model Loader does the following:
1. Weight streaming - Directly streams the model weights from S3 to GPU memory
2. Model sharding for streaming - Pre-shards the model in uniform chunks for optimal loading
"""
import os
import sys
import boto3
import time
from sagemaker import Session
from sagemaker import get_execution_role
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
import logging
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

# Load environment variables from .env file
load_dotenv()

# Change this line to look in the parent directory
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_dir, 'config.yaml')
    config_data: Dict = load_config(config_path)
    print(f"Config data that will be used in this deployment: {json.dumps(config_data, indent=4)}")
except Exception as e:
    print(f"An error occurred while loading the config data: {e}")

# Let's get the configuration for fast model loading information on sagemaker
FAST_MODEL_LOADING_INFO: Dict = config_data['fast_model_loading_info']

# Track overall execution time
script_start_time = time.perf_counter()

print("=== SageMaker Fast Model Loader Performance Analysis ===")

# Initialize the sagemaker role
print("Initializing SageMaker session and role...")
init_start_time = time.perf_counter()

role = FAST_MODEL_LOADING_INFO.get('sagemaker_exec_role')
region = boto3.Session().region_name
sess = Session()

# In this case, we will use a default bucket to store the model weights
bucket = FAST_MODEL_LOADING_INFO.get('bucket')

init_end_time = time.perf_counter()
print(f"✓ SageMaker initialization completed in {init_end_time - init_start_time:.2f} seconds")

# Model configuration
HF_MODEL: str = FAST_MODEL_LOADING_INFO.get('hf_model_id')
INSTANCE_TYPE: str = FAST_MODEL_LOADING_INFO.get('instance_type')

# Environment variables for the model
env_vars = FAST_MODEL_LOADING_INFO.get('container_env_vars', {}).copy()

# Always try to load HF token for gated models
hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
if hf_token:
    env_vars['HUGGING_FACE_HUB_TOKEN'] = hf_token
    env_vars['HF_TOKEN'] = hf_token  # Some containers expect HF_TOKEN instead
    print(f"✓ Hugging Face token loaded from environment")
    print(f"environment variables: {env_vars}")
else:
    print(f"⚠️ Warning: HUGGING_FACE_HUB_TOKEN not found in environment variables")
    print(f"   Please create a .env file with HUGGING_FACE_HUB_TOKEN='your_token_here'")
    print(f"   Or set the environment variable before running this script")
    # Set empty token to avoid undefined variable errors
    hf_token = ""
# Container image URI
# Latest image URI is not working so going to test with
# the prior version to check for deployment time improvements
# DJL_IMAGE_URI: str = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128"
LMI_IMAGE_URI = FAST_MODEL_LOADING_INFO.get('image_uri').format(region=region)
print(f"Using LMI container: {LMI_IMAGE_URI}")

# Step 1: Create the model builder class
print("\n=== Step 1: Creating Model Builder ===")
builder_start_time = time.perf_counter()

sample_input = FAST_MODEL_LOADING_INFO.get('sample_input')
print(f"going to use the following sample input: {sample_input}")

sample_output = FAST_MODEL_LOADING_INFO.get('sample_output')
print(f"going to use the following sample output: {sample_output}")


mb = ModelBuilder(
    model=HF_MODEL,
    role_arn=role,
    schema_builder=SchemaBuilder(sample_input=sample_input, sample_output=sample_output), 
    env_vars=env_vars,
    image_uri=LMI_IMAGE_URI,
)

builder_end_time = time.perf_counter()
print(f"✓ Model Builder creation completed in {builder_end_time - builder_start_time:.2f} seconds")

# Step 2: Optimize the model for fast loading
print("\n=== Step 2: Optimizing Model for Fast Loading ===")
optimize_start_time = time.perf_counter()

output_path = f"s3://{bucket}/sharding"
print(f"Going to optimize the model builder object...")
print(f"Output path: {output_path}")

mb.optimize(
    instance_type=INSTANCE_TYPE,
    output_path=output_path,
    sharding_config={
        # 🟢 This is the key line – use a modern LMI DLC for the OPTIMIZATION job
        "Image": LMI_IMAGE_URI,
        "OverrideEnvironment": {
            "HF_MODEL_ID": HF_MODEL,
            "OPTION_TENSOR_PARALLEL_DEGREE": "1",
            # Optional niceties
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # vLLM V1 is default on LMI v15, but you can pin it:
            # "VLLM_USE_V1": "1",
            "HUGGING_FACE_HUB_TOKEN": hf_token,
        },
    },
)

optimize_end_time = time.perf_counter()
print(f"✓ Model optimization completed in {optimize_end_time - optimize_start_time:.2f} seconds")

# Step 3: Build the model
print("\n=== Step 3: Building Model ===")
build_start_time = time.perf_counter()

final_model = mb.build()

build_end_time = time.perf_counter()
print(f"✓ Model build completed in {build_end_time - build_start_time:.2f} seconds")

# Step 4: Deploy the model endpoint
print("\n=== Step 4: Deploying Model Endpoint ===")
deploy_start_time = time.perf_counter()

print(f"Deploying model to {INSTANCE_TYPE} instance...")
predictor = final_model.deploy(
    instance_type=INSTANCE_TYPE,
    initial_instance_count=FAST_MODEL_LOADING_INFO.get('instance_count')
)

deploy_end_time = time.perf_counter()
print(f"✓ Model deployment completed in {deploy_end_time - deploy_start_time:.2f} seconds")

# Step 5: Test the deployed endpoint
print("\n=== Step 5: Testing Deployed Endpoint ===")
test_start_time = time.perf_counter()

try:
    # Test inference latency
    test_input = "Hello, how are you today?"
    print(f"Testing with input: '{test_input}'")
    # Measure inference time
    inference_start_time = time.perf_counter()
    response = predictor.predict(test_input)
    inference_end_time = time.perf_counter()
    print(f"✓ Inference completed in {inference_end_time - inference_start_time:.3f} seconds")
    print(f"Response: {response}")
except Exception as e:
    print(f"❌ Error during inference test: {str(e)}")

test_end_time = time.perf_counter()
print(f"✓ Endpoint testing completed in {test_end_time - test_start_time:.2f} seconds")

# Calculate and display total execution time
script_end_time = time.perf_counter()
total_execution_time = script_end_time - script_start_time

print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Initialization:      {init_end_time - init_start_time:>8.2f} seconds")
print(f"Model Builder:       {builder_end_time - builder_start_time:>8.2f} seconds")
print(f"Model Optimization:  {optimize_end_time - optimize_start_time:>8.2f} seconds")
print(f"Model Build:         {build_end_time - build_start_time:>8.2f} seconds")
print(f"Model Deployment:    {deploy_end_time - deploy_start_time:>8.2f} seconds")
print(f"Endpoint Testing:    {test_end_time - test_start_time:>8.2f} seconds")
print("-" * 40)
print(f"TOTAL EXECUTION:     {total_execution_time:>8.2f} seconds")
print("="*60)

# Optional: Save timing results to a file
timing_results = {
    "model": HF_MODEL,
    "instance_type": INSTANCE_TYPE,
    "initialization_time": init_end_time - init_start_time,
    "builder_creation_time": builder_end_time - builder_start_time,
    "optimization_time": optimize_end_time - optimize_start_time,
    "build_time": build_end_time - build_start_time,
    "deployment_time": deploy_end_time - deploy_start_time,
    "testing_time": test_end_time - test_start_time,
    "total_execution_time": total_execution_time,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Optionally save results to S3 or local file
try:
    import json
    results_file = f"timing_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(timing_results, f, indent=2)
    print(f"\n✓ Timing results saved to {results_file}")
except Exception as e:
    print(f"⚠️  Could not save timing results: {str(e)}")

print(f"\nEndpoint name: {predictor.endpoint_name}")
print("Note: Remember to delete the endpoint when done to avoid charges:")
print(f"predictor.delete_endpoint()")
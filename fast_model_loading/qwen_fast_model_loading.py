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
import boto3
import time
from sagemaker import Session
from sagemaker import get_execution_role
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
import logging

# Track overall execution time
script_start_time = time.perf_counter()

print("=== SageMaker Fast Model Loader Performance Analysis ===")

# Initialize the sagemaker role
print("Initializing SageMaker session and role...")
init_start_time = time.perf_counter()

role = get_execution_role()
region = boto3.Session().region_name
sess = Session()

# In this case, we will use a default bucket to store the model weights
bucket = sess.default_bucket()

init_end_time = time.perf_counter()
print(f"✓ SageMaker initialization completed in {init_end_time - init_start_time:.2f} seconds")

# Model configuration
HF_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTANCE_TYPE: str = "ml.g5.2xlarge"

# Environment variables for the model
env_vars = {
    "SERVING_FAIL_FAST": "true",
    "OPTION_ASYNC_MODE": "true",
    "OPTION_ROLLING_BATCH": "disable",
    "HF_MODEL_ID": HF_MODEL,
    "OPTION_MAX_MODEL_LEN": "4096",
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
}

# Container image URI
# Latest image URI is not working so going to test with
# the prior version to check for deployment time improvements
# DJL_IMAGE_URI: str = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128"
LMI_IMAGE_URI = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128"
print(f"Using LMI container: {LMI_IMAGE_URI}")

# Step 1: Create the model builder class
print("\n=== Step 1: Creating Model Builder ===")
builder_start_time = time.perf_counter()

"""
Next, we will create a model building class to prepare and package the model inference
components. In this example, we will use the Qwen/Qwen2.5-VL-7B-Instruct model from
hugging face on SageMaker
"""

sample_input = {
    "inputs": "What is the capital of France?",
    "parameters": {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True
    }
}

sample_output = [{"generated_text": "The capital of France is Paris."}]


mb = ModelBuilder(
    model=HF_MODEL,
    role_arn=role,
    sagemaker_session=sess,
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
print(f"Going to optimize the Qwen model builder object...")
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
    initial_instance_count=1
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
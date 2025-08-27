import os
import sys
import time
import json
import boto3
import sagemaker
from datetime import datetime, timedelta
from sagemaker.djl_inference import DJLModel
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

def format_duration(seconds):
    """Format duration in a human-readable way"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    else:
        return f"{remaining_seconds}s"

STANDARD_DEPLOYMENT_INFO: Dict = config_data['standard_deployment_info']
MODEL_ID: str = STANDARD_DEPLOYMENT_INFO.get('hf_model_id')
print(f"Going to deploy model: {MODEL_ID} in the standard deployment way...")
# Start overall timing
overall_start_time = time.time()
print(f"🚀 Starting SageMaker deployment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Setup phase timing
setup_start_time = time.time()

# Get execution role
try:
    iam = boto3.client('iam')
    role = STANDARD_DEPLOYMENT_INFO.get('sagemaker_exec_role')
    print(f"Using IAM role: {role}")
except Exception as e:
    print(f"An error occurred while getting a sagemaker execution role: {e}")

# Get current region for container URI
session = boto3.Session()
region = session.region_name

# Use the latest LMI v15 container explicitly
latest_lmi_image = STANDARD_DEPLOYMENT_INFO.get('image_uri').format(region=region)
print(f"Using LMI container: {latest_lmi_image}")

# LMI environment configuration 
lmi_env = STANDARD_DEPLOYMENT_INFO.get('container_env_vars', {}).copy()

# Always try to load HF token for gated models
hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
if hf_token:
    lmi_env['HUGGING_FACE_HUB_TOKEN'] = hf_token
    lmi_env['HF_TOKEN'] = hf_token  # Some containers expect HF_TOKEN instead
    print(f"✓ Hugging Face token loaded from environment")
else:
    print(f"⚠️ Warning: HUGGING_FACE_HUB_TOKEN not found in environment variables")
    print(f"   Please create a .env file with HUGGING_FACE_HUB_TOKEN='your_token_here'")
    print(f"   Or set the environment variable before running this script")

print("Environment configuration:")
for key, value in lmi_env.items():
    print(f"  {key}: {value}")

# Create DJL Model for Large Model Inference
model = DJLModel(
    model_id=MODEL_ID,
    env=lmi_env,
    role=role,
    image_uri=latest_lmi_image,
    predictor_cls=sagemaker.predictor.Predictor
)

setup_duration = time.time() - setup_start_time
print(f"\n✅ Setup completed in {format_duration(setup_duration)}")
print(f"Created DJL Model with LMI v15 configuration")
print(f"Container: {latest_lmi_image}")

# Generate a unique endpoint name with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
endpoint_name = f"llama3-1-8b-{timestamp}"
print(f"\n🚀 Deploying model to endpoint: {endpoint_name}")
print("This may take 10-15 minutes...")
print("-" * 40)

# Model deployment timing
deployment_start_time = time.time()
print(f"Deployment started at: {datetime.now().strftime('%H:%M:%S')}")

# Deploy model to SageMaker Inference
try:
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=STANDARD_DEPLOYMENT_INFO.get('instance_type'),
        container_startup_health_check_timeout=STANDARD_DEPLOYMENT_INFO.get('container_startup_health_check_timeout'),
        endpoint_name=endpoint_name
    )
    deployment_duration = time.time() - deployment_start_time
    print(f"✅ Model successfully deployed to endpoint: {endpoint_name}")
    print(f"⏱️  Deployment completed in {format_duration(deployment_duration)}")
except Exception as e:
    deployment_duration = time.time() - deployment_start_time
    print(f"❌ Deployment failed after {format_duration(deployment_duration)}")
    print(f"Error: {str(e)}")
    raise

# Test the deployed model with a simple text request
print(f"\n🧪 Testing the deployed model...")
test_start_time = time.time()

try:
    # Test request for vision-language model
    test_payload = {
        "inputs": "Hello! What can you help me with?",
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    response = predictor.predict(test_payload)
    test_duration = time.time() - test_start_time
    print(f"✅ Test request successful! (Response time: {format_duration(test_duration)})")
    print(f"Response: {response}")
except Exception as e:
    test_duration = time.time() - test_start_time
    print(f"❌ Test request failed after {format_duration(test_duration)}: {str(e)}")
    print("The endpoint is deployed but may need a few more minutes to be ready.")

# Calculate total time
overall_duration = time.time() - overall_start_time
# Code to save deployment summary to a text file

# Deployment Summary
summary_content = f"""
{"="*60}
🎉 DEPLOYMENT SUMMARY
{"="*60}
📊 TIMING METRICS:
  • Setup time:      {format_duration(setup_duration)}
  • Deployment time: {format_duration(deployment_duration)}
  • Total time:      {format_duration(overall_duration)}

📋 DEPLOYMENT DETAILS:
  • Endpoint name:   {endpoint_name}
  • Instance type:   ml.g5.2xlarge
  • Model:           Qwen/Qwen2.5-VL-7B-Instruct
  • Container:       {latest_lmi_image}
  • Completed at:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{"="*50}
EXAMPLE USAGE FOR VISION-LANGUAGE TASKS:
{"="*50}
# For text + image inputs, use this format:
payload = {{
    "inputs": [
        {{
            "role": "user",
            "content": [
                {{"type": "text", "text": "What do you see in this image?"}},
                {{"type": "image_url", "image_url": {{"url": "data:image/jpeg;base64,YOUR_BASE64_IMAGE"}}}}
            ]
        }}
    ],
    "parameters": {{
        "max_new_tokens": 512,
        "temperature": 0.7
    }}
}}
response = predictor.predict(payload)
"""

# Save to file
filename = f"deployment_summary_{timestamp}.txt"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(summary_content)

print(f"📁 Deployment summary saved to: {filename}")
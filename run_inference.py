#!/usr/bin/env python3
"""
Interactive SageMaker Endpoint Invoker

This script prompts the user for an endpoint name and invokes it with a test payload.

Usage:
    python run_inference.py
"""

import boto3
import json
import time
import base64
from typing import Dict, Any, Optional
from datetime import datetime


class SageMakerEndpointInvoker:
    def __init__(self, region_name: str = None):
        """Initialize the SageMaker runtime client."""
        self.runtime = boto3.client('sagemaker-runtime', region_name=region_name)
        
    def create_text_payload(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Create a text-only payload for inference."""
        return {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
    
    def create_multimodal_payload(self, text_prompt: str, image_base64: str = None, max_tokens: int = 512) -> Dict[str, Any]:
        """Create a multimodal payload for vision-language models."""
        if image_base64 is None:
            # Create a simple test image (1x1 pixel red image) as base64
            image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        return {
            "inputs": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
    
    def invoke_endpoint(self, endpoint_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a SageMaker endpoint with the given payload."""
        try:
            start_time = time.time()
            
            response = self.runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Parse response
            response_body = response['Body'].read().decode('utf-8')
            
            return {
                'success': True,
                'response': json.loads(response_body),
                'inference_time': inference_time,
                'status_code': response['ResponseMetadata']['HTTPStatusCode']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inference_time': None,
                'status_code': None
            }
    
    def test_endpoint_health(self, endpoint_name: str) -> bool:
        """Test if an endpoint is healthy with a simple request."""
        simple_payload = self.create_text_payload("Hello", max_tokens=10)
        result = self.invoke_endpoint(endpoint_name, simple_payload)
        return result['success']


def print_header():
    """Print a nice header for the application."""
    print("\n" + "="*60)
    print("🚀 SageMaker Endpoint Invoker")
    print("="*60)
    print("This tool allows you to test SageMaker endpoints interactively")
    print("-"*60)


def get_user_input():
    """Get user input for endpoint configuration."""
    print("\n📋 ENDPOINT CONFIGURATION")
    print("-"*30)
    
    # Get endpoint name
    endpoint_name = input("Enter SageMaker endpoint name: ").strip()
    if not endpoint_name:
        print("❌ Error: Endpoint name cannot be empty")
        return None
    
    # Ask for test type
    print("\n🧪 TEST TYPE OPTIONS:")
    print("1. Text-only inference")
    print("2. Multimodal (text + image) inference")
    test_type_choice = input("Select test type (1 or 2) [default: 1]: ").strip() or "1"
    
    test_type = "text" if test_type_choice == "1" else "multimodal"
    
    # Get custom prompt or use default
    print("\n💬 PROMPT CONFIGURATION")
    use_custom = input("Use custom prompt? (y/n) [default: n]: ").strip().lower()
    
    if use_custom == 'y':
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            prompt = "Hello! Can you tell me about machine learning?"
            print(f"Using default prompt: {prompt}")
    else:
        prompt = "Hello! Can you tell me about machine learning?"
        print(f"Using default prompt: {prompt}")
    
    # Get max tokens
    max_tokens_input = input("\nMax tokens to generate [default: 512]: ").strip()
    try:
        max_tokens = int(max_tokens_input) if max_tokens_input else 512
    except ValueError:
        max_tokens = 512
        print(f"Invalid input, using default: {max_tokens}")
    
    # Get region (optional)
    region = input("\nAWS Region (press Enter to use default): ").strip() or None
    
    return {
        'endpoint_name': endpoint_name,
        'test_type': test_type,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'region': region
    }


def main():
    """Main function to run the interactive endpoint invoker."""
    print_header()
    
    while True:
        # Get user configuration
        config = get_user_input()
        if not config:
            continue
        
        # Initialize the invoker
        print("\n" + "="*60)
        print("🔄 INITIALIZING")
        print("="*60)
        
        try:
            invoker = SageMakerEndpointInvoker(region_name=config['region'])
            print("✅ SageMaker client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing SageMaker client: {e}")
            print("\nMake sure you have AWS credentials configured!")
            continue
        
        # Display configuration summary
        print("\n📊 CONFIGURATION SUMMARY")
        print("-"*30)
        print(f"📍 Endpoint: {config['endpoint_name']}")
        print(f"🧪 Test type: {config['test_type']}")
        print(f"💬 Prompt: {config['prompt'][:100]}{'...' if len(config['prompt']) > 100 else ''}")
        print(f"🔢 Max tokens: {config['max_tokens']}")
        if config['region']:
            print(f"🌍 Region: {config['region']}")
        
        # Confirm before proceeding
        confirm = input("\nProceed with this configuration? (y/n) [default: y]: ").strip().lower()
        if confirm == 'n':
            continue
        
        # Prepare payload
        print("\n" + "="*60)
        print("🚀 RUNNING INFERENCE")
        print("="*60)
        
        if config['test_type'] == 'text':
            payload = invoker.create_text_payload(config['prompt'], config['max_tokens'])
        else:
            payload = invoker.create_multimodal_payload(config['prompt'], None, config['max_tokens'])
        
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test endpoint health
        print("\n🏥 Checking endpoint health...")
        if not invoker.test_endpoint_health(config['endpoint_name']):
            print(f"❌ Endpoint {config['endpoint_name']} appears to be unhealthy or doesn't exist")
            print("Please check the endpoint name and try again.")
        else:
            print("✅ Endpoint is healthy")
            
            # Run inference
            print("\n🔄 Running inference...")
            result = invoker.invoke_endpoint(config['endpoint_name'], payload)
            
            # Display results
            print("\n" + "="*60)
            print("📊 RESULTS")
            print("="*60)
            
            if result['success']:
                print(f"✅ SUCCESS!")
                print(f"⏱️  Inference time: {result['inference_time']:.3f} seconds")
                print(f"📊 Status code: {result['status_code']}")
                print("\n📝 Response:")
                print("-"*30)
                
                # Pretty print the response
                response_str = json.dumps(result['response'], indent=2)
                if len(response_str) > 1000:
                    print(response_str[:1000] + "\n... (truncated)")
                else:
                    print(response_str)
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"inference_result_{config['endpoint_name']}_{timestamp}.json"
                
                with open(results_file, 'w') as f:
                    json.dump({
                        'configuration': config,
                        'timestamp': datetime.now().isoformat(),
                        'result': result
                    }, f, indent=2)
                
                print(f"\n💾 Full results saved to: {results_file}")
                
            else:
                print(f"❌ FAILED")
                print(f"Error: {result['error']}")
                print("\nPossible issues:")
                print("• Check if the endpoint name is correct")
                print("• Verify you have permission to invoke this endpoint")
                print("• Ensure the endpoint is in service")
                print("• Check if the payload format matches the endpoint's expected input")
        
        # Ask if user wants to test another endpoint
        print("\n" + "="*60)
        another = input("Test another endpoint? (y/n) [default: n]: ").strip().lower()
        if another != 'y':
            break
    
    print("\n👋 Thank you for using SageMaker Endpoint Invoker!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your AWS configuration and try again.")
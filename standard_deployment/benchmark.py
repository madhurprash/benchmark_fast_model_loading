#!/usr/bin/env python3
import boto3
import json
import time
import statistics
from datetime import datetime

# Configuration
ENDPOINT_NAME = "qwen25-vl-7b-2025-08-25-22-46-59"
REGION = "us-west-2"
NUM_RUNS = 10

# Initialize the SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name=REGION)

# Test queries
queries = [
    "Hello, what can you help me with?",
    "Explain quantum physics in simple terms.",
    "Write a short story about a robot.",
    "What are the benefits of renewable energy?",
    "How do neural networks work?",
    "Describe the water cycle.",
    "What is machine learning?",
    "Tell me about climate change.",
    "How does photosynthesis work?",
    "What are the principles of good software design?"
]

# Test different payload formats to find working one
payload_formats = [
    # Format 1: Simple inputs
    lambda query: {
        "inputs": query,
        "parameters": {"max_new_tokens": 512, "temperature": 0.1}
    },
    # Format 2: Alternative format
    lambda query: {
        "text": query,
        "max_new_tokens": 512,
        "temperature": 0.1
    },
    # Format 3: Chat format
    lambda query: {
        "inputs": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ]
            }
        ],
        "parameters": {"max_new_tokens": 512, "temperature": 0.1}
    }
]

def test_endpoint(payload):
    """Test endpoint with a payload and measure latency."""
    try:
        body = json.dumps(payload).encode('utf-8')
        start_time = time.perf_counter()
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=body
        )
        end_time = time.perf_counter()
        # Read response
        result = json.loads(response['Body'].read().decode('utf-8'))
        latency = end_time - start_time
        return True, latency, result
    except Exception as e:
        return False, None, str(e)

def find_working_format():
    """Find which payload format works for this endpoint."""
    test_query = "Hello, test message."
    for i, format_func in enumerate(payload_formats, 1):
        print(f"Testing payload format {i}...")
        payload = format_func(test_query)
        success, latency, result = test_endpoint(payload)
        if success:
            print(f"✅ Format {i} works! Latency: {latency:.3f}s")
            print(f"Sample response: {str(result)[:200]}...")
            return format_func
        else:
            print(f"❌ Format {i} failed: {result}")
    return None

def run_benchmark():
    """Run the latency benchmark."""
    print(f"Starting benchmark for endpoint: {ENDPOINT_NAME}")
    print(f"Testing {NUM_RUNS} queries...")
    print("=" * 60)
    # Find working format
    working_format = find_working_format()
    if not working_format:
        print("❌ No working payload format found!")
        return
    
    print("\n" + "=" * 60)
    print("Running benchmark...")
    latencies = []
    results_log = []
    
    for i in range(NUM_RUNS):
        query = queries[i % len(queries)]  # Cycle through queries
        payload = working_format(query)
        print(f"\nTest {i+1}/{NUM_RUNS}: '{query[:50]}...'")
        success, latency, result = test_endpoint(payload)
        if success:
            latencies.append(latency)
            print(f"✅ Success - Latency: {latency:.3f}s")
            # Log result
            results_log.append({
                "test_number": i + 1,
                "query": query,
                "latency_seconds": latency,
                "response_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
                "timestamp": datetime.now().isoformat()
            })
        else:
            print(f"❌ Failed: {result}")
    if not latencies:
        print("❌ No successful requests!")
        return
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT_NAME}")
    print(f"Total successful requests: {len(latencies)}/{NUM_RUNS}")
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"Median latency: {median_latency:.3f}s")
    print(f"Min latency: {min_latency:.3f}s")
    print(f"Max latency: {max_latency:.3f}s")
    print(f"Standard deviation: {std_dev:.3f}s")
    print(f"Requests per second: {1/avg_latency:.2f}")
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write("SAGEMAKER ENDPOINT BENCHMARK RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Endpoint: {ENDPOINT_NAME}\n")
        f.write(f"Region: {REGION}\n")
        f.write(f"Total tests: {NUM_RUNS}\n")
        f.write(f"Successful requests: {len(latencies)}\n")
        f.write(f"Failed requests: {NUM_RUNS - len(latencies)}\n\n")
        
        f.write("LATENCY STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average latency: {avg_latency:.3f} seconds\n")
        f.write(f"Median latency: {median_latency:.3f} seconds\n")
        f.write(f"Min latency: {min_latency:.3f} seconds\n")
        f.write(f"Max latency: {max_latency:.3f} seconds\n")
        f.write(f"Standard deviation: {std_dev:.3f} seconds\n")
        f.write(f"Requests per second: {1/avg_latency:.2f}\n\n")
        
        f.write("INDIVIDUAL RESULTS\n")
        f.write("-" * 30 + "\n")
        for i, latency in enumerate(latencies, 1):
            f.write(f"Request {i}: {latency:.3f}s\n")
        
        f.write(f"\nDETAILED LOG\n")
        f.write("-" * 30 + "\n")
        for log_entry in results_log:
            f.write(f"Test {log_entry['test_number']}: {log_entry['latency_seconds']:.3f}s\n")
            f.write(f"Query: {log_entry['query']}\n")
            f.write(f"Response: {log_entry['response_preview']}\n")
            f.write(f"Time: {log_entry['timestamp']}\n\n")
    
    print(f"\n📊 Results saved to: {filename}")
    print(f"\n⚠️ Remember to delete the endpoint when done:")
    print(f"aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME} --region {REGION}")

if __name__ == "__main__":
    run_benchmark()
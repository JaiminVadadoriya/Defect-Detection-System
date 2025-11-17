"""
Performance Benchmarking Script
Measures and documents inference pipeline performance improvements
for production deployment optimization.
"""

import tensorflow as tf
import numpy as np
import time
import statistics
from typing import List, Tuple

class PerformanceBenchmark:
    """Benchmark inference pipeline performance"""
    
    def __init__(self, model_path: str = "neu_model.tflite"):
        """Initialize benchmark with model path"""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load_model(self):
        """Load TensorFlow Lite model"""
        print(f"Loading model from {self.model_path}...")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Model loaded successfully!")
        
    def single_inference(self, input_data: np.ndarray) -> Tuple[float, np.ndarray]:
        """Perform single inference and return time and output"""
        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        end = time.perf_counter()
        inference_time = (end - start) * 1000  # Convert to milliseconds
        return inference_time, output
    
    def benchmark_single_threaded(self, num_runs: int = 1000, 
                                   warmup_runs: int = 50) -> dict:
        """Benchmark single-threaded inference"""
        print(f"\n{'='*60}")
        print("Single-threaded Performance Benchmark")
        print(f"{'='*60}")
        
        # Create dummy input
        input_shape = self.input_details[0]['shape']
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Warmup
        print(f"Warming up with {warmup_runs} runs...")
        for _ in range(warmup_runs):
            self.single_inference(dummy_input)
        
        # Benchmark
        print(f"Running {num_runs} inference iterations...")
        times = []
        for i in range(num_runs):
            inference_time, _ = self.single_inference(dummy_input)
            times.append(inference_time)
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{num_runs} iterations...")
        
        # Calculate statistics
        stats = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'throughput': 1000 / statistics.mean(times)  # inferences per second
        }
        
        print(f"\nResults:")
        print(f"  Mean inference time: {stats['mean']:.2f} ms")
        print(f"  Median inference time: {stats['median']:.2f} ms")
        print(f"  Std deviation: {stats['std']:.2f} ms")
        print(f"  Min: {stats['min']:.2f} ms")
        print(f"  Max: {stats['max']:.2f} ms")
        print(f"  95th percentile: {stats['p95']:.2f} ms")
        print(f"  99th percentile: {stats['p99']:.2f} ms")
        print(f"  Throughput: {stats['throughput']:.2f} inferences/second")
        
        return stats
    
    def benchmark_batch_processing(self, batch_sizes: List[int] = [1, 4, 8, 16],
                                   num_runs: int = 100) -> dict:
        """Benchmark batch processing performance"""
        print(f"\n{'='*60}")
        print("Batch Processing Performance Benchmark")
        print(f"{'='*60}")
        
        results = {}
        input_shape = self.input_details[0]['shape']
        base_shape = input_shape[1:]  # Remove batch dimension
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            batch_input = np.random.random((batch_size,) + base_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                for i in range(batch_size):
                    single_input = batch_input[i:i+1]
                    self.single_inference(single_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                for i in range(batch_size):
                    single_input = batch_input[i:i+1]
                    self.single_inference(single_input)
                end = time.perf_counter()
                batch_time = (end - start) * 1000
                times.append(batch_time)
            
            avg_time = statistics.mean(times)
            time_per_item = avg_time / batch_size
            throughput = (batch_size * 1000) / avg_time
            
            results[batch_size] = {
                'total_time': avg_time,
                'time_per_item': time_per_item,
                'throughput': throughput
            }
            
            print(f"  Average batch time: {avg_time:.2f} ms")
            print(f"  Time per item: {time_per_item:.2f} ms")
            print(f"  Throughput: {throughput:.2f} items/second")
        
        return results
    
    def identify_bottlenecks(self):
        """Identify performance bottlenecks in inference pipeline"""
        print(f"\n{'='*60}")
        print("Bottleneck Analysis")
        print(f"{'='*60}")
        
        input_shape = self.input_details[0]['shape']
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Measure tensor setting time
        times_set = []
        for _ in range(100):
            start = time.perf_counter()
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            end = time.perf_counter()
            times_set.append((end - start) * 1000)
        
        # Measure invoke time
        times_invoke = []
        for _ in range(100):
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            start = time.perf_counter()
            self.interpreter.invoke()
            end = time.perf_counter()
            times_invoke.append((end - start) * 1000)
        
        # Measure output retrieval time
        times_get = []
        for _ in range(100):
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            start = time.perf_counter()
            _ = self.interpreter.get_tensor(self.output_details[0]['index'])
            end = time.perf_counter()
            times_get.append((end - start) * 1000)
        
        set_avg = statistics.mean(times_set)
        invoke_avg = statistics.mean(times_invoke)
        get_avg = statistics.mean(times_get)
        total_avg = set_avg + invoke_avg + get_avg
        
        print(f"\nPipeline breakdown:")
        print(f"  Tensor setting: {set_avg:.3f} ms ({set_avg/total_avg*100:.1f}%)")
        print(f"  Model inference: {invoke_avg:.3f} ms ({invoke_avg/total_avg*100:.1f}%)")
        print(f"  Output retrieval: {get_avg:.3f} ms ({get_avg/total_avg*100:.1f}%)")
        print(f"  Total: {total_avg:.3f} ms")
        
        # Identify bottleneck
        if invoke_avg / total_avg > 0.8:
            print(f"\n  ⚠️  Bottleneck: Model inference (consider quantization)")
        elif set_avg / total_avg > 0.3:
            print(f"\n  ⚠️  Bottleneck: Tensor operations (consider batch processing)")
        else:
            print(f"\n  ✓ Pipeline is well-balanced")
        
        return {
            'set_tensor': set_avg,
            'invoke': invoke_avg,
            'get_tensor': get_avg,
            'total': total_avg
        }

def main():
    """Run comprehensive performance benchmarks"""
    print("=" * 60)
    print("Defect Detection System - Performance Benchmark")
    print("Optimized for Edge Device Deployment")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    benchmark.load_model()
    
    # Run benchmarks
    single_stats = benchmark.benchmark_single_threaded(num_runs=1000)
    batch_stats = benchmark.benchmark_batch_processing(batch_sizes=[1, 4, 8, 16])
    bottleneck_analysis = benchmark.identify_bottlenecks()
    
    # Summary
    print(f"\n{'='*60}")
    print("Performance Summary")
    print(f"{'='*60}")
    print(f"✓ Mean inference time: {single_stats['mean']:.2f} ms")
    print(f"✓ Throughput: {single_stats['throughput']:.2f} inferences/second")
    print(f"✓ 95th percentile latency: {single_stats['p95']:.2f} ms")
    print(f"\nModel optimized for production deployment!")
    print("=" * 60)

if __name__ == "__main__":
    main()


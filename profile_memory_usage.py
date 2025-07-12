"""
Memory profiling and performance analysis script.
Profiles memory usage and training speed to optimize for full-scale training.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
import psutil
import os
from pathlib import Path
import gc

# Add src to path
sys.path.append('src')

from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.dataset import KhmerDataset, collate_fn
from src.utils.config import ConfigManager


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return {
        'cpu_memory_mb': process.memory_info().rss / 1024 / 1024,
        'cpu_memory_percent': process.memory_percent(),
        'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024
    }


def get_gpu_memory():
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        return {
            'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'gpu_cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'gpu_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        }
    return {}


def profile_model_memory(vocab_size: int = 117, device: torch.device = None):
    """Profile memory usage of model components."""
    print("=" * 60)
    print("MODEL MEMORY PROFILING")
    print("=" * 60)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Baseline memory
    baseline_memory = get_memory_usage()
    baseline_gpu = get_gpu_memory()
    
    print(f"\nBaseline Memory:")
    print(f"  CPU: {baseline_memory['cpu_memory_mb']:.1f} MB")
    if baseline_gpu:
        print(f"  GPU: {baseline_gpu['gpu_allocated_mb']:.1f} MB")
    
    # Create model
    print(f"\nCreating model...")
    model = KhmerOCRSeq2Seq(vocab_size=vocab_size)
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter memory: {param_memory_mb:.1f} MB")
    
    # Memory after model creation
    model_memory = get_memory_usage()
    model_gpu = get_gpu_memory()
    
    print(f"\nMemory after model creation:")
    print(f"  CPU: {model_memory['cpu_memory_mb']:.1f} MB (+{model_memory['cpu_memory_mb'] - baseline_memory['cpu_memory_mb']:.1f})")
    
    # Move to device
    model = model.to(device)
    
    device_memory = get_memory_usage()
    device_gpu = get_gpu_memory()
    
    print(f"\nMemory after moving to {device}:")
    print(f"  CPU: {device_memory['cpu_memory_mb']:.1f} MB")
    if device_gpu:
        print(f"  GPU: {device_gpu['gpu_allocated_mb']:.1f} MB (+{device_gpu['gpu_allocated_mb'] - baseline_gpu.get('gpu_allocated_mb', 0):.1f})")
    
    return {
        'model_cpu_memory_mb': model_memory['cpu_memory_mb'] - baseline_memory['cpu_memory_mb'],
        'model_gpu_memory_mb': device_gpu.get('gpu_allocated_mb', 0) - baseline_gpu.get('gpu_allocated_mb', 0),
        'total_params': total_params,
        'param_memory_mb': param_memory_mb
    }


def profile_training_memory(
    batch_sizes: list = [4, 8, 16, 32, 64],
    image_widths: list = [64, 128, 256, 512],
    sequence_lengths: list = [10, 20, 50, 100],
    device: torch.device = None
):
    """Profile memory usage during training with different configurations."""
    print("\n" + "=" * 60)
    print("TRAINING MEMORY PROFILING")
    print("=" * 60)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = ConfigManager()
    
    results = []
    
    for batch_size in batch_sizes:
        for image_width in image_widths:
            for seq_len in sequence_lengths:
                try:
                    # Clear cache
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Create model
                    model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
                    model = model.to(device)
                    
                    # Create optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    
                    # Create dummy batch
                    images = torch.randn(batch_size, 1, 32, image_width, device=device)
                    targets = torch.randint(0, len(config.vocab), (batch_size, seq_len), device=device)
                    
                    # Measure memory before forward pass
                    pre_memory = get_memory_usage()
                    pre_gpu = get_gpu_memory()
                    
                    # Forward pass
                    model.train()
                    optimizer.zero_grad()
                    
                    result = model(images, targets)
                    loss = result['loss']
                    
                    # Measure memory after forward pass
                    forward_memory = get_memory_usage()
                    forward_gpu = get_gpu_memory()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Measure memory after backward pass
                    backward_memory = get_memory_usage()
                    backward_gpu = get_gpu_memory()
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Final memory
                    final_memory = get_memory_usage()
                    final_gpu = get_gpu_memory()
                    
                    # Calculate memory usage
                    peak_cpu = max(forward_memory['cpu_memory_mb'], backward_memory['cpu_memory_mb'], final_memory['cpu_memory_mb'])
                    peak_gpu = max(forward_gpu.get('gpu_allocated_mb', 0), backward_gpu.get('gpu_allocated_mb', 0), final_gpu.get('gpu_allocated_mb', 0))
                    
                    result_data = {
                        'batch_size': batch_size,
                        'image_width': image_width,
                        'sequence_length': seq_len,
                        'peak_cpu_mb': peak_cpu,
                        'peak_gpu_mb': peak_gpu,
                        'forward_cpu_mb': forward_memory['cpu_memory_mb'],
                        'forward_gpu_mb': forward_gpu.get('gpu_allocated_mb', 0),
                        'backward_cpu_mb': backward_memory['cpu_memory_mb'],
                        'backward_gpu_mb': backward_gpu.get('gpu_allocated_mb', 0),
                        'loss_value': loss.item()
                    }
                    
                    results.append(result_data)
                    
                    print(f"Batch: {batch_size:2d}, Width: {image_width:3d}, Seq: {seq_len:3d} | "
                          f"CPU: {peak_cpu:6.1f} MB, GPU: {peak_gpu:6.1f} MB")
                    
                    # Clean up
                    del model, optimizer, images, targets, result, loss
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Batch: {batch_size:2d}, Width: {image_width:3d}, Seq: {seq_len:3d} | OOM")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    else:
                        print(f"Batch: {batch_size:2d}, Width: {image_width:3d}, Seq: {seq_len:3d} | Error: {e}")
                except Exception as e:
                    print(f"Batch: {batch_size:2d}, Width: {image_width:3d}, Seq: {seq_len:3d} | Error: {e}")
    
    return results


def profile_inference_speed(
    batch_sizes: list = [1, 4, 8, 16],
    image_widths: list = [64, 128, 256, 512],
    num_iterations: int = 10,
    device: torch.device = None
):
    """Profile inference speed with different configurations."""
    print("\n" + "=" * 60)
    print("INFERENCE SPEED PROFILING")
    print("=" * 60)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = ConfigManager()
    model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
    model = model.to(device)
    model.eval()
    
    results = []
    
    # Warmup
    print("Warming up GPU..." if device.type == 'cuda' else "Warming up...")
    with torch.no_grad():
        warmup_images = torch.randn(1, 1, 32, 128, device=device)
        for _ in range(5):
            _ = model.generate(warmup_images, method='greedy', max_length=10)
    
    for batch_size in batch_sizes:
        for image_width in image_widths:
            try:
                # Create test images
                images = torch.randn(batch_size, 1, 32, image_width, device=device)
                
                # Measure greedy decoding
                greedy_times = []
                with torch.no_grad():
                    for _ in range(num_iterations):
                        start_time = time.time()
                        _ = model.generate(images, method='greedy', max_length=20)
                        end_time = time.time()
                        greedy_times.append(end_time - start_time)
                
                avg_greedy_time = sum(greedy_times) / len(greedy_times)
                greedy_throughput = batch_size / avg_greedy_time
                
                # Measure beam search (only for batch size 1)
                beam_time = None
                beam_throughput = None
                if batch_size == 1:
                    beam_times = []
                    with torch.no_grad():
                        for _ in range(num_iterations):
                            start_time = time.time()
                            _ = model.generate(images, method='beam_search', beam_size=5, max_length=20)
                            end_time = time.time()
                            beam_times.append(end_time - start_time)
                    
                    avg_beam_time = sum(beam_times) / len(beam_times)
                    beam_throughput = 1 / avg_beam_time
                    beam_time = avg_beam_time
                
                result_data = {
                    'batch_size': batch_size,
                    'image_width': image_width,
                    'greedy_time_s': avg_greedy_time,
                    'greedy_throughput': greedy_throughput,
                    'beam_time_s': beam_time,
                    'beam_throughput': beam_throughput
                }
                
                results.append(result_data)
                
                beam_info = f", Beam: {beam_time:.3f}s ({beam_throughput:.1f} imgs/s)" if beam_time else ""
                print(f"Batch: {batch_size:2d}, Width: {image_width:3d} | "
                      f"Greedy: {avg_greedy_time:.3f}s ({greedy_throughput:.1f} imgs/s){beam_info}")
                
            except Exception as e:
                print(f"Batch: {batch_size:2d}, Width: {image_width:3d} | Error: {e}")
    
    return results


def analyze_optimal_settings(memory_results: list, speed_results: list, target_gpu_memory_gb: float = 16.0):
    """Analyze results to recommend optimal training settings."""
    print("\n" + "=" * 60)
    print("OPTIMAL SETTINGS ANALYSIS")
    print("=" * 60)
    
    target_gpu_memory_mb = target_gpu_memory_gb * 1024
    
    # Filter configurations that fit in memory
    feasible_configs = [r for r in memory_results if r['peak_gpu_mb'] < target_gpu_memory_mb * 0.8]  # 80% utilization
    
    if not feasible_configs:
        print("⚠️  No configurations fit in target GPU memory!")
        print("Consider reducing batch size or sequence length.")
        return
    
    # Find optimal batch size (largest that fits)
    max_batch_size = max(r['batch_size'] for r in feasible_configs)
    optimal_configs = [r for r in feasible_configs if r['batch_size'] == max_batch_size]
    
    print(f"Target GPU Memory: {target_gpu_memory_gb} GB")
    print(f"Maximum feasible batch size: {max_batch_size}")
    
    # Analyze by sequence length
    seq_lengths = sorted(set(r['sequence_length'] for r in optimal_configs))
    
    print(f"\nMemory usage by sequence length (batch size {max_batch_size}):")
    for seq_len in seq_lengths:
        seq_configs = [r for r in optimal_configs if r['sequence_length'] == seq_len]
        if seq_configs:
            avg_gpu_memory = sum(r['peak_gpu_mb'] for r in seq_configs) / len(seq_configs)
            print(f"  Seq {seq_len:3d}: {avg_gpu_memory:6.1f} MB ({avg_gpu_memory/1024:.1f} GB)")
    
    # Throughput analysis
    if speed_results:
        print(f"\nThroughput analysis:")
        greedy_throughputs = [r['greedy_throughput'] for r in speed_results if r['batch_size'] == max_batch_size]
        if greedy_throughputs:
            avg_throughput = sum(greedy_throughputs) / len(greedy_throughputs)
            print(f"  Average greedy throughput: {avg_throughput:.1f} images/second")
            
            # Estimate training time
            samples_per_epoch = 10000  # Typical dataset size
            estimated_time_per_epoch = samples_per_epoch / avg_throughput / 60  # minutes
            print(f"  Estimated time per epoch ({samples_per_epoch} samples): {estimated_time_per_epoch:.1f} minutes")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"  ✓ Recommended batch size: {max_batch_size}")
    
    # Find best sequence length (highest that fits comfortably)
    safe_configs = [r for r in optimal_configs if r['peak_gpu_mb'] < target_gpu_memory_mb * 0.7]  # 70% for safety
    if safe_configs:
        max_safe_seq = max(r['sequence_length'] for r in safe_configs)
        print(f"  ✓ Recommended max sequence length: {max_safe_seq}")
    
    # Memory efficiency
    param_memory = memory_results[0].get('param_memory_mb', 0) if memory_results else 0
    if param_memory > 0:
        memory_efficiency = param_memory / target_gpu_memory_mb * 100
        print(f"  ✓ Model parameters use {memory_efficiency:.1f}% of GPU memory")
    
    return {
        'recommended_batch_size': max_batch_size,
        'feasible_configs': len(feasible_configs),
        'max_feasible_seq_length': max(r['sequence_length'] for r in feasible_configs) if feasible_configs else 0
    }


def main():
    """Main profiling function."""
    print("🔍 MEMORY AND PERFORMANCE PROFILING")
    print("🎯 Optimizing for full-scale training")
    print()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check system resources
    print("System Resources:")
    memory_info = get_memory_usage()
    print(f"  CPU Memory: {memory_info['cpu_memory_mb']:.1f} MB used, {memory_info['available_memory_mb']:.1f} MB available")
    
    if device.type == 'cuda':
        gpu_info = get_gpu_memory()
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {gpu_info['gpu_total_mb']:.1f} MB total")
    
    print()
    
    # Profile model memory
    model_profile = profile_model_memory(device=device)
    
    # Profile training memory
    training_profile = profile_training_memory(
        batch_sizes=[4, 8, 16, 32] if device.type == 'cuda' else [4, 8],
        image_widths=[64, 128, 256] if device.type == 'cuda' else [64, 128],
        sequence_lengths=[10, 20, 50] if device.type == 'cuda' else [10, 20],
        device=device
    )
    
    # Profile inference speed
    speed_profile = profile_inference_speed(
        batch_sizes=[1, 4, 8] if device.type == 'cuda' else [1, 4],
        image_widths=[64, 128, 256] if device.type == 'cuda' else [64, 128],
        device=device
    )
    
    # Analyze optimal settings
    if device.type == 'cuda':
        gpu_info = get_gpu_memory()
        target_memory_gb = gpu_info['gpu_total_mb'] / 1024
    else:
        target_memory_gb = 8.0  # Default for CPU
    
    optimal_settings = analyze_optimal_settings(training_profile, speed_profile, target_memory_gb)
    
    print(f"\n🎉 PROFILING COMPLETED!")
    print(f"Results can be used to optimize full-scale training configuration.")


if __name__ == "__main__":
    main() 
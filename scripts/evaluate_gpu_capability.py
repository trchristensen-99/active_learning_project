#!/usr/bin/env python3
"""
GPU Capability Evaluation Script

Dynamically evaluates GPU memory and performance to determine optimal batch sizes
for reproducible training across different hardware configurations.
"""

import torch
import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.models import build_model
from code.utils import sequence_to_tensor


def get_gpu_info() -> Dict:
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(),
        "total_memory_gb": props.total_memory / 1e9,
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    }


def test_batch_size_for_model(
    model_type: str,
    batch_size: int,
    seqsize: int = 249,
    in_channels: int = 5,
    device: torch.device = None
) -> Tuple[bool, float, str]:
    """
    Test if a given batch size works for a specific model.
    
    Returns:
        (success, memory_used_gb, error_message)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Clear GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Build model
        if model_type == "deepstarr":
            # Import locally to avoid circular deps
            from code.active_learning.student import DeepSTARRStudent
            model = DeepSTARRStudent(seqsize=seqsize, in_channels=4,
                                     generator=torch.Generator(device=device).manual_seed(42))
        else:
            model = build_model(
                model_type,
                seqsize=seqsize,
                in_channels=in_channels,
                generator=torch.Generator(device=device).manual_seed(42)
            )
        model = model.to(device)
        
        # Create dummy data
        dummy_input = torch.randn(batch_size, in_channels, seqsize, device=device)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            if model_type == "dream_rnn":
                dev_output, hk_output = model(dummy_input)
            elif model_type == "deepstarr":
                dev_output, hk_output = model(dummy_input)
            else:
                # For other models, assume single output
                output = model(dummy_input)
        
        # Test backward pass (more memory intensive)
        model.train()
        if model_type in ("dream_rnn", "deepstarr"):
            dev_output, hk_output = model(dummy_input)
            dummy_target_dev = torch.randn_like(dev_output)
            dummy_target_hk = torch.randn_like(hk_output)
            loss_dev = torch.nn.functional.mse_loss(dev_output, dummy_target_dev)
            loss_hk = torch.nn.functional.mse_loss(hk_output, dummy_target_hk)
            loss = loss_dev + loss_hk
        else:
            output = model(dummy_input)
            dummy_target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, dummy_target)
        
        loss.backward()
        
        # Get memory usage
        if device.type == "cuda":
            memory_used_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            memory_used_gb = 0.0
        
        # Cleanup
        del model, dummy_input, loss
        if model_type in ("dream_rnn", "deepstarr"):
            del dev_output, hk_output, dummy_target_dev, dummy_target_hk, loss_dev, loss_hk
        else:
            del output, dummy_target
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return True, memory_used_gb, None
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, 0.0, "Out of memory"
        return False, 0.0, str(e)
    except Exception as e:
        return False, 0.0, str(e)


def find_optimal_batch_size(
    model_type: str,
    seqsize: int = 249,
    in_channels: int = 5,
    max_batch_size: int = 8192,
    safety_margin: float = 1.0
) -> Dict:
    """
    Find optimal batch size for a given model.
    
    Args:
        model_type: Type of model to test
        seqsize: Sequence length
        in_channels: Number of input channels
        max_batch_size: Maximum batch size to test
        safety_margin: Safety margin for memory usage (0.8 = use 80% of max)
    
    Returns:
        Dictionary with batch size recommendations
    """
    if not torch.cuda.is_available():
        return {
            "optimal_batch_size": 32,
            "max_safe_batch_size": 16,
            "recommended_batch_size": 8,
            "gpu_available": False,
            "reason": "CUDA not available"
        }
    
    device = torch.device("cuda")
    gpu_info = get_gpu_info()
    total_memory_gb = gpu_info["total_memory_gb"]
    
    print(f"🔍 Testing batch sizes for {model_type} model on {gpu_info['name']}")
    print(f"   GPU Memory: {total_memory_gb:.1f} GB")
    
    # Test batch sizes starting from small and doubling (extend up to 8192)
    if model_type == "deepstarr":
        # BatchNorm requires batch >= 2. Extend testing for very large GPUs
        base_sizes = [
            2, 4, 8, 16, 32, 64, 128, 256,
            512, 1024, 2048, 4096, 8192,
            16384, 32768, 65536
        ]
    else:
        base_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batch_sizes_to_test = [bs for bs in base_sizes if bs <= max_batch_size]
    batch_sizes_to_test = [bs for bs in batch_sizes_to_test if bs <= max_batch_size]
    
    results = {}
    max_successful_batch_size = 0
    max_memory_used = 0.0
    
    for batch_size in batch_sizes_to_test:
        print(f"   Testing batch size {batch_size:4d}... ", end="", flush=True)
        success, memory_used, error = test_batch_size_for_model(
            model_type, batch_size, seqsize, in_channels, device
        )
        
        results[batch_size] = {
            "success": success,
            "memory_used_gb": memory_used,
            "error": error
        }
        
        if success:
            print(f"✅ Success ({memory_used:.2f} GB)")
            max_successful_batch_size = batch_size
            max_memory_used = max(max_memory_used, memory_used)
        else:
            print(f"❌ Failed: {error}")
            break
    
    # Calculate recommendations
    optimal_batch_size = max_successful_batch_size
    max_safe_batch_size = int(max_successful_batch_size * safety_margin)
    # Use the maximum successful batch size as the recommendation
    recommended_batch_size = max_successful_batch_size
    
    return {
        "optimal_batch_size": optimal_batch_size,
        "max_safe_batch_size": max_safe_batch_size,
        "recommended_batch_size": recommended_batch_size,
        "gpu_available": True,
        "gpu_info": gpu_info,
        "max_memory_used_gb": max_memory_used,
        "test_results": results,
        "safety_margin": safety_margin
    }


def evaluate_oracle_capability() -> Dict:
    """Evaluate optimal batch size for oracle ensemble."""
    return find_optimal_batch_size(
        model_type="dream_rnn",
        seqsize=249,
        in_channels=5,
        max_batch_size=8192,
        safety_margin=1.0
    )


def evaluate_student_capability() -> Dict:
    """Evaluate optimal batch size for student model training."""
    return find_optimal_batch_size(
        model_type="deepstarr",
        seqsize=249,
        in_channels=4,
        max_batch_size=8192,
        safety_margin=1.0
    )


def save_gpu_capability_report(capabilities: Dict, output_path: str):
    """Save GPU capability report to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(capabilities, f, indent=2, default=str)
    
    print(f"📊 GPU capability report saved to: {output_path}")


def main():
    """Main evaluation function."""
    print("🚀 Evaluating GPU capabilities for active learning pipeline...")
    
    # Get GPU info
    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        print("❌ CUDA not available. Using CPU defaults.")
        capabilities = {
            "oracle": {"recommended_batch_size": 8, "gpu_available": False},
            "student": {"recommended_batch_size": 16, "gpu_available": False},
            "gpu_info": gpu_info
        }
    else:
        print(f"✅ GPU detected: {gpu_info['name']} ({gpu_info['total_memory_gb']:.1f} GB)")
        
        # Evaluate oracle capability
        print("\n🔮 Evaluating Oracle (DREAM-RNN) capability...")
        oracle_capability = evaluate_oracle_capability()
        
        # Evaluate student capability (if available)
        print("\n🎓 Evaluating Student (DeepSTARR) capability...")
        try:
            student_capability = evaluate_student_capability()
        except Exception as e:
            print(f"   ⚠️  Student model evaluation failed: {e}")
            student_capability = {"recommended_batch_size": 16, "gpu_available": True}
        
        capabilities = {
            "oracle": oracle_capability,
            "student": student_capability,
            "gpu_info": gpu_info,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Save report
    output_path = "results/gpu_capability_report.json"
    save_gpu_capability_report(capabilities, output_path)
    
    # Print summary
    print("\n📋 GPU Capability Summary:")
    print(f"   Oracle recommended batch size: {capabilities['oracle']['recommended_batch_size']}")
    print(f"   Student recommended batch size: {capabilities['student']['recommended_batch_size']}")
    
    return capabilities


if __name__ == "__main__":
    capabilities = main()
    sys.exit(0)


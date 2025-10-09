#!/usr/bin/env python3
"""
Test different batch sizes to find the maximum that fits in GPU memory.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.models import build_model
from code.prixfixe import DREAMRNNDataset


def test_batch_size(batch_size, device):
    """Test if a batch size fits in GPU memory."""
    try:
        # Create model
        model = build_model(
            "dream_rnn",
            seqsize=249,
            in_channels=5,
            generator=torch.Generator(device=device).manual_seed(42)
        )
        model = model.to(device)
        
        # Create dummy data
        dummy_input = torch.randn(batch_size, 5, 249, device=device)
        dummy_target_dev = torch.randn(batch_size, 1, device=device)  # Dev output
        dummy_target_hk = torch.randn(batch_size, 1, device=device)   # Hk output
        
        # Test forward pass
        with torch.no_grad():
            dev_output, hk_output = model(dummy_input)
        
        # Test backward pass (more memory intensive)
        model.train()
        dev_output, hk_output = model(dummy_input)
        loss_dev = torch.nn.functional.mse_loss(dev_output, dummy_target_dev)
        loss_hk = torch.nn.functional.mse_loss(hk_output, dummy_target_hk)
        loss = loss_dev + loss_hk
        loss.backward()
        
        # Clear memory
        del model, dummy_input, dummy_target_dev, dummy_target_hk, dev_output, hk_output, loss_dev, loss_hk, loss
        torch.cuda.empty_cache()
        
        return True, None
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False, "Out of memory"
        else:
            return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = torch.device("cuda")
    print(f"🔍 Testing batch sizes on {torch.cuda.get_device_name()}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    max_successful = 16
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size:4d}... ", end="", flush=True)
        success, error = test_batch_size(batch_size, device)
        
        if success:
            print("✅ Success")
            max_successful = batch_size
            results[batch_size] = "Success"
        else:
            print(f"❌ Failed: {error}")
            results[batch_size] = error
            break  # No point testing larger sizes
    
    print()
    print("=" * 50)
    print(f"🎯 Maximum successful batch size: {max_successful}")
    print()
    print("📊 Results:")
    for batch_size, result in results.items():
        status = "✅" if result == "Success" else "❌"
        print(f"  {status} Batch size {batch_size:4d}: {result}")
    
    print()
    print("💡 Recommendations:")
    print(f"  • Use batch size {max_successful} for maximum GPU utilization")
    print(f"  • Use batch size {max_successful // 2} for safety margin")
    print(f"  • Use batch size {max_successful // 4} for multiple models in parallel")
    
    # Estimate training time improvement
    if max_successful > 16:
        speedup = max_successful / 16
        print(f"  • Expected speedup vs batch_size=16: ~{speedup:.1f}x faster")


if __name__ == "__main__":
    main()

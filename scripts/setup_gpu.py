#!/usr/bin/env python3
"""
GPU Setup Script for DREAM-RNN Training

This script automatically detects NVIDIA driver version and installs
the compatible PyTorch version for optimal GPU performance.
"""

import subprocess
import sys
import re
import os
from pathlib import Path


def get_nvidia_driver_version():
    """Get NVIDIA driver version using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        # Parse driver version from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version:' in line:
                match = re.search(r'Driver Version: (\d+)\.(\d+)', line)
                if match:
                    major, minor = match.groups()
                    return int(major), int(minor)
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_cuda_version():
    """Get CUDA version using nvcc."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        # Parse CUDA version from nvcc output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'release' in line.lower():
                match = re.search(r'release (\d+)\.(\d+)', line)
                if match:
                    major, minor = match.groups()
                    return int(major), int(minor)
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_compatible_pytorch_versions(driver_major, driver_minor):
    """
    Determine compatible PyTorch versions based on driver version.
    
    Returns:
        tuple: (torch_version, torchvision_version, torchaudio_version, cuda_version, index_url)
    """
    # Driver version compatibility mapping with correct version combinations
    if driver_major >= 525:  # Driver 525+
        return "2.1.0", "0.16.0", "2.1.0", "cu121", "https://download.pytorch.org/whl/cu121"
    elif driver_major >= 520:  # Driver 520-524
        return "2.0.1", "0.15.2", "2.0.2", "cu118", "https://download.pytorch.org/whl/cu118"
    elif driver_major >= 515:  # Driver 515-519
        return "1.13.1", "0.14.1", "0.13.1", "cu117", "https://download.pytorch.org/whl/cu117"
    elif driver_major >= 510:  # Driver 510-514
        return "1.12.1", "0.13.1", "0.12.1", "cu116", "https://download.pytorch.org/whl/cu116"
    elif driver_major >= 470:  # Driver 470-509
        return "1.10.0", "0.11.0", "0.10.0", "cu111", "https://download.pytorch.org/whl/cu111"
    elif driver_major >= 450:  # Driver 450-469
        return "1.10.0", "0.11.0", "0.10.0", "cu111", "https://download.pytorch.org/whl/cu111"
    else:
        # Very old drivers - use CPU version with warning
        return "1.13.1", "0.14.1", "0.13.1", "cpu", "https://download.pytorch.org/whl/cpu"


def install_pytorch(torch_version, torchvision_version, torchaudio_version, cuda_version, index_url):
    """Install PyTorch with specified versions and CUDA support."""
    if cuda_version == "cpu":
        cmd = [
            sys.executable, "-m", "pip", "install",
            f"torch=={torch_version}",
            f"torchvision=={torchvision_version}",
            f"torchaudio=={torchaudio_version}",
            "--index-url", index_url
        ]
    else:
        cmd = [
            sys.executable, "-m", "pip", "install",
            f"torch=={torch_version}+{cuda_version}",
            f"torchvision=={torchvision_version}+{cuda_version}",
            f"torchaudio=={torchaudio_version}+{cuda_version}",
            "--extra-index-url", index_url
        ]
    
    print(f"Installing PyTorch {torch_version} with {cuda_version} support...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def fix_cudnn_compatibility():
    """Fix cuDNN version compatibility issues."""
    print("🔧 Checking cuDNN compatibility...")
    
    try:
        import torch
        # Test if cuDNN is working
        torch.backends.cudnn.version()
        print("✅ cuDNN compatibility check passed")
        return True
    except RuntimeError as e:
        if "cuDNN version incompatibility" in str(e):
            print("⚠️  cuDNN version incompatibility detected")
            print("🔧 Creating GPU wrapper script to fix this issue...")
            
            # Create a wrapper script that clears LD_LIBRARY_PATH
            wrapper_script = """#!/bin/bash
# GPU training wrapper script
# This script clears LD_LIBRARY_PATH to avoid cuDNN version conflicts
unset LD_LIBRARY_PATH
exec "$@"
"""
            
            wrapper_path = Path(__file__).parent.parent / "scripts" / "gpu_wrapper.sh"
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_script)
            
            # Make it executable
            os.chmod(wrapper_path, 0o755)
            
            print(f"✅ Created GPU wrapper script: {wrapper_path}")
            print("   Use this script to run GPU training commands:")
            print(f"   ./scripts/gpu_wrapper.sh python your_script.py")
            
            return True
        else:
            print(f"❌ cuDNN error: {e}")
            return False
    except Exception as e:
        print(f"❌ cuDNN check failed: {e}")
        return False


def test_gpu_availability():
    """Test if GPU is available after PyTorch installation."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   CUDA version: {torch.version.cuda}")
            
            # Check cuDNN compatibility
            fix_cudnn_compatibility()
            
            return True
        else:
            print("❌ GPU not available after installation")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def main():
    """Main setup function."""
    print("🔍 Detecting GPU setup...")
    
    # Check if nvidia-smi is available
    driver_version = get_nvidia_driver_version()
    if driver_version is None:
        print("❌ NVIDIA driver not found or nvidia-smi not available")
        print("   Please install NVIDIA drivers first")
        return False
    
    driver_major, driver_minor = driver_version
    print(f"📊 NVIDIA Driver Version: {driver_major}.{driver_minor}")
    
    # Check CUDA version
    cuda_version = get_cuda_version()
    if cuda_version:
        cuda_major, cuda_minor = cuda_version
        print(f"📊 CUDA Version: {cuda_major}.{cuda_minor}")
    else:
        print("📊 CUDA not found (driver-only installation)")
    
    # Determine compatible PyTorch versions
    torch_version, torchvision_version, torchaudio_version, cuda_support, index_url = get_compatible_pytorch_versions(
        driver_major, driver_minor
    )
    
    print(f"🎯 Recommended PyTorch: {torch_version} with {cuda_support} support")
    print(f"   torchvision: {torchvision_version}")
    print(f"   torchaudio: {torchaudio_version}")
    
    if cuda_support == "cpu":
        print("⚠️  Warning: Your NVIDIA driver is very old. Installing CPU-only PyTorch.")
        print("   For GPU training, please update your NVIDIA drivers.")
        response = input("Continue with CPU-only installation? (y/N): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return False
    
    # Install PyTorch
    success = install_pytorch(torch_version, torchvision_version, torchaudio_version, cuda_support, index_url)
    if not success:
        return False
    
    # Test GPU availability
    print("\n🧪 Testing GPU availability...")
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        print("\n🎉 GPU setup complete! Ready for training.")
    else:
        print("\n⚠️  GPU setup complete but GPU not available.")
        print("   You may need to restart your Python environment.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
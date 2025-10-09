#!/usr/bin/env python3
"""
Complete environment setup for DREAM-RNN training pipeline.

This script handles the entire setup process including GPU configuration.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install Python requirements."""
    print("📦 Installing Python requirements...")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)
        
        print("✅ Requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print(f"Error: {e.stderr}")
        return False


def setup_gpu():
    """Setup GPU support."""
    print("🔧 Setting up GPU support...")
    
    script_path = Path(__file__).parent / "setup_gpu.py"
    if not script_path.exists():
        print("❌ GPU setup script not found!")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ GPU setup completed successfully!")
            return True
        else:
            print("⚠️  GPU setup completed with warnings:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"⚠️  GPU setup failed: {e}")
        return False


def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        import pandas
        import numpy
        import tqdm
        import scipy
        import tensorboard
        
        print("✅ All required packages imported successfully!")
        
        # Test GPU availability
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  GPU not available - you may need to restart your environment")
            return False
            
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up DREAM-RNN training environment...")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    print()
    
    # Step 2: Setup GPU
    gpu_success = setup_gpu()
    
    print()
    
    # Step 3: Test installation
    test_success = test_installation()
    
    print()
    print("=" * 50)
    
    if gpu_success and test_success:
        print("🎉 Environment setup complete! Ready for training.")
        print("\nNext steps:")
        print("1. Download data: dvc repro download_data")
        print("2. Preprocess data: dvc repro preprocess_deepstarr")
        print("3. Train models: dvc repro train_dream_rnn_ensemble")
    else:
        print("⚠️  Setup completed with warnings.")
        print("   GPU training may not be available.")
        print("   You can run 'python scripts/setup_gpu.py' manually if needed.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

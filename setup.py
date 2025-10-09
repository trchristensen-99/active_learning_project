#!/usr/bin/env python3
"""
Setup script for DREAM-RNN training pipeline.

This script handles automatic GPU setup during package installation.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_gpu_setup():
    """Run GPU setup after package installation."""
    try:
        # Get the path to the setup_gpu.py script
        script_path = Path(__file__).parent / "scripts" / "setup_gpu.py"
        
        if script_path.exists():
            print("🔧 Running automatic GPU setup...")
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
        else:
            print("⚠️  GPU setup script not found, skipping...")
            return False
            
    except Exception as e:
        print(f"⚠️  GPU setup failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up DREAM-RNN training pipeline...")
    
    # Run GPU setup
    gpu_success = run_gpu_setup()
    
    if gpu_success:
        print("\n🎉 Setup complete! Ready for GPU training.")
    else:
        print("\n⚠️  Setup complete but GPU may not be available.")
        print("   You can run 'python scripts/setup_gpu.py' manually if needed.")
    
    return True


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Background training script for DREAM-RNN ensemble with comprehensive logging.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir):
    """Setup comprehensive logging."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    return log_file

def main():
    """Main training function with logging."""
    # Training configuration
    config = {
        "model_type": "dream_rnn",
        "train_data": "data/processed/train.txt",
        "val_data": "data/processed/val.txt", 
        "test_data": "data/processed/test.txt",
        "output_dir": "models/dream_rnn_ensemble",
        "n_models": 5,
        "epochs": 80,
        "batch_size": 1024,
        "lr": 0.005,
        "device": "auto",
        "n_workers": 16,
        "parallel": True,
        "max_parallel": 5
    }
    
    # Setup logging
    log_file = setup_logging(config["output_dir"])
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Build command
    cmd = [
        "python", "scripts/train_ensemble.py",
        "--model_type", config["model_type"],
        "--train_data", config["train_data"],
        "--val_data", config["val_data"],
        "--test_data", config["test_data"],
        "--output_dir", config["output_dir"],
        "--n_models", str(config["n_models"]),
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--device", config["device"],
        "--n_workers", str(config["n_workers"]),
        "--max_parallel", str(config["max_parallel"])
    ]
    
    if config["parallel"]:
        cmd.append("--parallel")
    
    # Start training with logging
    print(f"🚀 Starting DREAM-RNN ensemble training...")
    print(f"📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print(f"📝 Logging to: {log_file}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"DREAM-RNN Ensemble Training Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {config}\n")
            f.write("=" * 80 + "\n\n")
            
            # Run training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()
            
            process.wait()
            
            if process.returncode == 0:
                f.write(f"\n✅ Training completed successfully!\n")
                f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                print("\n✅ Training completed successfully!")
            else:
                f.write(f"\n❌ Training failed with return code: {process.returncode}\n")
                f.write(f"Failed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                print(f"\n❌ Training failed with return code: {process.returncode}")
                
    except Exception as e:
        error_msg = f"Training crashed: {str(e)}"
        print(f"\n💥 {error_msg}")
        with open(log_file, 'a') as f:
            f.write(f"\n💥 {error_msg}\n")
            f.write(f"Crashed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    
    print(f"⏱️  Total training time: {hours}h {minutes}m")
    print(f"📝 Full log available at: {log_file}")
    
    return process.returncode if 'process' in locals() else 1

if __name__ == "__main__":
    sys.exit(main())

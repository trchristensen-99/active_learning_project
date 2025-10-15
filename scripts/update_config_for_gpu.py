#!/usr/bin/env python3
"""
Dynamic Configuration Updater

Updates active learning configuration based on GPU capability evaluation
to ensure reproducible performance across different hardware.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_gpu_capability_report(report_path: str = "results/gpu_capability_report.json") -> Dict:
    """Load GPU capability report."""
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"GPU capability report not found: {report_path}")
    
    with open(report_path, 'r') as f:
        return json.load(f)


def update_config_with_gpu_capabilities(
    config_path: str,
    capabilities: Dict,
    output_path: str = None
) -> Dict:
    """
    Update configuration with GPU-optimized batch sizes.
    
    Args:
        config_path: Path to original configuration
        capabilities: GPU capability report
        output_path: Path to save updated configuration (optional)
    
    Returns:
        Updated configuration dictionary
    """
    # Load original configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update oracle batch size - use recommended (max successful)
    oracle_batch_size = capabilities["oracle"]["recommended_batch_size"]
    config["oracle"]["batch_size"] = oracle_batch_size
    
    # Update student batch size - use recommended (max successful)
    student_batch_size = capabilities["student"]["recommended_batch_size"]
    config["trainer"]["batch_size"] = student_batch_size
    
    # Add GPU capability metadata
    config["gpu_capabilities"] = {
        "oracle_batch_size": oracle_batch_size,
        "student_batch_size": student_batch_size,
        "gpu_name": capabilities["gpu_info"]["name"],
        "gpu_memory_gb": capabilities["gpu_info"]["total_memory_gb"],
        "evaluation_timestamp": capabilities.get("evaluation_timestamp", "unknown")
    }
    
    # Save updated configuration
    if output_path is None:
        output_path = config_path.replace(".json", "_gpu_optimized.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated configuration saved to: {output_path}")
    return config


def create_adaptive_config(
    base_config_path: str,
    capabilities: Dict,
    output_path: str = None
) -> Dict:
    """
    Create adaptive configuration that adjusts based on GPU capabilities.
    
    This version also adjusts the number of candidates per cycle based on GPU memory.
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Get GPU memory info
    gpu_memory_gb = capabilities["gpu_info"]["total_memory_gb"]
    oracle_batch_size = capabilities["oracle"]["recommended_batch_size"]
    
    # Lock requested values regardless of GPU memory (user preference)
    candidates_per_cycle = 100000
    acquire_per_cycle = 20000
    
    # Update configuration
    config["active_learning"]["n_candidates_per_cycle"] = candidates_per_cycle
    config["active_learning"]["n_acquire_per_cycle"] = acquire_per_cycle
    config["oracle"]["batch_size"] = oracle_batch_size
    config["trainer"]["batch_size"] = capabilities["student"]["recommended_batch_size"]
    
    # Add adaptive metadata
    config["adaptive_settings"] = {
        "gpu_memory_gb": gpu_memory_gb,
        "candidates_per_cycle": candidates_per_cycle,
        "acquire_per_cycle": acquire_per_cycle,
        "oracle_batch_size": oracle_batch_size,
        "student_batch_size": capabilities["student"]["recommended_batch_size"],
        "adaptation_reason": f"Adjusted for {gpu_memory_gb:.1f}GB GPU memory"
    }
    
    # Save adaptive configuration
    if output_path is None:
        output_path = base_config_path.replace(".json", "_adaptive.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Adaptive configuration saved to: {output_path}")
    print(f"   GPU Memory: {gpu_memory_gb:.1f} GB")
    print(f"   Candidates per cycle: {candidates_per_cycle:,}")
    print(f"   Acquire per cycle: {acquire_per_cycle:,}")
    print(f"   Oracle batch size: {oracle_batch_size}")
    print(f"   Student batch size: {capabilities['student']['recommended_batch_size']}")
    
    return config


def main():
    """Main function to update configuration based on GPU capabilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update configuration based on GPU capabilities")
    parser.add_argument("--config", required=True, help="Path to base configuration file")
    parser.add_argument("--capability-report", default="results/gpu_capability_report.json",
                       help="Path to GPU capability report")
    parser.add_argument("--output", help="Output path for updated configuration")
    parser.add_argument("--adaptive", action="store_true",
                       help="Create adaptive configuration that adjusts parameters based on GPU")
    
    args = parser.parse_args()
    
    # Load GPU capabilities
    print("📊 Loading GPU capability report...")
    capabilities = load_gpu_capability_report(args.capability_report)
    
    # Update configuration
    if args.adaptive:
        print("🔧 Creating adaptive configuration...")
        config = create_adaptive_config(args.config, capabilities, args.output)
    else:
        print("🔧 Updating configuration with GPU-optimized batch sizes...")
        config = update_config_with_gpu_capabilities(args.config, capabilities, args.output)
    
    print("✅ Configuration update complete!")
    return config


if __name__ == "__main__":
    main()


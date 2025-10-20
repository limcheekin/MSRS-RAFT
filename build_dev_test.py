#!/usr/bin/env python3
"""
Script to build dev and test RAFT datasets
"""

import os
import sys
from raft_config import RAFTConfig
from raft_pipeline import RAFTPipeline

def main():
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Load configuration
    config = RAFTConfig.from_yaml("raft_config.yaml")
    
    # Initialize pipeline
    print("Initializing RAFT pipeline...")
    pipeline = RAFTPipeline(config)
    
    # Step 1: Load data (if not already loaded)
    print("\n" + "="*60)
    print("Step 1: Loading data...")
    print("="*60)
    pipeline.step1_load_data()
    
    # Step 2: Build retrieval index (if not already built)
    print("\n" + "="*60)
    print("Step 2: Building retrieval index...")
    print("="*60)
    pipeline.step2_build_index()
    
    # Step 3: Build dev dataset (50 examples)
    print("\n" + "="*60)
    print("Step 3: Building dev dataset (50 examples)...")
    print("="*60)
    dev_path = pipeline.step3_build_raft_dataset(
        split="dev",
        max_examples=50,
        openai_api_key=openai_api_key
    )
    print(f"\n✓ Dev dataset created: {dev_path}")
    
    # Step 4: Build test dataset (100 examples)
    print("\n" + "="*60)
    print("Step 4: Building test dataset (100 examples)...")
    print("="*60)
    test_path = pipeline.step3_build_raft_dataset(
        split="test",
        max_examples=100,
        openai_api_key=openai_api_key
    )
    print(f"\n✓ Test dataset created: {test_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dev dataset:  {dev_path} (50 examples)")
    print(f"Test dataset: {test_path} (100 examples)")
    print("\nAll datasets created successfully! ✓")

if __name__ == "__main__":
    main()


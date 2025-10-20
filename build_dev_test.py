#!/usr/bin/env python3
"""
Script to build dev and test RAFT datasets (GPU-independent version)

This script directly uses the data loading, retrieval, and dataset building
components without importing the full pipeline, avoiding GPU dependencies
from the training modules.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from raft_config import RAFTConfig, setup_logging
from raft_data_loader import MSRSDataLoader
from raft_retrieval import RetrievalSystem
from raft_dataset_builder import (
    CoTGenerator,
    RAFTDatasetBuilder,
    prepare_examples_from_loader
)

logger = logging.getLogger("RAFT.BuildDevTest")


def step1_load_data(config: RAFTConfig) -> MSRSDataLoader:
    """Step 1: Load MSRS Story-QA dataset and corpus"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*60)

    # Initialize data loader
    data_loader = MSRSDataLoader(
        dataset_name=config.raft_data.dataset_name,
        dataset_config=config.raft_data.dataset_config,
        cache_dir=config.system.cache_dir
    )

    # Load dataset splits
    logger.info("Loading dataset splits...")
    data_loader.load_dataset()

    # Load corpus
    logger.info("Loading story corpus...")
    try:
        data_loader.load_corpus()
    except Exception as e:
        logger.warning(f"Could not load corpus from HF: {str(e)}")
        logger.info("You may need to provide corpus manually")

    # Print statistics
    stats = data_loader.get_statistics()
    logger.info(f"\nDataset Statistics:")
    logger.info(json.dumps(stats, indent=2))

    logger.info("✓ Data loading complete")
    return data_loader


def step2_build_index(
    config: RAFTConfig,
    data_loader: MSRSDataLoader,
    save_path: Optional[str] = None
) -> RetrievalSystem:
    """Step 2: Build retrieval index"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: BUILDING RETRIEVAL INDEX")
    logger.info("="*60)

    # Check if corpus is loaded
    if not data_loader._corpus:
        raise ValueError("Corpus is empty. Please ensure data loaded correctly.")

    # Determine index path
    index_path = Path(save_path) if save_path else Path(config.retrieval.index_path) / "raft_index"

    # If an index file exists on disk, load it instead of rebuilding
    if index_path.exists() or Path(str(index_path) + ".faiss").exists():
        logger.info(f"Existing index found at {index_path}, loading instead of rebuilding...")
        retrieval_system = RetrievalSystem(
            embedding_model=config.retrieval.embedding_model,
            reranker_model=config.retrieval.reranker_model if config.retrieval.use_reranker else None,
            index_path=str(index_path)
        )
        logger.info("✓ Loaded existing retrieval index")
        return retrieval_system

    # Initialize retrieval system (no existing index found)
    retrieval_system = RetrievalSystem(
        embedding_model=config.retrieval.embedding_model,
        reranker_model=config.retrieval.reranker_model if config.retrieval.use_reranker else None,
        chunk_size=config.raft_data.chunk_size,
        chunk_overlap=config.raft_data.chunk_overlap,
        device="cpu"  # Force CPU for non-GPU environments
    )

    # Prepare documents
    documents = {
        doc_id: chapter.text
        for doc_id, chapter in data_loader._corpus.items()
    }

    # Build index
    logger.info(f"Building index for {len(documents)} documents...")

    retrieval_system.build_index(
        documents,
        batch_size=config.retrieval.batch_size,
        save_path=str(index_path)
    )

    logger.info(f"✓ Index built and saved to {index_path}")
    return retrieval_system


def step3_build_raft_dataset(
    config: RAFTConfig,
    data_loader: MSRSDataLoader,
    retrieval_system: RetrievalSystem,
    split: str = "train",
    max_examples: Optional[int] = None,
    openai_api_key: Optional[str] = None
) -> str:
    """Step 3: Build RAFT training dataset"""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 3: BUILDING RAFT DATASET ({split} split)")
    logger.info("="*60)

    # Initialize CoT generator
    cot_generator = CoTGenerator(
        model=config.raft_data.judge_model,
        temperature=config.raft_data.judge_temperature,
        max_tokens=config.raft_data.judge_max_tokens,
        api_key=openai_api_key
    )

    # Initialize dataset builder
    dataset_builder = RAFTDatasetBuilder(
        retrieval_system=retrieval_system,
        cot_generator=cot_generator,
        oracle_percentage=config.raft_data.oracle_percentage,
        num_distractors=config.raft_data.num_distractors,
        distractor_pool_size=config.raft_data.distractor_pool_size,
        max_quote_length=config.raft_data.max_quote_length,
        min_quotes=config.raft_data.min_quotes_per_example,
        max_quotes=config.raft_data.max_quotes_per_example
    )

    # Prepare examples
    logger.info("Preparing examples from data loader...")
    examples = prepare_examples_from_loader(data_loader, split=split)

    if max_examples:
        examples = examples[:max_examples]
        logger.info(f"Limited to {max_examples} examples")

    # Build RAFT dataset
    output_path = Path(config.system.data_dir) / f"raft_{split}.jsonl"

    logger.info(f"Building RAFT dataset with {len(examples)} examples...")
    logger.info(f"Oracle percentage: {config.raft_data.oracle_percentage}")
    logger.info(f"Distractors per example: {config.raft_data.num_distractors}")

    raft_examples = dataset_builder.build_dataset(
        examples,
        str(output_path),
        max_examples=None  # Already limited above
    )

    # Print statistics
    stats = dataset_builder.get_statistics(raft_examples)
    logger.info(f"\nRAFT Dataset Statistics:")
    logger.info(json.dumps(stats, indent=2))

    logger.info(f"✓ RAFT dataset saved to {output_path}")

    return str(output_path)


def main():
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Load configuration
    config = RAFTConfig.from_yaml("raft_config.yaml")

    # Setup logging
    logger_instance = setup_logging(config.system)

    logger.info("="*60)
    logger.info("BUILD DEV/TEST DATASETS (GPU-INDEPENDENT)")
    logger.info("="*60)
    logger.info(f"Project: {config.system.project_name}")
    logger.info(f"Oracle %: {config.raft_data.oracle_percentage}")
    logger.info("="*60)

    try:
        # Step 1: Load data
        data_loader = step1_load_data(config)

        # Step 2: Build retrieval index
        retrieval_system = step2_build_index(config, data_loader)

        # Step 3: Build dev dataset (50 examples)
        print("\n" + "="*60)
        print("Building dev dataset (50 examples)...")
        print("="*60)
        dev_path = step3_build_raft_dataset(
            config=config,
            data_loader=data_loader,
            retrieval_system=retrieval_system,
            split="dev",
            max_examples=50,
            openai_api_key=openai_api_key
        )
        print(f"\n✓ Dev dataset created: {dev_path}")

        # Step 4: Build test dataset (100 examples)
        print("\n" + "="*60)
        print("Building test dataset (100 examples)...")
        print("="*60)
        test_path = step3_build_raft_dataset(
            config=config,
            data_loader=data_loader,
            retrieval_system=retrieval_system,
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

    except Exception as e:
        logger.error(f"Failed to build datasets: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Complete Example: RAFT Fine-tuning Pipeline
Demonstrates end-to-end usage with comments
"""

import os
import logging
from pathlib import Path

# Import all modules
from raft_config import RAFTConfig, setup_logging
from raft_pipeline import RAFTPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_quick_start():
    """Example 1: Quick start with minimal configuration"""
    logger.info("="*80)
    logger.info("EXAMPLE 1: Quick Start")
    logger.info("="*80)
    
    # Create default configuration
    config = RAFTConfig()
    
    # Customize for quick testing
    config.raft_data.oracle_percentage = 0.8
    config.training.num_train_epochs = 1  # Quick training
    config.training.per_device_train_batch_size = 2
    
    # Initialize pipeline
    pipeline = RAFTPipeline(config)
    
    # Run full pipeline with limited examples
    metrics = pipeline.run_full_pipeline(
        train_max_examples=10,  # Only 10 examples for testing
        eval_max_examples=5,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        skip_training=False
    )
    
    logger.info(f"Final metrics: {metrics}")


def example_2_step_by_step():
    """Example 2: Step-by-step execution with full control"""
    logger.info("="*80)
    logger.info("EXAMPLE 2: Step-by-Step Execution")
    logger.info("="*80)
    
    # Load configuration from file
    config = RAFTConfig.from_yaml("raft_config.yaml")
    
    # Initialize pipeline
    pipeline = RAFTPipeline(config)
    
    # Step 1: Load data
    logger.info("\n--- Step 1: Loading Data ---")
    pipeline.step1_load_data()
    
    # Inspect loaded data
    if pipeline.data_loader:
        stats = pipeline.data_loader.get_statistics()
        logger.info(f"Dataset statistics: {stats}")
    
    # Step 2: Build retrieval index
    logger.info("\n--- Step 2: Building Index ---")
    index_path = "./indices/story_qa_index"
    pipeline.step2_build_index(save_path=index_path)
    
    # Test retrieval
    if pipeline.retrieval_system:
        test_query = "What happens at the beginning of the story?"
        results = pipeline.retrieval_system.retrieve(test_query, top_k=3)
        logger.info(f"\nTest retrieval for: '{test_query}'")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result.doc_id} (score: {result.score:.4f})")
    
    # Step 3: Build RAFT training dataset
    logger.info("\n--- Step 3: Building RAFT Dataset ---")
    train_path = pipeline.step3_build_raft_dataset(
        split="train",
        max_examples=20,  # Limited for testing
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info(f"Training data saved to: {train_path}")
    
    # Build eval dataset
    eval_path = pipeline.step3_build_raft_dataset(
        split="dev",
        max_examples=10,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info(f"Eval data saved to: {eval_path}")
    
    # Step 4: Train model
    logger.info("\n--- Step 4: Training Model ---")
    pipeline.step4_train_model(train_path, eval_path)
    
    # Step 5: Save model
    logger.info("\n--- Step 5: Saving Model ---")
    model_output = "./models/raft_story_qa_v1"
    pipeline.step5_save_model(
        output_dir=model_output,
        save_method="merged_16bit"
    )
    
    # Step 6: Evaluate
    logger.info("\n--- Step 6: Evaluation ---")
    metrics = pipeline.step6_evaluate(
        test_split="test",
        model_path=model_output
    )
    
    logger.info(f"\nFinal evaluation metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


def example_3_custom_configuration():
    """Example 3: Custom configuration for specific use case"""
    logger.info("="*80)
    logger.info("EXAMPLE 3: Custom Configuration")
    logger.info("="*80)
    
    # Create custom configuration
    from raft_config import (
        ModelConfig,
        TrainingConfig,
        RAFTDataConfig,
        RetrievalConfig,
        EvaluationConfig,
        SystemConfig
    )
    
    # Customize model settings
    model_config = ModelConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length=2048,  # Shorter for faster training
        lora_r=16,  # Smaller LoRA rank
        lora_alpha=32
    )
    
    # Customize training settings
    training_config = TrainingConfig(
        num_train_epochs=2,
        per_device_train_batch_size=4,  # Larger batch
        gradient_accumulation_steps=2,
        learning_rate=3e-4,  # Higher LR
        eval_steps=100
    )
    
    # Customize RAFT data settings
    raft_data_config = RAFTDataConfig(
        oracle_percentage=0.7,  # Lower oracle %
        num_distractors=3,  # Fewer distractors
        chunk_size=1000,  # Smaller chunks
        judge_model="gpt-4-turbo-preview"
    )
    
    # Customize retrieval settings
    retrieval_config = RetrievalConfig(
        embedding_model="BAAI/bge-small-en-v1.5",  # Smaller, faster model
        use_reranker=False,  # Disable reranker for speed
        top_k=4
    )
    
    # Customize evaluation settings
    eval_config = EvaluationConfig(
        compute_faithfulness=True,
        compute_answer_relevance=True,
        compute_rouge=True,
        compute_bleu=False,  # Skip BLEU
        compute_bertscore=False  # Skip BERTScore for speed
    )
    
    # Customize system settings
    system_config = SystemConfig(
        project_name="MSRS-RAFT-CUSTOM",
        use_wandb=True,  # Enable W&B tracking
        wandb_project="raft-experiments",
        log_level="DEBUG"
    )
    
    # Create combined config
    config = RAFTConfig(
        model=model_config,
        training=training_config,
        raft_data=raft_data_config,
        retrieval=retrieval_config,
        evaluation=eval_config,
        system=system_config
    )
    
    # Save custom config
    config.to_yaml("custom_config.yaml")
    logger.info("Custom configuration saved to custom_config.yaml")
    
    # Run pipeline with custom config
    pipeline = RAFTPipeline(config)
    
    # Run with custom settings
    metrics = pipeline.run_full_pipeline(
        train_max_examples=50,
        eval_max_examples=20,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    logger.info(f"Results: {metrics}")


def example_4_evaluation_only():
    """Example 4: Evaluate a pre-trained model"""
    logger.info("="*80)
    logger.info("EXAMPLE 4: Evaluation Only")
    logger.info("="*80)
    
    # Load config
    config = RAFTConfig()
    
    # Initialize pipeline
    pipeline = RAFTPipeline(config)
    
    # Load data (needed for test examples)
    logger.info("Loading data...")
    pipeline.step1_load_data()
    
    # Load or build index
    index_path = "./indices/story_qa_index"
    if Path(index_path + ".faiss").exists():
        logger.info("Loading existing index...")
        from raft_retrieval import RetrievalSystem
        pipeline.retrieval_system = RetrievalSystem(
            embedding_model=config.retrieval.embedding_model,
            reranker_model=config.retrieval.reranker_model if config.retrieval.use_reranker else None,
            index_path=index_path
        )
    else:
        logger.info("Building new index...")
        pipeline.step2_build_index(save_path=index_path)
    
    # Evaluate pre-trained model
    model_path = "./models/raft_story_qa_v1"  # Path to your trained model
    
    if Path(model_path).exists():
        logger.info(f"Evaluating model from {model_path}")
        
        metrics = pipeline.step6_evaluate(
            test_split="test",
            model_path=model_path,
            output_path="./results/detailed_evaluation.jsonl"
        )
        
        # Print detailed results
        pipeline.evaluator.print_metrics(metrics)
    else:
        logger.warning(f"Model not found at {model_path}")
        logger.info("Train a model first using examples 1-3")


def example_5_compare_baselines():
    """Example 5: Compare RAFT against baselines"""
    logger.info("="*80)
    logger.info("EXAMPLE 5: Baseline Comparison")
    logger.info("="*80)
    
    from raft_config import RAFTConfig
    from raft_data_loader import MSRSDataLoader
    from raft_retrieval import RetrievalSystem
    from raft_evaluator import RAFTEvaluator
    from raft_dataset_builder import prepare_examples_from_loader
    from unsloth import FastLanguageModel
    import json
    
    # Setup
    config = RAFTConfig()
    
    # Load data
    logger.info("Loading data...")
    data_loader = MSRSDataLoader(cache_dir="./cache")
    data_loader.load_dataset()
    data_loader.load_corpus()
    
    # Build index
    logger.info("Building retrieval index...")
    retrieval_system = RetrievalSystem(
        embedding_model=config.retrieval.embedding_model,
        chunk_size=config.raft_data.chunk_size
    )
    
    documents = {
        doc_id: chapter.text
        for doc_id, chapter in data_loader._corpus.items()
    }
    retrieval_system.build_index(documents, save_path="./indices/comparison_index")
    
    # Prepare test examples
    test_examples = prepare_examples_from_loader(data_loader, split="test")
    eval_examples = [
        {
            'query': ex['query'],
            'gold_docs': [doc_id for doc_id, _ in ex['oracle_docs']],
            'answers': ex['answers']
        }
        for ex in test_examples[:20]  # Limited for demo
    ]
    
    # System 1: 0-shot baseline (no fine-tuning)
    logger.info("\n--- Evaluating 0-shot Baseline ---")
    model_0shot, tokenizer_0shot = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model_0shot)
    
    evaluator_0shot = RAFTEvaluator(
        config=config,
        retrieval_system=retrieval_system,
        model=model_0shot,
        tokenizer=tokenizer_0shot
    )
    
    metrics_0shot = evaluator_0shot.evaluate_dataset(
        eval_examples,
        output_path="./results/0shot_results.jsonl"
    )
    
    # System 2: RAFT fine-tuned model
    logger.info("\n--- Evaluating RAFT Model ---")
    model_raft_path = "./models/raft_story_qa_v1"
    
    if Path(model_raft_path).exists():
        model_raft, tokenizer_raft = FastLanguageModel.from_pretrained(
            model_name=model_raft_path,
            max_seq_length=config.model.max_seq_length,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(model_raft)
        
        evaluator_raft = RAFTEvaluator(
            config=config,
            retrieval_system=retrieval_system,
            model=model_raft,
            tokenizer=tokenizer_raft
        )
        
        metrics_raft = evaluator_raft.evaluate_dataset(
            eval_examples,
            output_path="./results/raft_results.jsonl"
        )
        
        # Compare results
        logger.info("\n" + "="*80)
        logger.info("COMPARISON RESULTS")
        logger.info("="*80)
        
        comparison = {
            "0-shot": metrics_0shot,
            "RAFT": metrics_raft
        }
        
        # Print side-by-side
        metrics_to_compare = [
            'faithfulness',
            'answer_relevance',
            'context_precision',
            'context_recall',
            'rouge_l'
        ]
        
        print(f"\n{'Metric':<25} {'0-shot':<15} {'RAFT':<15} {'Improvement':<15}")
        print("-" * 70)
        
        for metric in metrics_to_compare:
            if metric in metrics_0shot and metric in metrics_raft:
                val_0shot = metrics_0shot[metric]
                val_raft = metrics_raft[metric]
                
                if val_0shot and val_raft:
                    improvement = ((val_raft - val_0shot) / val_0shot) * 100
                    print(f"{metric:<25} {val_0shot:<15.4f} {val_raft:<15.4f} {improvement:>+14.2f}%")
        
        # Save comparison
        with open("./results/comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("\nComparison saved to ./results/comparison.json")
    else:
        logger.warning(f"RAFT model not found at {model_raft_path}")


def example_6_distractor_stress_test():
    """Example 6: Test model robustness with extra distractors"""
    logger.info("="*80)
    logger.info("EXAMPLE 6: Distractor Stress Test")
    logger.info("="*80)
    
    from raft_config import RAFTConfig
    from raft_evaluator import RAFTEvaluator
    from raft_retrieval import RetrievalSystem
    from raft_data_loader import MSRSDataLoader
    from raft_dataset_builder import prepare_examples_from_loader
    from unsloth import FastLanguageModel
    import random
    
    # Setup
    config = RAFTConfig()
    data_loader = MSRSDataLoader(cache_dir="./cache")
    data_loader.load_dataset()
    data_loader.load_corpus()
    
    # Load retrieval system
    retrieval_system = RetrievalSystem(
        embedding_model=config.retrieval.embedding_model
    )
    
    documents = {doc_id: ch.text for doc_id, ch in data_loader._corpus.items()}
    retrieval_system.build_index(documents, save_path="./indices/stress_test_index")
    
    # Load RAFT model
    model_path = "./models/raft_story_qa_v1"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    
    # Prepare test examples
    test_examples = prepare_examples_from_loader(data_loader, split="test")
    
    # Test with different distractor levels
    distractor_levels = [0, 2, 4, 6]
    results_by_level = {}
    
    for num_distractors in distractor_levels:
        logger.info(f"\n--- Testing with {num_distractors} extra distractors ---")
        
        eval_examples = []
        for ex in test_examples[:20]:
            # Get gold docs
            gold_doc_ids = [doc_id for doc_id, _ in ex['oracle_docs']]
            
            # Retrieve candidates
            results = retrieval_system.retrieve(ex['query'], top_k=20)
            
            # Add distractors
            distractors = [
                r for r in results
                if r.doc_id not in gold_doc_ids
            ][:num_distractors]
            
            eval_examples.append({
                'query': ex['query'],
                'gold_docs': gold_doc_ids,
                'answers': ex['answers'],
                'retrieved_docs': gold_doc_ids + [d.doc_id for d in distractors],
                'retrieved_texts': [data_loader._corpus[doc_id].text for doc_id in gold_doc_ids] +
                                 [d.text for d in distractors]
            })
        
        # Evaluate
        evaluator = RAFTEvaluator(
            config=config,
            retrieval_system=retrieval_system,
            model=model,
            tokenizer=tokenizer
        )
        
        metrics = evaluator.evaluate_dataset(
            eval_examples,
            output_path=f"./results/stress_test_{num_distractors}_distractors.jsonl"
        )
        
        results_by_level[f"{num_distractors}_distractors"] = metrics
    
    # Print stress test summary
    logger.info("\n" + "="*80)
    logger.info("STRESS TEST SUMMARY")
    logger.info("="*80)
    
    print(f"\n{'# Distractors':<15} {'Faithfulness':<15} {'Ans Relevance':<15} {'ROUGE-L':<15}")
    print("-" * 60)
    
    for level_name, metrics in results_by_level.items():
        num = level_name.split('_')[0]
        faith = metrics.get('faithfulness', 0.0)
        rel = metrics.get('answer_relevance', 0.0)
        rouge = metrics.get('rouge_l', 0.0)
        print(f"{num:<15} {faith:<15.4f} {rel:<15.4f} {rouge:<15.4f}")
    
    logger.info("\nRAFT should maintain performance even with distractors!")


def main():
    """Main function to run examples"""
    import sys
    
    examples = {
        '1': ('Quick Start', example_1_quick_start),
        '2': ('Step-by-Step', example_2_step_by_step),
        '3': ('Custom Config', example_3_custom_configuration),
        '4': ('Evaluation Only', example_4_evaluation_only),
        '5': ('Baseline Comparison', example_5_compare_baselines),
        '6': ('Stress Test', example_6_distractor_stress_test),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            logger.info(f"Running Example {choice}: {name}")
            func()
        else:
            logger.error(f"Invalid choice: {choice}")
            print_menu(examples)
    else:
        print_menu(examples)


def print_menu(examples):
    """Print example menu"""
    print("\n" + "="*80)
    print("RAFT Fine-tuning Examples")
    print("="*80)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\nUsage:")
    print("  python example_usage.py <example_number>")
    print("\nExamples:")
    print("  python example_usage.py 1  # Run quick start")
    print("  python example_usage.py 2  # Run step-by-step")
    print("\nNote: Set OPENAI_API_KEY environment variable before running")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
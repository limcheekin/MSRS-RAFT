"""
RAFT Pipeline Module
Complete end-to-end pipeline orchestration
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch

# Import all components
from raft_config import RAFTConfig, setup_logging
from raft_data_loader import MSRSDataLoader
from raft_retrieval import RetrievalSystem
from raft_dataset_builder import (
    CoTGenerator,
    RAFTDatasetBuilder,
    prepare_examples_from_loader
)
from raft_trainer import RAFTTrainer, LoggingCallback
from raft_evaluator import RAFTEvaluator

logger = logging.getLogger("RAFT.Pipeline")


class RAFTPipeline:
    """Complete RAFT training and evaluation pipeline"""
    
    def __init__(self, config: RAFTConfig):
        """
        Initialize RAFT pipeline
        
        Args:
            config: RAFTConfig object with all settings
        """
        self.config = config
        
        # Setup logging
        self.logger = setup_logging(config.system)
        
        # Initialize components
        self.data_loader = None
        self.retrieval_system = None
        self.dataset_builder = None
        self.trainer = None
        self.evaluator = None
        
        self.logger.info("="*60)
        self.logger.info("RAFT PIPELINE INITIALIZED")
        self.logger.info("="*60)
        self.logger.info(f"Project: {config.system.project_name}")
        self.logger.info(f"Model: {config.model.model_name}")
        self.logger.info(f"Oracle %: {config.raft_data.oracle_percentage}")
        self.logger.info("="*60)
    
    def step1_load_data(self):
        """Step 1: Load MSRS Story-QA dataset and corpus"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 1: LOADING DATA")
        self.logger.info("="*60)
        
        # Initialize data loader
        self.data_loader = MSRSDataLoader(
            dataset_name=self.config.raft_data.dataset_name,
            dataset_config=self.config.raft_data.dataset_config,
            cache_dir=self.config.system.cache_dir
        )
        
        # Load dataset splits
        self.logger.info("Loading dataset splits...")
        self.data_loader.load_dataset()
        
        # Load corpus
        self.logger.info("Loading story corpus...")
        try:
            self.data_loader.load_corpus()
        except Exception as e:
            self.logger.warning(f"Could not load corpus from HF: {str(e)}")
            self.logger.info("You may need to provide corpus manually")
        
        # Print statistics
        stats = self.data_loader.get_statistics()
        self.logger.info(f"\nDataset Statistics:")
        self.logger.info(json.dumps(stats, indent=2))
        
        self.logger.info("✓ Data loading complete")
    
    def step2_build_index(self, save_path: Optional[str] = None):
        """Step 2: Build retrieval index"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 2: BUILDING RETRIEVAL INDEX")
        self.logger.info("="*60)
        
        # Auto-load data if not already loaded
        if not self.data_loader or not self.data_loader._corpus:
            self.logger.info("Data not loaded yet, loading first...")
            self.step1_load_data()
        
        # Check again after auto-load attempt
        if not self.data_loader._corpus:
            raise ValueError("Corpus is empty. Please ensure data loaded correctly.")
        
        # Determine index path
        index_path = Path(save_path) if save_path else Path(self.config.retrieval.index_path) / "raft_index"
        
        # If an index file exists on disk, load it instead of rebuilding
        # Matches the logic used in the "eval" branch (checks for index_path.faiss)
        if index_path.exists() or Path(str(index_path) + ".faiss").exists():
            self.logger.info(f"Existing index found at {index_path}, loading instead of rebuilding...")
            self.retrieval_system = RetrievalSystem(
                embedding_model=self.config.retrieval.embedding_model,
                reranker_model=self.config.retrieval.reranker_model if self.config.retrieval.use_reranker else None,
                index_path=str(index_path)
            )
            self.logger.info("✓ Loaded existing retrieval index")
            return
        
        # Initialize retrieval system (no existing index found)
        self.retrieval_system = RetrievalSystem(
            embedding_model=self.config.retrieval.embedding_model,
            reranker_model=self.config.retrieval.reranker_model if self.config.retrieval.use_reranker else None,
            chunk_size=self.config.raft_data.chunk_size,
            chunk_overlap=self.config.raft_data.chunk_overlap,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Prepare documents
        documents = {
            doc_id: chapter.text
            for doc_id, chapter in self.data_loader._corpus.items()
        }
        
        # Build index
        self.logger.info(f"Building index for {len(documents)} documents...")
        
        self.retrieval_system.build_index(
            documents,
            batch_size=self.config.retrieval.batch_size,
            save_path=str(index_path)
        )
        
        self.logger.info(f"✓ Index built and saved to {index_path}")
    
    def step3_build_raft_dataset(
        self,
        split: str = "train",
        max_examples: Optional[int] = None,
        openai_api_key: Optional[str] = None
    ):
        """Step 3: Build RAFT training dataset"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"STEP 3: BUILDING RAFT DATASET ({split} split)")
        self.logger.info("="*60)
        
        # Auto-load data if needed
        if not self.data_loader:
            self.logger.info("Data not loaded yet, loading first...")
            self.step1_load_data()
        
        # Auto-build index if needed
        if not self.retrieval_system:
            self.logger.info("Retrieval index not built yet, building first...")
            self.step2_build_index()
        
        # Initialize CoT generator
        cot_generator = CoTGenerator(
            model=self.config.raft_data.judge_model,
            temperature=self.config.raft_data.judge_temperature,
            max_tokens=self.config.raft_data.judge_max_tokens,
            api_key=openai_api_key
        )
        
        # Initialize dataset builder
        self.dataset_builder = RAFTDatasetBuilder(
            retrieval_system=self.retrieval_system,
            cot_generator=cot_generator,
            oracle_percentage=self.config.raft_data.oracle_percentage,
            num_distractors=self.config.raft_data.num_distractors,
            distractor_pool_size=self.config.raft_data.distractor_pool_size,
            max_quote_length=self.config.raft_data.max_quote_length,
            min_quotes=self.config.raft_data.min_quotes_per_example,
            max_quotes=self.config.raft_data.max_quotes_per_example
        )
        
        # Prepare examples
        self.logger.info("Preparing examples from data loader...")
        examples = prepare_examples_from_loader(self.data_loader, split=split)
        
        if max_examples:
            examples = examples[:max_examples]
            self.logger.info(f"Limited to {max_examples} examples")
        
        # Build RAFT dataset
        output_path = Path(self.config.system.data_dir) / f"raft_{split}.jsonl"
        
        self.logger.info(f"Building RAFT dataset with {len(examples)} examples...")
        self.logger.info(f"Oracle percentage: {self.config.raft_data.oracle_percentage}")
        self.logger.info(f"Distractors per example: {self.config.raft_data.num_distractors}")
        
        raft_examples = self.dataset_builder.build_dataset(
            examples,
            str(output_path),
            max_examples=None  # Already limited above
        )
        
        # Print statistics
        stats = self.dataset_builder.get_statistics(raft_examples)
        self.logger.info(f"\nRAFT Dataset Statistics:")
        self.logger.info(json.dumps(stats, indent=2))
        
        self.logger.info(f"✓ RAFT dataset saved to {output_path}")
        
        return str(output_path)
    
    def step4_train_model(
        self,
        train_path: str,
        eval_path: Optional[str] = None
    ):
        """Step 4: Fine-tune model with RAFT"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 4: TRAINING MODEL")
        self.logger.info("="*60)
        
        # Initialize trainer
        self.trainer = RAFTTrainer(self.config)
        
        # Load model
        self.logger.info("Loading model and applying LoRA...")
        self.trainer.load_model()
        
        # Load datasets
        self.logger.info(f"Loading training data from {train_path}")
        train_dataset, eval_dataset = self.trainer.load_dataset(
            train_path,
            eval_path
        )
        
        # Setup callbacks
        log_file = Path(self.config.system.log_dir) / "training_logs.jsonl"
        callbacks = [LoggingCallback(str(log_file))]
        
        # Train
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.config.training.num_train_epochs}")
        self.logger.info(f"Batch size: {self.config.training.per_device_train_batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        self.trainer.train(
            train_dataset,
            eval_dataset,
            callbacks=callbacks
        )
        
        self.logger.info("✓ Training complete")
    
    def step5_save_model(
        self,
        output_dir: Optional[str] = None,
        save_method: str = "merged_16bit"
    ):
        """Step 5: Save trained model"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 5: SAVING MODEL")
        self.logger.info("="*60)
        
        if not self.trainer:
            raise ValueError("Must train model first (step4_train_model)")
        
        output_dir = output_dir or str(Path(self.config.training.output_dir) / "final_model")
        
        self.logger.info(f"Saving model to {output_dir} (method: {save_method})")
        self.trainer.save_model(output_dir, save_method=save_method)
        
        self.logger.info(f"✓ Model saved to {output_dir}")
    
    def step6_evaluate(
        self,
        test_split: str = "test",
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Step 6: Evaluate the model"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"STEP 6: EVALUATION ({test_split} split)")
        self.logger.info("="*60)
        
        # Load model if path provided
        if model_path:
            self.logger.info(f"Loading model from {model_path}")
            from unsloth import FastLanguageModel
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.config.model.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)  # Enable inference mode
        elif self.trainer:
            model = self.trainer.model
            tokenizer = self.trainer.tokenizer
            # Enable inference mode
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        else:
            raise ValueError("Must provide model_path or train model first")
        
        # Initialize evaluator
        self.evaluator = RAFTEvaluator(
            config=self.config,
            retrieval_system=self.retrieval_system,
            model=model,
            tokenizer=tokenizer,
            openai_api_key=openai_api_key
        )
        
        # Prepare test examples
        self.logger.info("Preparing test examples...")
        test_examples = prepare_examples_from_loader(
            self.data_loader,
            split=test_split
        )
        
        # Convert to evaluation format
        eval_examples = []
        for ex in test_examples:
            eval_examples.append({
                'query': ex['query'],
                'gold_docs': [doc_id for doc_id, _ in ex['oracle_docs']],
                'answers': ex['answers']
            })
        
        # Evaluate
        self.logger.info(f"Evaluating {len(eval_examples)} examples...")
        
        output_path = output_path or str(
            Path(self.config.system.results_dir) / f"eval_{test_split}.jsonl"
        )
        
        metrics = self.evaluator.evaluate_dataset(
            eval_examples,
            output_path=output_path
        )
        
        # Print results
        self.evaluator.print_metrics(metrics)
        
            # Save metrics
        metrics_path = Path(self.config.system.results_dir) / f"metrics_{test_split}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"✓ Evaluation complete. Results saved to {output_path}")
        
        return metrics
    
    def run_full_pipeline(
        self,
        train_max_examples: Optional[int] = None,
        eval_max_examples: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        skip_training: bool = False
    ):
        """Run complete end-to-end pipeline"""
        self.logger.info("\n" + "="*80)
        self.logger.info("RUNNING FULL RAFT PIPELINE")
        self.logger.info("="*80)
        
        try:
            # Step 1: Load data
            self.step1_load_data()
            
            # Step 2: Build index
            self.step2_build_index()
            
            # Step 3: Build RAFT datasets
            train_path = self.step3_build_raft_dataset(
                split="train",
                max_examples=train_max_examples,
                openai_api_key=openai_api_key
            )
            
            eval_path = None
            if not skip_training:
                eval_path = self.step3_build_raft_dataset(
                    split="dev",
                    max_examples=eval_max_examples,
                    openai_api_key=openai_api_key
                )
            
            if not skip_training:
                # Step 4: Train model
                self.step4_train_model(train_path, eval_path)
                
                # Step 5: Save model
                self.step5_save_model()
            
            # Step 6: Evaluate
            metrics = self.step6_evaluate(test_split="test", openai_api_key=openai_api_key)
            
            self.logger.info("\n" + "="*80)
            self.logger.info("PIPELINE COMPLETE!")
            self.logger.info("="*80)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point for RAFT pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAFT Fine-tuning Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "data", "index", "dataset", "train", "eval"],
        default="all",
        help="Pipeline step to run"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "validation", "test"],
        default="train",
        help="Dataset split to build (for --step dataset)"
    )
    parser.add_argument(
        "--train-max-examples",
        type=int,
        help="Max training examples (for testing)"
    )
    parser.add_argument(
        "--eval-max-examples",
        type=int,
        help="Max eval examples (for testing)"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key for CoT generation"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model for evaluation"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (for evaluation only)"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = RAFTConfig.from_yaml(args.config)
    else:
        config = RAFTConfig()
    
    # Initialize pipeline
    pipeline = RAFTPipeline(config)
    
    try:
        # Run requested step(s)
        if args.step == "all":
            pipeline.run_full_pipeline(
                train_max_examples=args.train_max_examples,
                eval_max_examples=args.eval_max_examples,
                openai_api_key=args.openai_api_key,
                skip_training=args.skip_training
            )
        
        elif args.step == "data":
            pipeline.step1_load_data()
        
        elif args.step == "index":
            # Load data first if not already loaded
            if pipeline.data_loader is None:
                logger.info("Data not loaded, loading first...")
                pipeline.step1_load_data()
            pipeline.step2_build_index()
        
        elif args.step == "dataset":
            # Load data and build index first if needed
            if pipeline.data_loader is None:
                logger.info("Data not loaded, loading first...")
                pipeline.step1_load_data()
            if pipeline.retrieval_system is None:
                logger.info("Index not built, building first...")
                pipeline.step2_build_index()

            # Normalize split name (validation -> dev)
            split = args.split
            if split == "validation":
                split = "dev"

            # Determine max_examples based on split
            if split == "train":
                max_examples = args.train_max_examples
            else:
                max_examples = args.eval_max_examples

            pipeline.step3_build_raft_dataset(
                split=split,
                max_examples=max_examples,
                openai_api_key=args.openai_api_key
            )
        
        elif args.step == "train":
            # Assumes dataset already built
            train_path = str(Path(config.system.data_dir) / "raft_train.jsonl")
            eval_path = str(Path(config.system.data_dir) / "raft_dev.jsonl")
            
            if not Path(train_path).exists():
                pipeline.logger.error(f"Training data not found: {train_path}")
                pipeline.logger.info("Run with --step dataset first")
                sys.exit(1)
            
            pipeline.step4_train_model(train_path, eval_path)
            pipeline.step5_save_model()
        
        elif args.step == "eval":
            # Load data first if needed
            if pipeline.data_loader is None:
                logger.info("Data not loaded, loading first...")
                pipeline.step1_load_data()
            
            if not args.model_path:
                # Try to use default trained model path
                model_path = str(Path(config.training.output_dir) / "final_model")
                if not Path(model_path).exists():
                    pipeline.logger.error("No model found. Provide --model-path")
                    sys.exit(1)
            else:
                model_path = args.model_path
            
            # Load or build index if available
            index_path = str(Path(config.retrieval.index_path) / "raft_index")
            if Path(index_path + ".faiss").exists():
                logger.info("Loading existing index...")
                pipeline.retrieval_system = RetrievalSystem(
                    embedding_model=config.retrieval.embedding_model,
                    reranker_model=config.retrieval.reranker_model if config.retrieval.use_reranker else None,
                    index_path=index_path
                )
            else:
                logger.info("Index not found, building first...")
                pipeline.step2_build_index()
            
            pipeline.step6_evaluate(model_path=model_path)
        
        pipeline.logger.info("\n✓ All requested steps completed successfully!")
        
    except KeyboardInterrupt:
        pipeline.logger.info("\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        pipeline.logger.error(f"\n✗ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
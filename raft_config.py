"""
RAFT Fine-tuning Configuration Module
Centralizes all configuration for the RAFT training pipeline
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import logging


@dataclass
class ModelConfig:
    """Model and training configuration"""
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    full_finetuning: bool = False
    
    # LoRA parameters
    lora_r: int = 32
    lora_alpha: int = 64  # 2 * lora_r
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        if self.lora_alpha != 2 * self.lora_r:
            logging.warning(
                f"lora_alpha ({self.lora_alpha}) is not 2*lora_r ({self.lora_r}). "
                f"Setting to {2 * self.lora_r}"
            )
            self.lora_alpha = 2 * self.lora_r


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    output_dir: str = "./raft_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: str = "unsloth"
    
    # Optimizer
    optim: str = "paged_adamw_32bit"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Scheduler
    lr_scheduler_type: str = "cosine"
    
    # Logging and evaluation
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 400  # Multiple of eval_steps (200)
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    
    # Mixed precision
    fp16: bool = True  # Works on T4 and newer GPUs
    bf16: bool = False  # Requires Ampere+ (A100, A10, etc.)
    
    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    
    # Early stopping
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    group_by_length: bool = False
    report_to: str = "tensorboard"
    
    def __post_init__(self):
        """Validate training configuration"""
        # Ensure save_steps is compatible with eval_steps
        if (self.load_best_model_at_end and 
            self.eval_strategy != "no" and
            self.save_steps % self.eval_steps != 0):
            # Auto-adjust save_steps to be a multiple of eval_steps
            old_save_steps = self.save_steps
            self.save_steps = self.eval_steps * (self.save_steps // self.eval_steps)
            if self.save_steps == 0:
                self.save_steps = self.eval_steps
            logging.warning(
                f"Auto-adjusted save_steps from {old_save_steps} to {self.save_steps} "
                f"(must be multiple of eval_steps={self.eval_steps})"
            )


@dataclass
class RAFTDataConfig:
    """RAFT dataset configuration"""
    dataset_name: str = "yale-nlp/MSRS"
    dataset_config: str = "story-qa"
    oracle_percentage: float = 0.8  # P in RAFT paper
    num_distractors: int = 4
    distractor_pool_size: int = 20
    
    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 200
    
    # Quote validation
    max_quote_length: int = 300
    min_quotes_per_example: int = 1
    max_quotes_per_example: int = 3
    
    # Judge model for CoT generation
    judge_model: str = "gpt-4-turbo-preview"
    judge_temperature: float = 0.2
    judge_max_tokens: int = 1500
    
    def validate(self):
        """Validate RAFT configuration"""
        assert 0.0 <= self.oracle_percentage <= 1.0, \
            "oracle_percentage must be between 0 and 1"
        assert self.num_distractors > 0, \
            "num_distractors must be positive"
        assert self.distractor_pool_size >= self.num_distractors, \
            "distractor_pool_size must be >= num_distractors"


@dataclass
class RetrievalConfig:
    """Retrieval system configuration"""
    embedding_model: str = "BAAI/bge-m3"
    index_type: str = "faiss"  # faiss or qdrant
    similarity_metric: str = "cosine"
    top_k: int = 6
    
    # Re-ranking
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 4
    
    # Index settings
    index_path: str = "./indices"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    eval_batch_size: int = 8
    
    # Retrieval metrics
    compute_context_precision: bool = True
    compute_context_recall: bool = True
    compute_context_relevance: bool = True
    
    # Generation metrics
    compute_faithfulness: bool = True
    compute_answer_relevance: bool = True
    compute_answer_correctness: bool = True
    
    # Traditional metrics
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_bertscore: bool = True
    
    # RAGAS settings
    ragas_llm: str = "gpt-4-turbo-preview"
    ragas_embeddings: str = "text-embedding-3-large"
    
    # Test scenarios
    test_with_perfect_retrieval: bool = True
    test_with_distractors: bool = True
    num_stress_test_distractors: int = 3


@dataclass
class SystemConfig:
    """Overall system configuration"""
    project_name: str = "raft-story-qa"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    results_dir: str = "./results"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # GPU
    use_flash_attention: bool = True
    device_map: str = "auto"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    def setup_directories(self):
        """Create all necessary directories"""
        for dir_path in [
            self.data_dir,
            self.cache_dir,
            self.log_dir,
            self.results_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class RAFTConfig:
    """Main configuration class that combines all configs"""
    
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        training: Optional[TrainingConfig] = None,
        raft_data: Optional[RAFTDataConfig] = None,
        retrieval: Optional[RetrievalConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        system: Optional[SystemConfig] = None
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.raft_data = raft_data or RAFTDataConfig()
        self.retrieval = retrieval or RetrievalConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.system = system or SystemConfig()
        
        # Validate configurations
        self.raft_data.validate()
        self.system.setup_directories()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RAFTConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            raft_data=RAFTDataConfig(**config_dict.get('raft_data', {})),
            retrieval=RetrievalConfig(**config_dict.get('retrieval', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'raft_data': self.raft_data.__dict__,
            'retrieval': self.retrieval.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'raft_data': self.raft_data.__dict__,
            'retrieval': self.retrieval.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__
        }


def setup_logging(config: SystemConfig) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("RAFT")
    logger.setLevel(getattr(logging, config.log_level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if config.log_to_file:
        log_file = Path(config.log_dir) / f"{config.project_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Example usage and defaults
if __name__ == "__main__":
    # Create default configuration
    config = RAFTConfig()
    
    # Save to YAML
    config.to_yaml("raft_config.yaml")
    
    # Setup logging
    logger = setup_logging(config.system)
    logger.info("Configuration initialized successfully")
    
    # Print configuration
    import json
    print(json.dumps(config.to_dict(), indent=2))
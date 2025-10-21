"""
RAFT Trainer Module
Fine-tunes Qwen3-4B-Instruct with Unsloth QLoRA
"""

import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

import torch
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
import wandb

logger = logging.getLogger("RAFT.Trainer")


class RAFTTrainer:
    """RAFT fine-tuning with Unsloth"""
    
    def __init__(self, config):
        """
        Initialize RAFT trainer
        
        Args:
            config: RAFTConfig object with all settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup environment
        self._setup_environment()
        
        logger.info("Initialized RAFT Trainer")
    
    def _setup_environment(self):
        """Setup training environment"""
        # Set seed
        torch.manual_seed(self.config.system.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.system.seed)
        
        # Setup wandb if enabled
        if self.config.system.use_wandb:
            wandb.init(
                project=self.config.system.wandb_project or self.config.system.project_name,
                entity=self.config.system.wandb_entity,
                config=self.config.to_dict()
            )
    
    def load_model(self):
        """Load model and tokenizer with Unsloth"""
        try:
            from unsloth import FastLanguageModel
            
            logger.info(f"Loading model: {self.config.model.model_name}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model.model_name,
                max_seq_length=self.config.model.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=self.config.model.load_in_4bit,
                device_map=self.config.system.device_map
            )
            
            logger.info("Model loaded successfully")
            
            # Apply LoRA
            self._apply_lora()
            
            # Setup chat template
            self._setup_chat_template()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _apply_lora(self):
        """Apply LoRA adapters"""
        try:
            from unsloth import FastLanguageModel
            
            logger.info("Applying LoRA adapters")
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                bias=self.config.model.lora_bias,
                target_modules=self.config.model.lora_target_modules,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing,
                random_state=self.config.system.seed,
                use_rslora=False,
                loftq_config=None,
            )
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {str(e)}")
            raise
    
    def _setup_chat_template(self):
        """Setup chat template for tokenizer"""
        if self.tokenizer.chat_template is None:
            # Default chat template for Qwen3
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
                "{% endif %}"
            )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def load_dataset(
        self,
        train_path: str,
        eval_path: Optional[str] = None
    ) -> tuple:
        """
        Load training and evaluation datasets
        
        Args:
            train_path: Path to training JSONL file
            eval_path: Path to evaluation JSONL file
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info(f"Loading training dataset from {train_path}")
        
        if not Path(train_path).exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        # Load from JSONL
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))
        
        train_dataset = Dataset.from_list(train_data)
        logger.info(f"Loaded {len(train_dataset)} training examples")
        
        eval_dataset = None
        if eval_path:
            if not Path(eval_path).exists():
                logger.warning(f"Evaluation data not found: {eval_path}")
            else:
                logger.info(f"Loading evaluation dataset from {eval_path}")
                eval_data = []
                with open(eval_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        eval_data.append(json.loads(line))
                
                eval_dataset = Dataset.from_list(eval_data)
                logger.info(f"Loaded {len(eval_dataset)} evaluation examples")
        
        return train_dataset, eval_dataset
    
    def prepare_training_args(self) -> TrainingArguments:
        """Prepare training arguments"""
        
        # Ensure save_steps is a multiple of eval_steps when load_best_model_at_end is True
        save_steps = self.config.training.save_steps
        eval_steps = self.config.training.eval_steps
        
        if (self.config.training.load_best_model_at_end and 
            self.config.training.eval_strategy != "no" and
            save_steps % eval_steps != 0):
            # Adjust save_steps to be a multiple of eval_steps
            save_steps = eval_steps * (save_steps // eval_steps)
            if save_steps == 0:
                save_steps = eval_steps
            logger.warning(
                f"Adjusted save_steps from {self.config.training.save_steps} to {save_steps} "
                f"to be a multiple of eval_steps ({eval_steps})"
            )
        
        # Auto-detect GPU capabilities and adjust precision settings
        fp16 = self.config.training.fp16
        bf16 = self.config.training.bf16
        
        if torch.cuda.is_available():
            # Get GPU compute capability
            capability = torch.cuda.get_device_capability()
            gpu_name = torch.cuda.get_device_name(0)
            
            # BF16 requires compute capability >= 8.0 (Ampere+)
            # T4 is compute capability 7.5, so it doesn't support BF16
            if bf16 and capability[0] < 8:
                logger.warning(
                    f"BF16 not supported on {gpu_name} (compute capability {capability[0]}.{capability[1]}). "
                    f"Switching to FP16."
                )
                bf16 = False
                fp16 = True
            
            # If neither is set and GPU supports it, use FP16
            if not fp16 and not bf16:
                logger.info(f"Enabling FP16 for {gpu_name}")
                fp16 = True
        
        args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            gradient_checkpointing=True,
            
            # Optimizer
            optim=self.config.training.optim,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            max_grad_norm=self.config.training.max_grad_norm,
            
            # Scheduler
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            
            # Logging
            logging_steps=self.config.training.logging_steps,
            logging_dir=os.path.join(self.config.training.output_dir, "logs"),
            
            # Evaluation
            eval_strategy=self.config.training.eval_strategy,
            eval_steps=eval_steps if self.config.training.eval_strategy != "no" else None,
            
            # Saving
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=self.config.training.save_total_limit,
            
            # Mixed precision (auto-adjusted for GPU)
            fp16=fp16,
            bf16=bf16,
            
            # Early stopping
            load_best_model_at_end=self.config.training.load_best_model_at_end if self.config.training.eval_strategy != "no" else False,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            
            # Misc
            seed=self.config.system.seed,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            group_by_length=self.config.training.group_by_length,
            report_to=self.config.training.report_to if self.config.system.use_wandb else "none",
            
            # Disable things that cause issues
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        )
        
        return args
    
    def formatting_func(self, examples: Dict[str, List]) -> List[str]:
        """
        Format examples for training
        
        Args:
            examples: Batch of examples with 'messages' field
            
        Returns:
            List of formatted strings
        """
        texts = []
        for messages in examples['messages']:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        return texts
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None
    ):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of callbacks
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Preparing trainer...")
        
        # Prepare training arguments
        training_args = self.prepare_training_args()
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=self.formatting_func,
            max_seq_length=self.config.model.max_seq_length,
            packing=False,  # Don't pack examples
            callbacks=callbacks or [],
        )
        
        logger.info("Starting training...")
        logger.info(f"  Num examples: {len(train_dataset)}")
        logger.info(f"  Num epochs: {self.config.training.num_train_epochs}")
        logger.info(f"  Batch size: {self.config.training.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total batch size: {self.config.training.per_device_train_batch_size * self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {self.config.training.learning_rate}")
        
        # Train
        train_result = self.trainer.train()
        
        logger.info("Training complete!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save training metrics
        metrics_path = Path(self.config.training.output_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        return train_result
    
    def save_model(self, output_dir: str, save_method: str = "merged_16bit"):
        """
        Save the trained model
        
        Args:
            output_dir: Directory to save model
            save_method: Save method (lora, merged_16bit, merged_4bit)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        logger.info(f"Saving model to {output_dir} (method: {save_method})")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            if save_method == "lora":
                # Save only LoRA adapters
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                logger.info("Saved LoRA adapters")
                
            elif save_method == "merged_16bit":
                # Save merged model in 16bit
                self.model.save_pretrained_merged(
                    output_dir,
                    self.tokenizer,
                    save_method="merged_16bit"
                )
                logger.info("Saved merged 16bit model")
                
            elif save_method == "merged_4bit":
                # Save merged model in 4bit
                self.model.save_pretrained_merged(
                    output_dir,
                    self.tokenizer,
                    save_method="merged_4bit_forced"
                )
                logger.info("Saved merged 4bit model")
            
            else:
                raise ValueError(f"Unknown save_method: {save_method}")
            
            logger.info(f"Model saved successfully to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            # Fallback: save as LoRA adapters
            logger.info("Falling back to saving LoRA adapters only")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary of metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        logger.info("Evaluating model...")
        
        metrics = self.trainer.evaluate(eval_dataset)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics


class LoggingCallback(TrainerCallback):
    """Custom logging callback"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics"""
        if logs:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps({
                    'step': state.global_step,
                    'epoch': state.epoch,
                    **logs
                }) + '\n')


# Example usage
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Check if running in demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("Demo mode: RAFT Trainer")
        print("\nTo use RAFT Trainer:")
        print("1. Install Unsloth: pip install unsloth unsloth_zoo")
        print("2. Create RAFTConfig with desired settings")
        print("3. Initialize RAFTTrainer(config)")
        print("4. Load model: trainer.load_model()")
        print("5. Load dataset: train_ds, eval_ds = trainer.load_dataset(paths)")
        print("6. Train: trainer.train(train_ds, eval_ds)")
        print("7. Save: trainer.save_model(output_dir)")
        
        print("\nExample configuration:")
        from raft_config import RAFTConfig
        config = RAFTConfig()
        print(f"  Model: {config.model.model_name}")
        print(f"  LoRA r: {config.model.lora_r}")
        print(f"  Max seq length: {config.model.max_seq_length}")
        print(f"  Epochs: {config.training.num_train_epochs}")
        print(f"  Batch size: {config.training.per_device_train_batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")
    else:
        print("Run with --demo flag for demo mode")
        print("Full training requires GPU and prepared RAFT dataset")

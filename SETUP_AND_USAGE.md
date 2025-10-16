# RAFT Fine-tuning Setup and Usage Guide

Complete guide for setting up and running RAFT fine-tuning on Qwen3-4B-Instruct for MSRS Story-QA.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [Step-by-Step Usage](#step-by-step-usage)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on T4, V100, A100)
  - Single 16GB GPU sufficient with 4-bit quantization
  - Multiple GPUs supported for distributed training
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 50GB+ free disk space

### Software

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.10 or 3.11
- **CUDA**: 11.8 or 12.1+ (for GPU support)
- **Driver**: NVIDIA driver 525.60.13+

---

## Installation

### 1. Clone/Create Project

```bash
# Create project directory
mkdir MSRS-RAFT
cd MSRS-RAFT

# Copy all Python modules to this directory
# - raft_config.py
# - raft_data_loader.py
# - raft_retrieval.py
# - raft_dataset_builder.py
# - raft_trainer.py
# - raft_evaluator.py
# - raft_pipeline.py
# - requirements.txt
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 3. Install PyTorch

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (not recommended for training)
pip install torch torchvision torchaudio
```

### 4. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Upgrade Unsloth to latest
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Install NLTK data (for BLEU)
python -c "import nltk; nltk.download('punkt')"
```

### 5. Setup API Keys

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
WANDB_API_KEY=your_wandb_key_here  # Optional
HF_TOKEN=your_huggingface_token_here  # Optional, for private models
EOF

# Load environment variables
export $(cat .env | xargs)
```

---

## Configuration

### Create Configuration File

```bash
# Generate default config
python raft_config.py

# This creates raft_config.yaml
```

### Customize Configuration

Edit `raft_config.yaml`:

```yaml
model:
  model_name: "Qwen/Qwen3-4B-Instruct-2507"
  max_seq_length: 4096
  lora_r: 32
  lora_alpha: 64

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 0.0002
  warmup_ratio: 0.03

raft_data:
  oracle_percentage: 0.8
  num_distractors: 4
  judge_model: "gpt-4-turbo-preview"

retrieval:
  embedding_model: "BAAI/bge-m3"
  use_reranker: true
  reranker_model: "BAAI/bge-reranker-v2-m3"

evaluation:
  compute_faithfulness: true
  compute_answer_relevance: true
  compute_rouge: true

system:
  project_name: "MSRS-RAFT"
  use_wandb: false  # Set to true for experiment tracking
```

---

## Quick Start

### Option 1: Run Full Pipeline

```bash
# Run complete pipeline (data loading, indexing, dataset creation, training, evaluation)
python raft_pipeline.py \
  --step all \
  --config raft_config.yaml \
  --train-max-examples 50 \
  --eval-max-examples 20 \
  --openai-api-key $OPENAI_API_KEY
```

### Option 2: Run Individual Steps

```bash
# 1. Load data and build index
python raft_pipeline.py --step index --config raft_config.yaml

# 2. Build RAFT training dataset
python raft_pipeline.py \
  --step dataset \
  --config raft_config.yaml \
  --train-max-examples 100 \
  --openai-api-key $OPENAI_API_KEY

# 3. Train model
python raft_pipeline.py --step train --config raft_config.yaml

# 4. Evaluate
python raft_pipeline.py --step eval --config raft_config.yaml
```

---

## Step-by-Step Usage

### Step 1: Data Loading

```python
from raft_config import RAFTConfig
from raft_data_loader import MSRSDataLoader

# Load configuration
config = RAFTConfig.from_yaml("raft_config.yaml")

# Initialize data loader
loader = MSRSDataLoader(
    dataset_name="yale-nlp/MSRS",
    dataset_config="story-qa",
    cache_dir="./cache"
)

# Load dataset
dataset = loader.load_dataset()

# Parse examples
train_examples = loader.parse_examples(split="train")
print(f"Loaded {len(train_examples)} training examples")

# Load corpus (story chapters)
corpus = loader.load_corpus()
print(f"Loaded {len(corpus)} chapters")
```

### Step 2: Build Retrieval Index

```python
from raft_retrieval import RetrievalSystem

# Initialize retrieval system
retrieval_system = RetrievalSystem(
    embedding_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    chunk_size=1500,
    chunk_overlap=200
)

# Prepare documents
documents = {
    doc_id: chapter.text
    for doc_id, chapter in loader._corpus.items()
}

# Build index
retrieval_system.build_index(
    documents,
    batch_size=32,
    save_path="./indices/raft_index"
)

# Test retrieval
query = "What happened in the story?"
results = retrieval_system.retrieve(query, top_k=5)
for result in results:
    print(f"Doc: {result.doc_id}, Score: {result.score:.4f}")
```

### Step 3: Build RAFT Dataset

```python
from raft_dataset_builder import CoTGenerator, RAFTDatasetBuilder, prepare_examples_from_loader

# Initialize CoT generator (requires OpenAI API key)
cot_generator = CoTGenerator(
    model="gpt-4-turbo-preview",
    temperature=0.2,
    api_key="your_openai_key"
)

# Initialize RAFT builder
raft_builder = RAFTDatasetBuilder(
    retrieval_system=retrieval_system,
    cot_generator=cot_generator,
    oracle_percentage=0.8,
    num_distractors=4
)

# Prepare examples
examples = prepare_examples_from_loader(loader, split="train")

# Build RAFT dataset
raft_examples = raft_builder.build_dataset(
    examples[:50],  # Start with 50 examples
    output_path="./data/raft_train.jsonl"
)

# View statistics
stats = raft_builder.get_statistics(raft_examples)
print(stats)
```

### Step 4: Train Model

```python
from raft_trainer import RAFTTrainer, LoggingCallback

# Initialize trainer
trainer = RAFTTrainer(config)

# Load model with LoRA
trainer.load_model()

# Load datasets
train_dataset, eval_dataset = trainer.load_dataset(
    train_path="./data/raft_train.jsonl",
    eval_path="./data/raft_dev.jsonl"
)

# Setup callbacks
callbacks = [LoggingCallback("./logs/training.jsonl")]

# Train
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=callbacks
)

# Save model
trainer.save_model(
    output_dir="./models/raft_qwen3_4b",
    save_method="merged_16bit"
)
```

### Step 5: Evaluate Model

```python
from raft_evaluator import RAFTEvaluator
from unsloth import FastLanguageModel

# Load trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/raft_qwen3_4b",
    max_seq_length=4096,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)

# Initialize evaluator
evaluator = RAFTEvaluator(
    config=config,
    retrieval_system=retrieval_system,
    model=model,
    tokenizer=tokenizer
)

# Prepare test examples
test_examples = prepare_examples_from_loader(loader, split="test")

eval_examples = [
    {
        'query': ex['query'],
        'gold_docs': [doc_id for doc_id, _ in ex['oracle_docs']],
        'answers': ex['answers']
    }
    for ex in test_examples[:20]
]

# Evaluate
metrics = evaluator.evaluate_dataset(
    eval_examples,
    output_path="./results/evaluation.jsonl"
)

# Print results
evaluator.print_metrics(metrics)
```

---

## Advanced Usage

### Custom RAFT Data Generation

```python
# Control oracle percentage for different scenarios
scenarios = [
    ("high_oracle", 0.9),
    ("medium_oracle", 0.7),
    ("low_oracle", 0.5)
]

for name, percentage in scenarios:
    raft_builder.oracle_percentage = percentage
    raft_examples = raft_builder.build_dataset(
        examples,
        output_path=f"./data/raft_{name}.jsonl"
    )
```

### Multi-System Comparison

```python
# Evaluate multiple systems
systems = {
    "0-shot": evaluate_zero_shot(),
    "SFT": evaluate_sft_model(),
    "RAFT": evaluate_raft_model()
}

comparison = evaluator.compare_systems(systems)

# Print comparison
for system_name, metrics in comparison.items():
    print(f"\n{system_name}:")
    evaluator.print_metrics(metrics)
```

### Stress Testing with Distractors

```python
# Add extra distractors to test robustness
def add_stress_distractors(examples, num_extra=3):
    for ex in examples:
        # Retrieve additional high-scoring but wrong docs
        extra_results = retrieval_system.retrieve(
            ex['query'],
            top_k=num_extra + len(ex['gold_docs'])
        )
        
        distractors = [
            r for r in extra_results
            if r.doc_id not in ex['gold_docs']
        ][:num_extra]
        
        ex['retrieved_docs'].extend([d.doc_id for d in distractors])
        ex['retrieved_texts'].extend([d.text for d in distractors])
    
    return examples

# Evaluate under stress
stress_examples = add_stress_distractors(eval_examples.copy(), num_extra=3)
stress_metrics = evaluator.evaluate_dataset(stress_examples)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```python
# Reduce batch size
config.training.per_device_train_batch_size = 1
config.training.gradient_accumulation_steps = 8

# Reduce sequence length
config.model.max_seq_length = 2048

# Enable gradient checkpointing
config.training.gradient_checkpointing = "unsloth"
```

#### 2. Slow Training

```python
# Use Flash Attention 2
config.system.use_flash_attention = True

# Increase batch size if GPU allows
config.training.per_device_train_batch_size = 4

# Disable eval during training
config.training.eval_strategy = "no"
```

#### 3. CUDA Out of Memory During Indexing

```python
# Reduce embedding batch size
retrieval_system.build_index(
    documents,
    batch_size=16,  # Reduced from 32
    save_path="./indices/raft_index"
)

# Or use CPU for embedding
retrieval_system = RetrievalSystem(
    embedding_model="BAAI/bge-m3",
    device="cpu"
)
```

#### 4. OpenAI API Rate Limits

```python
# Add delay between API calls
import time

class RateLimitedCoTGenerator(CoTGenerator):
    def generate_cot(self, *args, **kwargs):
        result = super().generate_cot(*args, **kwargs)
        time.sleep(1)  # 1 second delay
        return result
```

#### 5. Missing NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Logging and Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log to file
logger = logging.getLogger("RAFT")
handler = logging.FileHandler("raft_debug.log")
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
```

### Performance Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training
tensorboard --logdir ./raft_checkpoints/logs

# Monitor with wandb
python raft_pipeline.py --config raft_config.yaml --step all
# View at https://wandb.ai
```

---

## Expected Results

Based on the RAFT paper, you should see:

- **Faithfulness**: >0.7 (RAFT model grounds answers in context)
- **Answer Relevance**: >0.75 (answers address the query)
- **ROUGE-L**: Improvement over 0-shot and SFT baselines
- **Robustness**: RAFT outperforms under distractor stress tests

### Typical Timeline

- **Data Loading**: 5-10 minutes
- **Index Building**: 10-20 minutes (depends on corpus size)
- **RAFT Dataset Generation**: 2-5 hours (depends on # examples and API speed)
- **Training**: 2-8 hours (depends on dataset size and GPU)
- **Evaluation**: 30-60 minutes

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{raft2024,
  title={RAFT: Adapting Language Model to Domain Specific RAG},
  author={Zhang, Tianjun and others},
  journal={arXiv preprint arXiv:2403.10131},
  year={2024}
}

@article{msrs2024,
  title={MSRS: A Benchmark for Multi-Source Retrieval and Synthesis},
  author={Yale NLP Group},
  year={2024}
}
```

---

## Support

For issues and questions:

1. Check logs in `./logs/`
2. Review configuration in `raft_config.yaml`
3. Consult [Unsloth documentation](https://docs.unsloth.ai/)
4. Check [RAFT paper](https://arxiv.org/abs/2403.10131)
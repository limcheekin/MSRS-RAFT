# RAFT Project Structure

Complete file organization and description of the RAFT fine-tuning system.

## 📁 Project Layout

```
MSRS-RAFT/
├── README.md                      # Main project documentation
├── SETUP_AND_USAGE.md            # Detailed setup and usage guide
├── PROJECT_STRUCTURE.md          # This file
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
├── .gitignore                   # Git ignore file (create this)
│
├── Core Modules (Required)
├── raft_config.py               # ✓ Configuration management
├── raft_data_loader.py          # ✓ MSRS dataset loading
├── raft_retrieval.py            # ✓ Vector search & retrieval
├── raft_dataset_builder.py      # ✓ RAFT training data creation
├── raft_trainer.py              # ✓ Unsloth QLoRA fine-tuning
├── raft_evaluator.py            # ✓ RAG metrics evaluation
├── raft_pipeline.py             # ✓ End-to-end orchestration
│
├── Utilities
├── example_usage.py             # ✓ Complete usage examples
├── test_installation.py         # ✓ Installation validation
│
├── Configuration Files (Generated)
├── raft_config.yaml             # Default configuration
├── custom_config.yaml           # Custom configurations
│
├── Data Directories (Generated)
├── data/
│   ├── raft_train.jsonl        # RAFT training dataset
│   ├── raft_dev.jsonl          # RAFT dev dataset
│   └── raft_test.jsonl         # RAFT test dataset
│
├── cache/                       # HuggingFace cache
│   └── datasets/               # Cached datasets
│
├── indices/                     # Vector indices
│   ├── raft_index.faiss        # FAISS index file
│   ├── raft_index.mapping.pkl  # Index mapping
│   └── raft_index.chunks.pkl   # Chunk store
│
├── models/                      # Trained models
│   └── raft_story_qa_v1/       # Saved model
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       ├── tokenizer_config.json
│       └── ...
│
├── raft_checkpoints/           # Training checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── logs/                   # TensorBoard logs
│
├── logs/                       # Application logs
│   ├── MSRS-RAFT.log      # Main log file
│   └── training_logs.jsonl    # Training metrics
│
└── results/                    # Evaluation results
    ├── eval_test.jsonl        # Detailed test results
    ├── metrics_test.json      # Aggregated metrics
    ├── comparison.json        # System comparisons
    └── stress_test_*.jsonl    # Stress test results
```

## 📄 Core Module Descriptions

### 1. `raft_config.py` (Configuration Management)

**Purpose**: Centralized configuration for all pipeline components

**Key Classes**:
- `ModelConfig`: Model and LoRA parameters
- `TrainingConfig`: Training hyperparameters
- `RAFTDataConfig`: RAFT dataset generation settings
- `RetrievalConfig`: Embedding and retrieval settings
- `EvaluationConfig`: Evaluation metrics configuration
- `SystemConfig`: Project-wide settings
- `RAFTConfig`: Combined configuration container

**Key Functions**:
- `from_yaml()`: Load config from YAML
- `to_yaml()`: Save config to YAML
- `setup_logging()`: Initialize logging

**Usage**:
```python
config = RAFTConfig.from_yaml("config.yaml")
config.training.learning_rate = 3e-4
config.to_yaml("updated_config.yaml")
```

---

### 2. `raft_data_loader.py` (Dataset Loading)

**Purpose**: Load and process MSRS Story-QA dataset

**Key Classes**:
- `StoryQAExample`: Container for QA examples
- `Chapter`: Container for story chapters
- `MSRSDataLoader`: Main data loading class

**Key Methods**:
- `load_dataset()`: Load from HuggingFace
- `parse_examples()`: Convert to structured format
- `load_corpus()`: Load story chapters
- `get_chapters_by_ids()`: Retrieve specific chapters
- `get_statistics()`: Dataset statistics

**Usage**:
```python
loader = MSRSDataLoader()
loader.load_dataset()
loader.load_corpus()
examples = loader.parse_examples(split="train")
```

---

### 3. `raft_retrieval.py` (Vector Search)

**Purpose**: Build and query vector indices for document retrieval

**Key Classes**:
- `TextChunker`: Split documents into chunks
- `EmbeddingModel`: Encode text to vectors
- `FAISSIndex`: FAISS vector index wrapper
- `Reranker`: Cross-encoder reranking
- `RetrievalSystem`: Complete retrieval pipeline
- `RetrievalResult`: Container for search results

**Key Methods**:
- `build_index()`: Create searchable index
- `retrieve()`: Search for relevant documents
- `save_index()` / `load_index()`: Persistence

**Usage**:
```python
retrieval = RetrievalSystem(embedding_model="BAAI/bge-m3")
retrieval.build_index(documents)
results = retrieval.retrieve(query, top_k=5)
```

---

### 4. `raft_dataset_builder.py` (RAFT Data Generation)

**Purpose**: Create RAFT training examples with CoT and citations

**Key Classes**:
- `CitationValidator`: Validate quote accuracy
- `CoTGenerator`: Generate reasoning with GPT-4
- `RAFTDatasetBuilder`: Build complete RAFT dataset
- `RAFTExample`: Container for RAFT training example

**Key Methods**:
- `build_raft_example()`: Create single RAFT example
- `build_dataset()`: Create full dataset
- `to_chat_format()`: Format for training
- `validate_example()`: Check citation quality

**Usage**:
```python
cot_gen = CoTGenerator(model="gpt-4-turbo-preview")
builder = RAFTDatasetBuilder(retrieval, cot_gen)
examples = builder.build_dataset(qa_examples, "train.jsonl")
```

---

### 5. `raft_trainer.py` (Model Training)

**Purpose**: Fine-tune Qwen3-4B with Unsloth QLoRA

**Key Classes**:
- `RAFTTrainer`: Main training orchestrator
- `LoggingCallback`: Custom training callback

**Key Methods**:
- `load_model()`: Load model with LoRA
- `load_dataset()`: Load training data
- `train()`: Execute training
- `save_model()`: Save trained model
- `evaluate()`: Run evaluation

**Usage**:
```python
trainer = RAFTTrainer(config)
trainer.load_model()
train_ds, eval_ds = trainer.load_dataset("train.jsonl")
trainer.train(train_ds, eval_ds)
trainer.save_model("output/")
```

---

### 6. `raft_evaluator.py` (Evaluation)

**Purpose**: Comprehensive RAG evaluation with multiple metrics

**Key Classes**:
- `RetrievalMetrics`: Context precision/recall
- `GenerationMetrics`: Faithfulness, relevance, ROUGE, BLEU
- `RAFTEvaluator`: Complete evaluation system
- `EvaluationResult`: Container for eval results

**Key Methods**:
- `evaluate_example()`: Evaluate single example
- `evaluate_dataset()`: Evaluate full dataset
- `generate_answer()`: Generate with model
- `aggregate_results()`: Compute mean metrics
- `compare_systems()`: Multi-system comparison

**Usage**:
```python
evaluator = RAFTEvaluator(config, retrieval, model, tokenizer)
metrics = evaluator.evaluate_dataset(test_examples)
evaluator.print_metrics(metrics)
```

---

### 7. `raft_pipeline.py` (Orchestration)

**Purpose**: End-to-end pipeline execution

**Key Classes**:
- `RAFTPipeline`: Main pipeline orchestrator

**Key Methods**:
- `step1_load_data()`: Load dataset
- `step2_build_index()`: Build retrieval index
- `step3_build_raft_dataset()`: Create training data
- `step4_train_model()`: Fine-tune model
- `step5_save_model()`: Save trained model
- `step6_evaluate()`: Run evaluation
- `run_full_pipeline()`: Execute all steps

**Usage**:
```python
pipeline = RAFTPipeline(config)
metrics = pipeline.run_full_pipeline(
    train_max_examples=100,
    openai_api_key="key"
)
```

**Command Line**:
```bash
python raft_pipeline.py --step all --config config.yaml
```

---

## 🔧 Utility Scripts

### `example_usage.py`

Six complete working examples:
1. Quick start (minimal setup)
2. Step-by-step walkthrough
3. Custom configuration
4. Evaluation only
5. Baseline comparison
6. Distractor stress test

### `test_installation.py`

Validates installation:
- Python version
- PyTorch and CUDA
- All dependencies
- Project modules
- GPU memory
- Functionality tests

---

## 📋 Configuration Files

### `raft_config.yaml`

Default configuration with all parameters. Generated by:
```python
from raft_config import RAFTConfig
config = RAFTConfig()
config.to_yaml("raft_config.yaml")
```

### `.env`

Environment variables:
```bash
OPENAI_API_KEY=your_key_here
WANDB_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

---

## 📊 Generated Data

### Training Data Format (JSONL)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful story QA assistant..."
    },
    {
      "role": "user",
      "content": "Question: ...\n\nContext:\n[Chapter: ...]\n..."
    },
    {
      "role": "assistant",
      "content": "##begin_quote##...##end_quote##\n\n##Answer: ..."
    }
  ],
  "metadata": {...}
}
```

### Evaluation Results Format (JSONL)

```json
{
  "query": "What happened?",
  "retrieved_docs": ["doc1", "doc2"],
  "gold_docs": ["doc1"],
  "generated_answer": "The answer is...",
  "reference_answers": ["Reference 1", "Reference 2"],
  "context_precision": 0.75,
  "context_recall": 0.85,
  "faithfulness": 0.82,
  "answer_relevance": 0.78,
  "rouge_l": 0.45,
  "metadata": {...}
}
```

---

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate installation
python test_installation.py

# 3. Run example
python example_usage.py 1

# 4. Run full pipeline
python raft_pipeline.py --step all --train-max-examples 50

# 5. Evaluate model
python raft_pipeline.py --step eval --model-path ./models/raft_story_qa_v1
```

---

## 📦 Dependencies Summary

### Core ML
- torch >= 2.0.0
- transformers >= 4.38.0
- unsloth (latest)
- peft >= 0.8.0
- bitsandbytes >= 0.42.0

### Training
- trl >= 0.7.10
- accelerate >= 0.27.0

### Retrieval
- sentence-transformers >= 2.3.0
- faiss-cpu/gpu >= 1.7.4

### Evaluation
- rouge-score >= 0.1.2
- bert-score >= 0.3.13
- nltk >= 3.8.1

### Utilities
- datasets >= 2.16.0
- openai >= 1.10.0
- numpy, pandas, tqdm, PyYAML

---

## 💾 Disk Space Requirements

- **Cache**: 5-10 GB (datasets, models)
- **Indices**: 1-2 GB (embeddings)
- **Training checkpoints**: 10-20 GB
- **Final model**: 8-10 GB
- **Results**: < 1 GB

**Total**: ~50 GB recommended

---

## 🔒 Git Ignore Recommendations

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data
data/
cache/
indices/
*.jsonl
*.pkl
*.faiss

# Models
models/
raft_checkpoints/
*.bin
*.safetensors

# Logs
logs/
*.log

# Results
results/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

---

## 📝 File Size Reference

| File | Size | Description |
|------|------|-------------|
| raft_config.py | ~12 KB | Configuration |
| raft_data_loader.py | ~15 KB | Data loading |
| raft_retrieval.py | ~20 KB | Retrieval system |
| raft_dataset_builder.py | ~18 KB | RAFT builder |
| raft_trainer.py | ~15 KB | Training |
| raft_evaluator.py | ~18 KB | Evaluation |
| raft_pipeline.py | ~16 KB | Pipeline |
| example_usage.py | ~20 KB | Examples |
| test_installation.py | ~12 KB | Tests |

**Total Source Code**: ~150 KB

---

## ✅ Checklist for Deployment

- [ ] All Python modules in project directory
- [ ] requirements.txt present
- [ ] .env file with API keys
- [ ] Python 3.10+ installed
- [ ] CUDA drivers installed (if using GPU)
- [ ] Run `pip install -r requirements.txt`
- [ ] Run `python test_installation.py`
- [ ] Run `python example_usage.py 1` (quick test)
- [ ] Configure raft_config.yaml
- [ ] Set OPENAI_API_KEY for dataset generation
- [ ] Verify 50GB+ disk space available
- [ ] Verify 16GB+ GPU RAM (if training)

---

**Project complete and ready for production use! 🎉**
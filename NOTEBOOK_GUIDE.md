# 🚀 Google Colab Training Guide

Complete guide for running RAFT fine-tuning on Google Colab with T4 GPU.

## 📋 Quick Start

### 1. Open the Notebook

**Option A: Direct Link**
```
https://colab.research.google.com/github/limcheekin/MSRS-RAFT/blob/main/RAFT_Training_Colab.ipynb
```

**Option B: Upload to Colab**
1. Download `RAFT_Training_Colab.ipynb` from the repository
2. Go to [Google Colab](https://colab.research.google.com/)
3. Click "Upload" and select the notebook

### 2. Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** from Hardware accelerator
3. (Optional) Select **High-RAM** for better performance
4. Click **Save**

### 3. Run the Notebook

Click **Runtime** → **Run all** or run cells sequentially (Ctrl+Enter)

---

## 💰 Cost Estimate

### Google Colab Costs
- **Colab Free**: Limited GPU time (~12 hours/day)
- **Colab Pro**: $9.99/month - Better GPU access
- **Colab Pro+**: $49.99/month - Priority GPU access

### OpenAI API Costs
- **GPT-4 Turbo**: ~$0.01-0.03 per training example
- **20 examples**: ~$0.40
- **100 examples**: ~$2.00
- **500 examples**: ~$10.00

### Recommended Budget
- **Quick Test**: $1-2 (20 examples)
- **Small Dataset**: $5-10 (100 examples)
- **Full Training**: $20-50 (500+ examples)

---

## ⏱️ Time Estimates

### T4 GPU (Free Tier)

| Task | Examples | Time | Notes |
|------|----------|------|-------|
| Setup | - | 5-10 min | Installation & data loading |
| Index Building | All | 10-15 min | One-time setup |
| RAFT Generation | 20 | 30-40 min | Depends on API speed |
| RAFT Generation | 100 | 2-3 hours | Can be interrupted |
| Training | 20 | 30-45 min | 2 epochs |
| Training | 100 | 2-3 hours | 2 epochs |
| Evaluation | 5-10 | 5-10 min | Quick testing |

**Total for 20 examples**: ~1.5-2 hours  
**Total for 100 examples**: ~5-7 hours

---

## 🎯 Recommended Configurations

### Quick Test (Free Tier)
```python
NUM_TRAIN_EXAMPLES = 10
NUM_DEV_EXAMPLES = 3
num_train_epochs = 1
per_device_train_batch_size = 1
max_seq_length = 2048
```
- **Time**: ~1 hour
- **Cost**: ~$0.30
- **Purpose**: Test the pipeline

### Small Training (Colab Pro)
```python
NUM_TRAIN_EXAMPLES = 50
NUM_DEV_EXAMPLES = 10
num_train_epochs = 2
per_device_train_batch_size = 2
max_seq_length = 2048
```
- **Time**: ~3-4 hours
- **Cost**: ~$1.50
- **Purpose**: Initial experiments

### Full Training (Colab Pro+)
```python
NUM_TRAIN_EXAMPLES = 200
NUM_DEV_EXAMPLES = 30
num_train_epochs = 3
per_device_train_batch_size = 2
max_seq_length = 4096
```
- **Time**: ~8-12 hours
- **Cost**: ~$6-10
- **Purpose**: Production model

---

## 🔧 Optimization Tips

### Memory Optimization

**If you get OOM errors:**

1. **Reduce sequence length**
```python
config.model.max_seq_length = 1536  # Lower from 2048
```

2. **Reduce batch size**
```python
config.training.per_device_train_batch_size = 1
config.training.gradient_accumulation_steps = 16  # Keep effective batch
```

3. **Reduce LoRA rank**
```python
config.model.lora_r = 8  # Lower from 16
config.model.lora_alpha = 16
```

4. **Disable reranker**
```python
retrieval_system = RetrievalSystem(
    embedding_model="BAAI/bge-small-en-v1.5",
    reranker_model=None,  # Disable reranker
    ...
)
```

### Speed Optimization

**To train faster:**

1. **Use smaller embedding model**
```python
embedding_model="BAAI/bge-small-en-v1.5"  # Faster than bge-m3
```

2. **Increase batch size** (if you have headroom)
```python
config.training.per_device_train_batch_size = 2
```

3. **Reduce logging frequency**
```python
config.training.logging_steps = 50  # Instead of 10
config.training.eval_steps = 200    # Instead of 50
```

4. **Disable evaluation during training**
```python
config.training.eval_strategy = "no"
```

---

## 📊 Monitoring Training

### TensorBoard (In Notebook)

```python
# Add this cell after training starts
%load_ext tensorboard
%tensorboard --logdir ./raft_checkpoints/logs
```

### Check GPU Usage

```python
# Add this cell anytime
!nvidia-smi
```

### Watch Training Progress

```python
# Tail training logs
!tail -f ./logs/training_colab.jsonl
```

---

## 💾 Saving Your Work

### Option 1: Download to Local

The notebook automatically creates zip files:
- `raft_results.zip` - Logs and metrics
- `raft_model_colab.zip` - Trained model

Download from the Files panel (left sidebar)

### Option 2: Save to Google Drive

Run the last cell to mount Drive and copy files:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy files
!cp -r ./models/raft_qwen3_colab /content/drive/MyDrive/
```

### Option 3: Push to GitHub

```python
# Configure git
!git config --global user.email "your@email.com"
!git config --global user.name "Your Name"

# Commit and push
!git add results/ logs/
!git commit -m "Add training results"
!git push
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Runtime Disconnected

**Problem**: Colab disconnected during training

**Solution**:
- Colab Free has time limits (~12 hours)
- Upgrade to Colab Pro for longer sessions
- Save checkpoints frequently (already configured)
- Use Google Drive backup

#### 2. Out of Memory

**Problem**: `CUDA out of memory` error

**Solutions**:
```python
# Reduce batch size
config.training.per_device_train_batch_size = 1

# Reduce sequence length
config.model.max_seq_length = 1536

# Clear cache
import torch
torch.cuda.empty_cache()

# Restart runtime and try again
```

#### 3. OpenAI API Errors

**Problem**: API rate limits or errors

**Solutions**:
- Check API key is correct
- Verify you have credits
- Add delays between requests (already implemented)
- Use fewer examples initially

#### 4. Slow Dataset Generation

**Problem**: RAFT generation taking too long

**Solutions**:
- Start with fewer examples (10-20)
- Use GPT-3.5-turbo instead of GPT-4:
```python
cot_generator = CoTGenerator(
    model="gpt-3.5-turbo",  # Faster and cheaper
    ...
)
```

#### 5. Import Errors

**Problem**: Module not found errors

**Solution**:
```python
# Restart runtime
# Runtime → Restart runtime

# Reinstall packages
!pip install --upgrade --force-reinstall -r requirements.txt
```

---

## 📈 Expected Results

### Performance Benchmarks (T4 GPU)

| Metric | Value | Notes |
|--------|-------|-------|
| Training Speed | ~3-5 steps/sec | With batch_size=1 |
| Memory Usage | ~10-12 GB | Out of 15 GB |
| Final Loss | 0.5-1.5 | After 2-3 epochs |
| Faithfulness | >0.70 | On dev set |
| Answer Relevance | >0.75 | On dev set |

### Training Logs

Watch for these indicators of good training:

✅ **Good Signs**:
- Loss decreasing steadily
- Evaluation loss following training loss
- GPU utilization >80%
- No OOM errors

⚠️ **Warning Signs**:
- Loss not decreasing after 1 epoch
- Large gap between train and eval loss (overfitting)
- Frequent OOM errors
- Very slow training speed (<1 step/sec)
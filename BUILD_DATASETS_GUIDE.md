# Guide: Building RAFT Dev and Test Datasets

## Summary of Changes

The `raft_pipeline.py` has been modified to add a `--split` argument, allowing you to build specific dataset splits (train, dev, validation, or test).

## New Command-Line Usage

### Build Dev Dataset (50 examples)

```bash
python raft_pipeline.py \
  --step dataset \
  --split dev \
  --eval-max-examples 50 \
  --openai-api-key $OPENAI_API_KEY
```

### Build Test Dataset (100 examples)

```bash
python raft_pipeline.py \
  --step dataset \
  --split test \
  --eval-max-examples 100 \
  --openai-api-key $OPENAI_API_KEY
```

### Build Train Dataset (if needed)

```bash
python raft_pipeline.py \
  --step dataset \
  --split train \
  --train-max-examples 100 \
  --openai-api-key $OPENAI_API_KEY
```

## Complete Workflow

### Step 1: Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Step 2: Build All Three Datasets

```bash
# Build train dataset (100 examples) - if not already done
python raft_pipeline.py \
  --step dataset \
  --split train \
  --train-max-examples 100 \
  --openai-api-key $OPENAI_API_KEY

# Build dev dataset (50 examples)
python raft_pipeline.py \
  --step dataset \
  --split dev \
  --eval-max-examples 50 \
  --openai-api-key $OPENAI_API_KEY

# Build test dataset (100 examples)
python raft_pipeline.py \
  --step dataset \
  --split test \
  --eval-max-examples 100 \
  --openai-api-key $OPENAI_API_KEY
```

### Step 3: Verify the Datasets

```bash
# Check that all files exist
ls -lh data/raft_*.jsonl

# Count examples in each file
echo "Train examples: $(wc -l < data/raft_train.jsonl)"
echo "Dev examples: $(wc -l < data/raft_dev.jsonl)"
echo "Test examples: $(wc -l < data/raft_test.jsonl)"

# View first example from dev dataset
head -n 1 data/raft_dev.jsonl | python -m json.tool
```

## Dataset Size Recommendations

Based on the original MSRS Story-QA dataset proportions:

| Split | Original Size | Your Size | Ratio |
|-------|--------------|-----------|-------|
| Train | 250          | 100       | 40%   |
| Dev   | 125          | 50        | 40%   |
| Test  | 260          | 100       | 38%   |

This maintains approximately the same proportions as the original dataset.

## Arguments Explained

### `--step dataset`
Specifies that you want to build a RAFT dataset.

### `--split {train,dev,validation,test}`
**NEW!** Specifies which split to build:
- `train`: Training dataset
- `dev` or `validation`: Validation dataset (both map to "dev")
- `test`: Test dataset

### `--train-max-examples N`
Maximum number of examples to use when building the **train** split.

### `--eval-max-examples N`
Maximum number of examples to use when building **dev** or **test** splits.

### `--openai-api-key KEY`
Your OpenAI API key for generating Chain-of-Thought reasoning with GPT-4.

## Cost and Time Estimates

### Dev Dataset (50 examples)
- **Cost**: ~$0.50 - $1.50
- **Time**: ~15-25 minutes

### Test Dataset (100 examples)
- **Cost**: ~$1.00 - $3.00
- **Time**: ~30-50 minutes

### Total for Both
- **Cost**: ~$1.50 - $4.50
- **Time**: ~45-75 minutes

## Troubleshooting

### Error: "Data not loaded"
The pipeline will automatically load the data if needed. Just make sure you have internet access to download from HuggingFace.

### Error: "Index not built"
The pipeline will automatically build the retrieval index if needed. This takes 5-10 minutes on first run.

### Error: "OpenAI API key not found"
Make sure you've exported the environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

Or pass it directly:
```bash
python raft_pipeline.py --step dataset --split dev --eval-max-examples 50 --openai-api-key "sk-..."
```

### Error: "No module named 'torch'"
Activate your virtual environment first:
```bash
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

## Output Files

After running the commands, you'll have:

```
data/
├── raft_train.jsonl    # 100 examples (already created)
├── raft_dev.jsonl      # 50 examples (to be created)
└── raft_test.jsonl     # 100 examples (to be created)
```

Each file contains JSONL format with chat-formatted examples including:
- User message with question and context (oracle + distractor documents)
- Assistant message with Chain-of-Thought reasoning and citations
- Quotes marked with `##begin_quote##` and `##end_quote##` tokens

## Next Steps

After building all datasets:

1. **Train the model**:
   ```bash
   python raft_pipeline.py --step train
   ```

2. **Evaluate the model**:
   ```bash
   python raft_pipeline.py --step eval --model-path ./models/raft_story_qa_v1
   ```

## Technical Details

### What Changed in `raft_pipeline.py`

1. **Added `--split` argument** (line 433-439):
   - Accepts: `train`, `dev`, `validation`, or `test`
   - Default: `train`
   - Only used when `--step dataset`

2. **Updated dataset building logic** (line 497-521):
   - Uses `args.split` to determine which split to build
   - Normalizes `validation` → `dev`
   - Uses `--train-max-examples` for train split
   - Uses `--eval-max-examples` for dev/test splits

### Backward Compatibility

The changes are fully backward compatible:
- Old command still works: `python raft_pipeline.py --step dataset` (builds train by default)
- New command with split: `python raft_pipeline.py --step dataset --split dev`


# LLM Judge Implementation

## Overview

This document describes the implementation of LLM-based evaluation judges for the RAFT evaluator, replacing the heuristic-only approach with proper OpenAI API integration.

## Changes Made

### 1. Updated `raft_evaluator.py`

#### Added OpenAI Import
```python
import os

try:
    import openai
except ImportError:
    openai = None
```

#### Enhanced `GenerationMetrics.__init__`
- Added `api_key` parameter to accept OpenAI API key
- Initializes OpenAI client when LLM judge model is specified
- Supports both new OpenAI client (openai>=1.0.0) and legacy API
- Falls back to heuristic evaluation if OpenAI is not available

**Key Features:**
- Auto-detects OpenAI API key from environment if not provided
- Graceful fallback to heuristics if initialization fails
- Logs initialization status for debugging

#### Implemented `_faithfulness_llm_judge`
Evaluates whether generated answers are grounded in the provided contexts using GPT-4.

**Evaluation Criteria:**
- Checks if all claims can be verified from contexts
- Identifies hallucinations or unsupported statements
- Returns score from 0.0 (unfaithful) to 1.0 (fully faithful)

**Prompt Design:**
- Clear instructions for the LLM judge
- Structured context presentation
- Requests numeric score only for easy parsing
- Temperature set to 0.0 for deterministic evaluation

#### Implemented `_relevance_llm_judge`
Evaluates whether generated answers are relevant to the query using GPT-4.

**Evaluation Criteria:**
- Checks if answer addresses the question
- Assesses usefulness of information provided
- Returns score from 0.0 (irrelevant) to 1.0 (highly relevant)

**Prompt Design:**
- Clear evaluation instructions
- Structured query and answer presentation
- Numeric score output for consistency
- Temperature set to 0.0 for reproducibility

#### Updated `RAFTEvaluator.__init__`
- Added `openai_api_key` parameter
- Passes API key to `GenerationMetrics` initialization

### 2. Updated `raft_pipeline.py`

#### Enhanced `step6_evaluate`
- Added `openai_api_key` parameter
- Passes API key to `RAFTEvaluator` initialization

#### Updated `run_full_pipeline`
- Passes `openai_api_key` to `step6_evaluate` call

### 3. Created Test Script

Created `test_llm_judge.py` to verify the implementation:
- Tests LLM judge initialization
- Tests faithfulness evaluation
- Tests relevance evaluation
- Tests integration with RAFTConfig
- Provides clear pass/fail results

## Usage

### Basic Usage

```python
from raft_evaluator import GenerationMetrics
import os

# Initialize with LLM judge
metrics = GenerationMetrics(
    llm_judge_model="gpt-4-turbo-preview",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Evaluate faithfulness
contexts = ["The hero saved the village from the dragon."]
answer = "The hero defeated the dragon."
faithfulness_score = metrics.compute_faithfulness(answer, contexts)

# Evaluate relevance
query = "Who saved the village?"
relevance_score = metrics.compute_answer_relevance(answer, query)
```

### With RAFTEvaluator

```python
from raft_evaluator import RAFTEvaluator
from raft_config import RAFTConfig
import os

config = RAFTConfig()

evaluator = RAFTEvaluator(
    config=config,
    retrieval_system=retrieval_system,
    model=model,
    tokenizer=tokenizer,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Evaluate dataset
metrics = evaluator.evaluate_dataset(examples)
```

### With Pipeline

```bash
# Run evaluation with LLM judge
python raft_pipeline.py \
  --step evaluate \
  --openai-api-key $OPENAI_API_KEY
```

## Configuration

The LLM judge is configured through `RAFTConfig`:

```python
from raft_config import RAFTConfig, EvaluationConfig

config = RAFTConfig(
    evaluation=EvaluationConfig(
        compute_faithfulness=True,      # Enable faithfulness evaluation
        compute_answer_relevance=True,  # Enable relevance evaluation
        ragas_llm="gpt-4-turbo-preview" # LLM model for judge
    )
)
```

## Fallback Behavior

The implementation includes robust fallback mechanisms:

1. **No OpenAI Package**: Falls back to heuristic evaluation
2. **No API Key**: Falls back to heuristic evaluation
3. **API Call Fails**: Falls back to heuristic evaluation for that specific call
4. **Invalid Response**: Falls back to heuristic evaluation

All fallbacks are logged with warnings for debugging.

## Benefits

### Before (Heuristic Only)
- Simple token overlap for faithfulness
- Query term coverage for relevance
- Fast but less accurate
- Always showed warning: "LLM judge not implemented, using heuristic"

### After (LLM Judge)
- GPT-4 powered semantic evaluation
- Understands context and meaning
- More accurate quality assessment
- No warnings when API key is provided
- Graceful fallback to heuristics when needed

## Testing

Run the test script to verify the implementation:

```bash
# Without API key (tests heuristic fallback)
python test_llm_judge.py

# With API key (tests actual LLM judge)
export OPENAI_API_KEY='your-key-here'
python test_llm_judge.py
```

Expected output:
```
============================================================
TEST SUMMARY
============================================================
Initialization       ✓ PASS
Faithfulness         ✓ PASS
Relevance            ✓ PASS
Config Integration   ✓ PASS

============================================================
✓ ALL TESTS PASSED
============================================================
```

## Cost Considerations

LLM judge evaluation uses OpenAI API calls:
- **Model**: gpt-4-turbo-preview (configurable)
- **Tokens per evaluation**: ~100-200 tokens
- **Cost**: ~$0.01-0.03 per 1000 evaluations (as of 2025)

For large-scale evaluation, consider:
1. Using a smaller model (e.g., gpt-3.5-turbo)
2. Sampling a subset of examples
3. Caching results for repeated evaluations
4. Using heuristic evaluation for development

## Troubleshooting

### Warning: "LLM judge not available, using heuristic"

**Causes:**
1. OpenAI package not installed: `pip install openai`
2. API key not set: `export OPENAI_API_KEY='your-key'`
3. Invalid API key

**Solution:**
```bash
# Install OpenAI package
pip install openai

# Set API key
export OPENAI_API_KEY='your-openai-api-key'

# Verify
python -c "import openai; print('OpenAI installed:', openai.__version__)"
```

### Error: "LLM judge failed: ..."

**Causes:**
1. Network issues
2. API rate limits
3. Invalid model name
4. Insufficient API credits

**Solution:**
- Check network connectivity
- Verify API key has credits
- Check model name in config
- The system will automatically fall back to heuristics

## Future Enhancements

Potential improvements:
1. Support for other LLM providers (Anthropic, Cohere)
2. Batch evaluation for efficiency
3. Caching of LLM judge results
4. Custom evaluation prompts
5. Multi-aspect evaluation (coherence, completeness, etc.)
6. Confidence scores from LLM judge

## References

- OpenAI API Documentation: https://platform.openai.com/docs
- RAGAS Framework: https://docs.ragas.io/
- RAFT Paper: https://arxiv.org/abs/2403.10131


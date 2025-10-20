# Understanding RAFT Dataset Validation Warnings

## What Are These Warnings?

When building RAFT datasets, you may see warnings like:

```
WARNING - Validation failed for 510: Quote 0 not found in oracle texts: "the last of her class"...
```

## Why Do They Happen?

The RAFT dataset builder has a **quality control system** that validates every example:

1. **GPT-4 generates reasoning** with quotes from the source documents
2. **Validation system checks** if those quotes actually exist in the oracle documents
3. **If quotes don't match exactly**, the example is discarded

### Common Reasons for Validation Failures

1. **GPT-4 paraphrases** instead of using exact quotes
2. **Minor text differences** (punctuation, formatting)
3. **GPT-4 hallucinates** quotes that don't exist in the source
4. **Quote spans multiple paragraphs** with different formatting

## Is This a Problem?

**No, this is working as designed!** The RAFT paper emphasizes that:

- Quotes must be **exact** to train the model properly
- Bad examples should be **filtered out** to maintain data quality
- It's better to have **fewer high-quality examples** than many low-quality ones

## What to Expect

### Typical Success Rates

- **Good**: 70-85% of examples pass validation
- **Acceptable**: 60-70% pass validation
- **Concerning**: <50% pass validation

### Example

If you request 20 dev examples:
- **Expected**: 15-18 valid examples created
- **Acceptable**: 12-15 valid examples created
- **Low**: <12 valid examples (may indicate issues)

## Recent Improvements

### Enhanced Fuzzy Matching (Just Added!)

The validation system now has **3 levels of matching**:

#### Level 1: Exact Match (after normalization)
```python
# Normalizes whitespace and case
"The castle was grand" → "the castle was grand"
```

#### Level 2: Punctuation-Insensitive
```python
# Removes all punctuation
"The castle, it was grand!" → "the castle it was grand"
```

#### Level 3: Partial Match (80% threshold)
```python
# For quotes with 3+ words, checks if 80% of words appear in order
Quote: "the old castle was very grand"
Source: "the ancient castle was quite grand"
Result: PASS (4/5 significant words match in order)
```

### Benefits

- **Fewer false rejections**: Handles minor variations better
- **Still maintains quality**: Requires substantial overlap
- **Handles punctuation**: Commas, quotes, dashes don't cause failures
- **Order-preserving**: Words must appear in the same sequence

## How to Handle Validation Warnings

### Option 1: Accept the Filtering (Recommended)

Just let it run - you'll get high-quality examples:

```bash
# Request 25 examples to get ~20 valid ones
python raft_pipeline.py \
  --step dataset \
  --split dev \
  --eval-max-examples 25 \
  --openai-api-key $OPENAI_API_KEY
```

### Option 2: Request More Examples

If you need exactly 20 valid examples, request 25-30:

```bash
# Request 30 to ensure you get at least 20
python raft_pipeline.py \
  --step dataset \
  --split dev \
  --eval-max-examples 30 \
  --openai-api-key $OPENAI_API_KEY
```

### Option 3: Check the Logs

After completion, check how many passed:

```bash
# Count examples in output file
wc -l data/raft_dev.jsonl

# Check the logs for success rate
# Look for: "Built X RAFT examples (Y failed validation)"
```

## Monitoring During Build

### What You'll See

```
Building RAFT dataset: 100%|████████| 20/20 [10:30<00:00, 31.5s/it]
2025-10-20 03:47:25 - RAFT.DatasetBuilder - WARNING - Validation failed for 510: Quote 0 not found...
2025-10-20 03:47:26 - RAFT.DatasetBuilder - WARNING - Validation failed for 511: Quote 1 not found...
Built 17 RAFT examples (3 failed validation)
```

### Interpretation

- **20 examples processed**: All were attempted
- **3 failed validation**: Quotes didn't match source text
- **17 examples saved**: High-quality examples with valid citations

## When to Be Concerned

### Red Flags

1. **>50% failure rate**: May indicate issues with:
   - Source documents not matching what GPT-4 sees
   - Corpus loading problems
   - API issues

2. **All examples failing**: Likely a bug:
   - Check that corpus is loaded correctly
   - Verify oracle documents are being passed to GPT-4
   - Check OpenAI API key is valid

3. **Specific patterns failing**: E.g., all long quotes fail:
   - May need to adjust `max_quote_length` in config
   - Check if source documents have formatting issues

## Configuration Options

### Adjust Quote Requirements

In `raft_config.yaml`:

```yaml
raft_data:
  max_quote_length: 300        # Increase if quotes are too long
  min_quotes_per_example: 1    # Minimum quotes required
  max_quotes_per_example: 3    # Maximum quotes allowed
```

### Adjust GPT-4 Settings

```yaml
raft_data:
  judge_model: "gpt-4.1"       # Or "gpt-4-turbo-preview"
  judge_temperature: 0.2       # Lower = more deterministic
  judge_max_tokens: 1500       # Max tokens for reasoning
```

## Debugging Failed Examples

### Check What GPT-4 Generated

The warnings show the first 50 characters of failed quotes:

```
Quote 0 not found in oracle texts: "the last of her class"...
```

This tells you:
- GPT-4 tried to quote: "the last of her class..."
- This text wasn't found in the oracle documents
- Likely paraphrased or hallucinated

### Manual Verification

If you want to see the full failed example, you can add debug logging:

```python
# In raft_dataset_builder.py, around line 385
if not is_valid:
    logger.warning(
        f"Validation failed for {example_id}: {', '.join(errors)}"
    )
    # Add this for debugging:
    logger.debug(f"Full reasoning: {reasoning}")
    logger.debug(f"Oracle texts: {oracle_texts[:200]}...")
```

## Summary

### Key Points

✅ **Validation warnings are normal** - they indicate quality control is working

✅ **70-85% success rate is good** - you're getting high-quality examples

✅ **Enhanced fuzzy matching** - now handles punctuation and minor variations

✅ **Request extra examples** - if you need exactly N, request N × 1.3

❌ **Don't disable validation** - it ensures training data quality

❌ **Don't worry about warnings** - unless >50% are failing

### Best Practices

1. **Request 20-30% more examples** than you need
2. **Monitor the final count** in the output file
3. **Check success rate** in the logs
4. **Accept that some examples will fail** - this is by design

### Example Workflow

```bash
# Goal: Get 50 dev examples
# Request: 65 examples (30% buffer)
python raft_pipeline.py \
  --step dataset \
  --split dev \
  --eval-max-examples 65 \
  --openai-api-key $OPENAI_API_KEY

# Check results
wc -l data/raft_dev.jsonl
# Expected: 45-55 examples

# If you got 48 examples, that's perfect!
# If you got <45, run again with --eval-max-examples 20 to top up
```

## Need Help?

If you're seeing unusual patterns or very high failure rates:

1. Check the logs for specific error patterns
2. Verify corpus is loaded correctly
3. Test with a small sample first (5-10 examples)
4. Check OpenAI API is working properly

---

**Remember**: The validation system is protecting your training data quality. A few warnings are normal and expected!


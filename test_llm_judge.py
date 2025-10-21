"""
Test script for LLM judge implementation
"""

import logging
import os
from raft_config import RAFTConfig
from raft_evaluator import GenerationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llm_judge_initialization():
    """Test that LLM judge initializes correctly"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: LLM Judge Initialization")
    logger.info("="*60)
    
    # Test without API key (should fall back to heuristic)
    logger.info("\n1. Testing without API key (should use heuristic)...")
    metrics_no_key = GenerationMetrics(llm_judge_model="gpt-4-turbo-preview")
    
    # Test with API key from environment
    logger.info("\n2. Testing with API key from environment...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        metrics_with_key = GenerationMetrics(
            llm_judge_model="gpt-4-turbo-preview",
            api_key=api_key
        )
        logger.info("✓ LLM judge initialized with API key")
    else:
        logger.warning("⚠ OPENAI_API_KEY not set in environment")
    
    # Test without LLM judge (should use heuristic)
    logger.info("\n3. Testing without LLM judge model...")
    metrics_heuristic = GenerationMetrics(llm_judge_model=None)
    logger.info("✓ Heuristic-only mode initialized")
    
    return True


def test_faithfulness_evaluation():
    """Test faithfulness evaluation"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Faithfulness Evaluation")
    logger.info("="*60)
    
    # Test data
    contexts = [
        "The hero saved the village from the dragon.",
        "The dragon was defeated in an epic battle."
    ]
    
    # Good answer (faithful)
    good_answer = "The hero defeated the dragon and saved the village."
    
    # Bad answer (hallucination)
    bad_answer = "The hero rode a unicorn to the moon and found treasure."
    
    # Initialize metrics
    api_key = os.getenv("OPENAI_API_KEY")
    metrics = GenerationMetrics(
        llm_judge_model="gpt-4-turbo-preview" if api_key else None,
        api_key=api_key
    )
    
    # Test good answer
    logger.info("\n1. Testing faithful answer...")
    logger.info(f"   Answer: {good_answer}")
    score_good = metrics.compute_faithfulness(good_answer, contexts)
    logger.info(f"   Faithfulness score: {score_good:.3f}")
    
    # Test bad answer
    logger.info("\n2. Testing unfaithful answer...")
    logger.info(f"   Answer: {bad_answer}")
    score_bad = metrics.compute_faithfulness(bad_answer, contexts)
    logger.info(f"   Faithfulness score: {score_bad:.3f}")
    
    # Good answer should score higher
    if score_good > score_bad:
        logger.info("✓ Faithfulness evaluation working correctly")
        return True
    else:
        logger.warning("⚠ Unexpected scores (good should be > bad)")
        return False


def test_relevance_evaluation():
    """Test answer relevance evaluation"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Answer Relevance Evaluation")
    logger.info("="*60)
    
    # Test data
    query = "Who saved the village?"
    
    # Good answer (relevant)
    good_answer = "The hero saved the village from the dragon."
    
    # Bad answer (irrelevant)
    bad_answer = "The weather was nice and sunny that day."
    
    # Initialize metrics
    api_key = os.getenv("OPENAI_API_KEY")
    metrics = GenerationMetrics(
        llm_judge_model="gpt-4-turbo-preview" if api_key else None,
        api_key=api_key
    )
    
    # Test good answer
    logger.info("\n1. Testing relevant answer...")
    logger.info(f"   Query: {query}")
    logger.info(f"   Answer: {good_answer}")
    score_good = metrics.compute_answer_relevance(good_answer, query)
    logger.info(f"   Relevance score: {score_good:.3f}")
    
    # Test bad answer
    logger.info("\n2. Testing irrelevant answer...")
    logger.info(f"   Query: {query}")
    logger.info(f"   Answer: {bad_answer}")
    score_bad = metrics.compute_answer_relevance(bad_answer, query)
    logger.info(f"   Relevance score: {score_bad:.3f}")
    
    # Good answer should score higher
    if score_good > score_bad:
        logger.info("✓ Relevance evaluation working correctly")
        return True
    else:
        logger.warning("⚠ Unexpected scores (good should be > bad)")
        return False


def test_with_config():
    """Test LLM judge with RAFTConfig"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Integration with RAFTConfig")
    logger.info("="*60)
    
    # Create config
    config = RAFTConfig()
    
    # Check if LLM judge is configured
    logger.info(f"\nRAGAS LLM model: {config.evaluation.ragas_llm}")
    logger.info(f"Compute faithfulness: {config.evaluation.compute_faithfulness}")
    logger.info(f"Compute answer relevance: {config.evaluation.compute_answer_relevance}")
    
    # Initialize metrics from config
    api_key = os.getenv("OPENAI_API_KEY")
    metrics = GenerationMetrics(
        llm_judge_model=config.evaluation.ragas_llm if config.evaluation.compute_faithfulness else None,
        api_key=api_key
    )
    
    if metrics.llm_judge_model:
        logger.info(f"✓ LLM judge configured: {metrics.llm_judge_model}")
    else:
        logger.info("✓ Using heuristic evaluation (no API key)")
    
    return True


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("LLM JUDGE IMPLEMENTATION TEST")
    logger.info("="*60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("\n⚠ WARNING: OPENAI_API_KEY not set in environment")
        logger.warning("Tests will run with heuristic fallback only")
        logger.warning("To test LLM judge, set: export OPENAI_API_KEY='your-key'\n")
    else:
        logger.info("\n✓ OPENAI_API_KEY found in environment\n")
    
    results = []
    
    # Run tests
    try:
        results.append(("Initialization", test_llm_judge_initialization()))
    except Exception as e:
        logger.error(f"Initialization test failed: {e}")
        results.append(("Initialization", False))
    
    try:
        results.append(("Faithfulness", test_faithfulness_evaluation()))
    except Exception as e:
        logger.error(f"Faithfulness test failed: {e}")
        results.append(("Faithfulness", False))
    
    try:
        results.append(("Relevance", test_relevance_evaluation()))
    except Exception as e:
        logger.error(f"Relevance test failed: {e}")
        results.append(("Relevance", False))
    
    try:
        results.append(("Config Integration", test_with_config()))
    except Exception as e:
        logger.error(f"Config integration test failed: {e}")
        results.append(("Config Integration", False))
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:<20} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        if not api_key:
            logger.info("\nNote: Tests ran with heuristic fallback")
            logger.info("Set OPENAI_API_KEY to test actual LLM judge")
    else:
        logger.info("✗ SOME TESTS FAILED")
    logger.info("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


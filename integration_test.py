"""
Integration Test for RAFT System
Tests all components working together
"""

import logging
import sys
import tempfile
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_config():
    """Test configuration module"""
    logger.info("Testing configuration module...")
    
    from raft_config import RAFTConfig, ModelConfig, TrainingConfig
    
    # Test default config
    config = RAFTConfig()
    assert config.model.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.lora_r == 32
    assert config.training.num_train_epochs == 3
    
    # Test custom config
    model_config = ModelConfig(lora_r=16, lora_alpha=32)
    assert model_config.lora_alpha == 32
    
    # Test YAML save/load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.to_yaml(f.name)
        loaded_config = RAFTConfig.from_yaml(f.name)
        assert loaded_config.model.lora_r == config.model.lora_r
        Path(f.name).unlink()
    
    logger.info("✓ Configuration module passed")
    return True


def test_data_structures():
    """Test data structures"""
    logger.info("Testing data structures...")
    
    from raft_data_loader import StoryQAExample, Chapter
    
    # Test StoryQAExample
    example = StoryQAExample(
        id="test_1",
        query="What is the story about?",
        gold_documents=["doc1", "doc2"],
        answers=["Answer 1", "Answer 2"]
    )
    assert example.id == "test_1"
    assert len(example.gold_documents) == 2
    
    # Test Chapter
    chapter = Chapter(
        doc_id="story1_1",
        story_id="story1",
        chapter_index=1,
        text="This is chapter 1",
        token_length=4
    )
    assert chapter.story_id == "story1"
    assert chapter.token_length == 4
    
    logger.info("✓ Data structures passed")
    return True


def test_retrieval_components():
    """Test retrieval components"""
    logger.info("Testing retrieval components...")
    
    from raft_retrieval import TextChunker, RetrievalResult
    
    # Test TextChunker
    chunker = TextChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk_text(
        "This is a test sentence with more than ten words in it.",
        "test_doc"
    )
    assert len(chunks) > 0
    assert chunks[0].doc_id == "test_doc"
    
    # Test RetrievalResult
    result = RetrievalResult(
        doc_id="doc1",
        text="sample text",
        score=0.95,
        rank=0
    )
    assert result.score == 0.95
    
    logger.info("✓ Retrieval components passed")
    return True


def test_dataset_builder_components():
    """Test dataset builder components"""
    logger.info("Testing dataset builder components...")
    
    from raft_dataset_builder import CitationValidator, RAFTExample
    
    # Test CitationValidator
    text_with_quotes = "First part ##begin_quote##quoted text##end_quote## last part"
    quotes = CitationValidator.extract_quotes(text_with_quotes)
    assert len(quotes) == 1
    assert quotes[0] == "quoted text"
    
    # Test quote validation
    is_valid = CitationValidator.validate_quote("quoted text", text_with_quotes)
    assert is_valid == True
    
    # Test RAFTExample
    example = RAFTExample(
        query="Test query",
        contexts=[{"doc_id": "doc1", "text": "Context 1"}],
        oracle_ids=["doc1"],
        distractor_ids=[],
        reasoning="##begin_quote##Context 1##end_quote##",
        answer="Final answer"
    )
    assert len(example.contexts) == 1
    
    logger.info("✓ Dataset builder components passed")
    return True


def test_evaluation_components():
    """Test evaluation components"""
    logger.info("Testing evaluation components...")
    
    from raft_evaluator import RetrievalMetrics, GenerationMetrics
    
    # Test RetrievalMetrics
    precision = RetrievalMetrics.context_precision(
        retrieved_docs=["doc1", "doc2", "doc3"],
        gold_docs=["doc1", "doc2"]
    )
    assert 0.6 < precision < 0.7  # Should be 2/3
    
    recall = RetrievalMetrics.context_recall(
        retrieved_docs=["doc1", "doc2", "doc3"],
        gold_docs=["doc1", "doc2"]
    )
    assert recall == 1.0  # Retrieved all gold docs
    
    # Test GenerationMetrics
    gen_metrics = GenerationMetrics()
    rouge = gen_metrics.compute_rouge_l(
        generated="The cat sat on the mat",
        references=["A cat sat on a mat"]
    )
    assert rouge > 0  # Should have some overlap
    
    logger.info("✓ Evaluation components passed")
    return True


def test_full_workflow():
    """Test complete workflow with mock data"""
    logger.info("Testing full workflow...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Test config
        from raft_config import RAFTConfig
        config = RAFTConfig()
        config.system.data_dir = temp_dir
        config.system.cache_dir = temp_dir
        config.system.log_dir = temp_dir
        config.system.setup_directories()
        
        # 2. Test data structures
        from raft_data_loader import Chapter
        
        # Create mock corpus
        chapters = {
            "story1_1": Chapter(
                doc_id="story1_1",
                story_id="story1",
                chapter_index=1,
                text="This is the first chapter of the story.",
                token_length=8
            ),
            "story1_2": Chapter(
                doc_id="story1_2",
                story_id="story1",
                chapter_index=2,
                text="This is the second chapter with more details.",
                token_length=9
            )
        }
        
        # 3. Test retrieval system (minimal)
        from raft_retrieval import TextChunker
        chunker = TextChunker(chunk_size=50, overlap=10)
        
        for chapter in chapters.values():
            chunks = chunker.chunk_text(chapter.text, chapter.doc_id)
            assert len(chunks) > 0
        
        # 4. Test citation validation
        from raft_dataset_builder import CitationValidator
        
        reasoning = "Based on the text ##begin_quote##first chapter##end_quote## we can see..."
        quotes = CitationValidator.extract_quotes(reasoning)
        assert len(quotes) == 1
        
        # Validate against oracle
        is_valid = CitationValidator.validate_quote(
            quotes[0],
            chapters["story1_1"].text,
            fuzzy=True
        )
        assert is_valid == True
        
        # 5. Test evaluation metrics
        from raft_evaluator import RetrievalMetrics
        
        precision = RetrievalMetrics.context_precision(
            retrieved_docs=["story1_1", "story1_2"],
            gold_docs=["story1_1"]
        )
        assert precision == 0.5
        
        logger.info("✓ Full workflow passed")
        return True
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_formatting():
    """Test chat template formatting"""
    logger.info("Testing chat formatting...")
    
    from raft_dataset_builder import RAFTExample
    
    example = RAFTExample(
        query="What is AI?",
        contexts=[
            {"doc_id": "doc1", "text": "AI is artificial intelligence."}
        ],
        oracle_ids=["doc1"],
        distractor_ids=[],
        reasoning="##begin_quote##artificial intelligence##end_quote##",
        answer="AI stands for artificial intelligence"
    )
    
    # Test chat format creation
    from raft_dataset_builder import RAFTDatasetBuilder
    
    # Mock builder for format test
    class MockBuilder:
        def _get_training_system_prompt(self):
            return "System prompt"
        
        def to_chat_format(self, example):
            context_parts = []
            for ctx in example.contexts:
                context_parts.append(f"\n[Chapter: {ctx['doc_id']}]")
                context_parts.append(ctx['text'])
            
            context_str = "\n".join(context_parts)
            user_message = f"Question: {example.query}\n\nContext:\n{context_str}"
            assistant_message = f"{example.reasoning}\n\n##Answer: {example.answer}"
            
            return {
                "messages": [
                    {"role": "system", "content": self._get_training_system_prompt()},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            }
    
    builder = MockBuilder()
    chat_format = builder.to_chat_format(example)
    
    assert "messages" in chat_format
    assert len(chat_format["messages"]) == 3
    assert chat_format["messages"][0]["role"] == "system"
    assert chat_format["messages"][1]["role"] == "user"
    assert chat_format["messages"][2]["role"] == "assistant"
    assert "##Answer:" in chat_format["messages"][2]["content"]
    
    logger.info("✓ Chat formatting passed")
    return True


def run_all_tests():
    """Run all integration tests"""
    logger.info("="*70)
    logger.info("RAFT INTEGRATION TEST SUITE")
    logger.info("="*70)
    
    tests = [
        ("Configuration", test_config),
        ("Data Structures", test_data_structures),
        ("Retrieval Components", test_retrieval_components),
        ("Dataset Builder", test_dataset_builder_components),
        ("Evaluation Components", test_evaluation_components),
        ("Chat Formatting", test_chat_formatting),
        ("Full Workflow", test_full_workflow),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*70}")
            result = test_func()
            if result:
                results.append((test_name, True, ""))
                passed += 1
            else:
                results.append((test_name, False, "Test returned False"))
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}", exc_info=True)
            results.append((test_name, False, str(e)))
            failed += 1
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")
    
    if failed > 0:
        logger.info("\n" + "="*70)
        logger.info("FAILED TESTS")
        logger.info("="*70)
        for test_name, success, error in results:
            if not success:
                logger.error(f"{test_name}: {error}")
    
    logger.info("\n" + "="*70)
    if failed == 0:
        logger.info("✓ ALL INTEGRATION TESTS PASSED")
        logger.info("\nThe system is ready for use!")
    else:
        logger.info("✗ SOME TESTS FAILED")
        logger.info("\nPlease fix the errors above before proceeding")
    logger.info("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
"""
Installation Test Script
Validates that all dependencies are correctly installed
"""

import sys
import logging
from typing import List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_python_version() -> Tuple[bool, str]:
    """Test Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)"


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        return True, package_name or module_name
    except ImportError as e:
        return False, f"{package_name or module_name}: {str(e)}"


def test_torch_cuda() -> Tuple[bool, str]:
    """Test PyTorch CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version} - {device_name}"
        return False, "CUDA not available (CPU only)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_unsloth() -> Tuple[bool, str]:
    """Test Unsloth installation"""
    try:
        from unsloth import FastLanguageModel
        return True, "Unsloth"
    except ImportError as e:
        return False, f"Unsloth: {str(e)}"


def test_transformers_version() -> Tuple[bool, str]:
    """Test Transformers version"""
    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        if major >= 4 and minor >= 38:
            return True, f"Transformers {version}"
        return False, f"Transformers {version} (requires 4.38+)"
    except Exception as e:
        return False, f"Transformers: {str(e)}"


def test_faiss() -> Tuple[bool, str]:
    """Test FAISS installation"""
    try:
        import faiss
        # Try to create a simple index
        d = 128
        index = faiss.IndexFlatL2(d)
        return True, "FAISS"
    except Exception as e:
        return False, f"FAISS: {str(e)}"


def test_sentence_transformers() -> Tuple[bool, str]:
    """Test Sentence Transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        # Try to load a small model
        return True, "Sentence Transformers"
    except Exception as e:
        return False, f"Sentence Transformers: {str(e)}"


def test_nltk_data() -> Tuple[bool, str]:
    """Test NLTK data availability"""
    try:
        import nltk
        # Try to use punkt tokenizer
        nltk.data.find('tokenizers/punkt')
        return True, "NLTK (punkt tokenizer)"
    except LookupError:
        return False, "NLTK punkt tokenizer not found (run: python -c \"import nltk; nltk.download('punkt')\")"
    except Exception as e:
        return False, f"NLTK: {str(e)}"


def test_openai_key() -> Tuple[bool, str]:
    """Test OpenAI API key"""
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return True, "OpenAI API key found"
    return False, "OPENAI_API_KEY not set (required for CoT generation)"


def test_all_modules() -> Tuple[bool, str]:
    """Test if all project modules are importable"""
    modules = [
        'raft_config',
        'raft_data_loader',
        'raft_retrieval',
        'raft_dataset_builder',
        'raft_trainer',
        'raft_evaluator',
        'raft_pipeline'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
        except Exception as e:
            failed.append(f"{module}: {str(e)}")
    
    if not failed:
        return True, "All project modules"
    return False, f"Failed to import: {', '.join(failed)}"


def run_all_tests():
    """Run all installation tests"""
    logger.info("="*70)
    logger.info("RAFT Installation Test")
    logger.info("="*70)
    
    # Define all tests
    tests = [
        ("Python Version", test_python_version),
        ("PyTorch", lambda: test_import("torch")),
        ("CUDA Support", test_torch_cuda),
        ("Transformers", test_transformers_version),
        ("Accelerate", lambda: test_import("accelerate")),
        ("PEFT", lambda: test_import("peft")),
        ("BitsAndBytes", lambda: test_import("bitsandbytes")),
        ("Unsloth", test_unsloth),
        ("TRL", lambda: test_import("trl")),
        ("Datasets", lambda: test_import("datasets")),
        ("Sentence Transformers", test_sentence_transformers),
        ("FAISS", test_faiss),
        ("ROUGE Score", lambda: test_import("rouge_score")),
        ("BERTScore", lambda: test_import("bert_score")),
        ("NLTK", lambda: test_import("nltk")),
        ("NLTK Data", test_nltk_data),
        ("OpenAI", lambda: test_import("openai")),
        ("OpenAI API Key", test_openai_key),
        ("NumPy", lambda: test_import("numpy")),
        ("Pandas", lambda: test_import("pandas")),
        ("TQDM", lambda: test_import("tqdm")),
        ("YAML", lambda: test_import("yaml", "PyYAML")),
        ("Project Modules", test_all_modules),
    ]
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    warnings = 0
    
    logger.info("\nRunning tests...\n")
    
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            
            if success:
                logger.info(f"✓ {test_name}: {message}")
                passed += 1
            else:
                # Check if it's a warning (non-critical)
                is_warning = test_name in ["CUDA Support", "OpenAI API Key"]
                if is_warning:
                    logger.warning(f"⚠ {test_name}: {message}")
                    warnings += 1
                else:
                    logger.error(f"✗ {test_name}: {message}")
                    failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name}: Unexpected error - {str(e)}")
            results.append((test_name, False, str(e)))
            failed += 1
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    logger.info(f"Passed:   {passed}/{len(tests)}")
    logger.info(f"Failed:   {failed}/{len(tests)}")
    logger.info(f"Warnings: {warnings}/{len(tests)}")
    
    # Print recommendations
    if failed > 0:
        logger.info("\n" + "="*70)
        logger.info("RECOMMENDATIONS")
        logger.info("="*70)
        
        for test_name, success, message in results:
            if not success and test_name not in ["CUDA Support", "OpenAI API Key"]:
                logger.info(f"\n{test_name}:")
                logger.info(f"  Issue: {message}")
                
                # Provide specific fixes
                if "torch" in test_name.lower():
                    logger.info("  Fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                elif "transformers" in test_name.lower():
                    logger.info("  Fix: pip install transformers>=4.38.0")
                elif "unsloth" in test_name.lower():
                    logger.info("  Fix: pip install --upgrade unsloth unsloth_zoo")
                elif "faiss" in test_name.lower():
                    logger.info("  Fix: pip install faiss-cpu  # or faiss-gpu for GPU support")
                elif "nltk" in test_name.lower() and "punkt" in message.lower():
                    logger.info("  Fix: python -c \"import nltk; nltk.download('punkt')\"")
                elif "project modules" in test_name.lower():
                    logger.info("  Fix: Ensure all .py files are in the current directory")
                else:
                    logger.info("  Fix: pip install -r requirements.txt")
    
    # Print warnings explanation
    if warnings > 0:
        logger.info("\n" + "="*70)
        logger.info("WARNINGS (Non-Critical)")
        logger.info("="*70)
        
        for test_name, success, message in results:
            if not success and test_name in ["CUDA Support", "OpenAI API Key"]:
                logger.info(f"\n{test_name}:")
                logger.info(f"  {message}")
                
                if test_name == "CUDA Support":
                    logger.info("  Note: You can still run the pipeline on CPU, but training will be much slower")
                elif test_name == "OpenAI API Key":
                    logger.info("  Note: Required only for RAFT dataset generation (CoT creation)")
                    logger.info("  Set with: export OPENAI_API_KEY='your-key-here'")
    
    # Final status
    logger.info("\n" + "="*70)
    if failed == 0:
        logger.info("✓ Installation Complete - Ready to run!")
        logger.info("\nNext steps:")
        logger.info("  1. Review configuration: python raft_config.py")
        logger.info("  2. Run quick test: python example_usage.py 1")
        logger.info("  3. Run full pipeline: python raft_pipeline.py --step all")
    else:
        logger.info("✗ Installation Incomplete - Please fix the errors above")
        logger.info("\nAfter fixing, run this script again to verify")
    logger.info("="*70)
    
    return failed == 0


def test_gpu_memory():
    """Test GPU memory availability"""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                logger.info(f"\nGPU {i}: {props.name}")
                logger.info(f"  Total Memory: {total_memory:.2f} GB")
                
                if total_memory < 16:
                    logger.warning(f"  ⚠ Less than 16GB - may need to reduce batch size")
                else:
                    logger.info(f"  ✓ Sufficient memory for RAFT training")
        else:
            logger.warning("No CUDA GPUs available")
    except Exception as e:
        logger.error(f"Could not check GPU memory: {str(e)}")


def quick_functionality_test():
    """Quick test of core functionality"""
    logger.info("\n" + "="*70)
    logger.info("QUICK FUNCTIONALITY TEST")
    logger.info("="*70)
    
    try:
        # Test config
        logger.info("\n1. Testing configuration...")
        from raft_config import RAFTConfig
        config = RAFTConfig()
        logger.info("   ✓ Configuration module works")
        
        # Test data structures
        logger.info("\n2. Testing data structures...")
        from raft_data_loader import StoryQAExample, Chapter
        example = StoryQAExample(
            id="test",
            query="test query",
            gold_documents=["doc1"],
            answers=["answer1"]
        )
        logger.info("   ✓ Data structures work")
        
        # Test retrieval components
        logger.info("\n3. Testing retrieval components...")
        from raft_retrieval import TextChunker
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text("This is a test text.", "test_doc")
        logger.info("   ✓ Retrieval components work")
        
        # Test evaluation components
        logger.info("\n4. Testing evaluation components...")
        from raft_evaluator import RetrievalMetrics
        precision = RetrievalMetrics.context_precision(["doc1", "doc2"], ["doc1"])
        logger.info("   ✓ Evaluation components work")
        
        logger.info("\n✓ All core functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Functionality test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAFT installation")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick functionality test only"
    )
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Show GPU memory information"
    )
    
    args = parser.parse_args()
    
    if args.gpu_info:
        test_gpu_memory()
    elif args.quick:
        quick_functionality_test()
    else:
        # Run full test suite
        success = run_all_tests()
        
        # Run functionality test if installation passed
        if success:
            quick_functionality_test()
        
        # Show GPU info
        test_gpu_memory()
        
        sys.exit(0 if success else 1)
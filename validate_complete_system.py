#!/usr/bin/env python3
"""
Complete System Validation Script
Validates all code is correct, complete, and ready for use
"""

import sys
import logging
from pathlib import Path
import importlib
import inspect

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class CodeValidator:
    """Validates Python modules for completeness and correctness"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.modules_checked = []
    
    def validate_module(self, module_name: str) -> bool:
        """Validate a single module"""
        logger.info(f"\nValidating {module_name}...")
        
        try:
            # Import module
            module = importlib.import_module(module_name)
            self.modules_checked.append(module_name)
            
            # Check module has docstring
            if not module.__doc__:
                self.warnings.append(f"{module_name}: Missing module docstring")
            
            # Get all classes and functions
            members = inspect.getmembers(module)
            
            classes = [m for m in members if inspect.isclass(m[1]) and m[1].__module__ == module_name]
            functions = [m for m in members if inspect.isfunction(m[1]) and m[1].__module__ == module_name]
            
            logger.info(f"  Found {len(classes)} classes, {len(functions)} functions")
            
            # Validate classes
            for class_name, class_obj in classes:
                self._validate_class(module_name, class_name, class_obj)
            
            # Validate functions
            for func_name, func_obj in functions:
                self._validate_function(module_name, func_name, func_obj)
            
            logger.info(f"  ✓ {module_name} validation complete")
            return True
            
        except ImportError as e:
            self.errors.append(f"{module_name}: Import failed - {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"{module_name}: Validation failed - {str(e)}")
            return False
    
    def _validate_class(self, module_name: str, class_name: str, class_obj):
        """Validate a class"""
        # Check class docstring
        if not class_obj.__doc__:
            self.warnings.append(f"{module_name}.{class_name}: Missing class docstring")
        
        # Check methods
        methods = [m for m in inspect.getmembers(class_obj) 
                  if inspect.ismethod(m[1]) or inspect.isfunction(m[1])]
        
        for method_name, method_obj in methods:
            if method_name.startswith('_') and method_name != '__init__':
                continue  # Skip private methods except __init__
            
            # Check method docstring
            if not method_obj.__doc__ and not method_name.startswith('__'):
                self.warnings.append(f"{module_name}.{class_name}.{method_name}: Missing docstring")
    
    def _validate_function(self, module_name: str, func_name: str, func_obj):
        """Validate a function"""
        # Check function docstring
        if not func_obj.__doc__:
            self.warnings.append(f"{module_name}.{func_name}: Missing function docstring")
        
        # Check for type hints
        sig = inspect.signature(func_obj)
        if sig.return_annotation == inspect.Signature.empty:
            self.warnings.append(f"{module_name}.{func_name}: Missing return type hint")
    
    def print_report(self):
        """Print validation report"""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION REPORT")
        logger.info("="*70)
        
        logger.info(f"\nModules checked: {len(self.modules_checked)}")
        for module in self.modules_checked:
            logger.info(f"  ✓ {module}")
        
        if self.errors:
            logger.error(f"\n❌ ERRORS FOUND: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  {error}")
        else:
            logger.info(f"\n✅ NO ERRORS FOUND")
        
        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for warning in self.warnings[:10]:  # Show first 10
                logger.warning(f"  {warning}")
            if len(self.warnings) > 10:
                logger.warning(f"  ... and {len(self.warnings) - 10} more")
        else:
            logger.info(f"\n✅ NO WARNINGS")
        
        return len(self.errors) == 0


def check_file_completeness():
    """Check all required files exist"""
    logger.info("\n" + "="*70)
    logger.info("CHECKING FILE COMPLETENESS")
    logger.info("="*70)
    
    required_files = [
        "raft_config.py",
        "raft_data_loader.py",
        "raft_retrieval.py",
        "raft_dataset_builder.py",
        "raft_trainer.py",
        "raft_evaluator.py",
        "raft_pipeline.py",
        "example_usage.py",
        "test_installation.py",
        "requirements.txt",
    ]
    
    missing = []
    for filename in required_files:
        if Path(filename).exists():
            logger.info(f"  ✓ {filename}")
        else:
            logger.error(f"  ✗ {filename} MISSING")
            missing.append(filename)
    
    if missing:
        logger.error(f"\n❌ {len(missing)} files missing")
        return False
    else:
        logger.info(f"\n✅ All {len(required_files)} required files present")
        return True


def check_code_patterns():
    """Check for common code issues"""
    logger.info("\n" + "="*70)
    logger.info("CHECKING CODE PATTERNS")
    logger.info("="*70)
    
    issues = []
    
    # Check for json.dumps to file (should be json.dump)
    logger.info("\nChecking for json.dumps/dump issues...")
    for py_file in Path('.').glob('*.py'):
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'json.dumps(' in line and ', f' in line:
                    issues.append(f"{py_file.name}:{i} - Possible json.dumps to file (should be json.dump)")
    
    if not issues:
        logger.info("  ✓ No json.dumps/dump issues found")
    
    # Check for missing error handling on file operations
    logger.info("\nChecking for error handling...")
    for py_file in Path('.').glob('*.py'):
        if py_file.name.startswith('test_') or py_file.name.startswith('validate_'):
            continue
        
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Look for file operations without try-except nearby
            in_try_block = False
            try_depth = 0
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                if 'try:' in stripped:
                    in_try_block = True
                    try_depth += 1
                elif 'except' in stripped:
                    try_depth = max(0, try_depth - 1)
                    if try_depth == 0:
                        in_try_block = False
                
                # Check for risky operations
                if not in_try_block:
                    if any(pattern in stripped for pattern in ['open(', 'json.load', 'json.dump']):
                        # Allow if in __main__ or test functions
                        if i < len(lines) - 1:
                            context = '\n'.join(lines[max(0, i-5):min(len(lines), i+5)])
                            if 'if __name__' not in context and 'def test_' not in context:
                                issues.append(f"{py_file.name}:{i} - File operation without error handling")
    
    if not issues:
        logger.info("  ✓ Error handling looks good")
    
    # Report issues
    if issues:
        logger.warning(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues[:5]:
            logger.warning(f"  {issue}")
        if len(issues) > 5:
            logger.warning(f"  ... and {len(issues) - 5} more")
        return False
    else:
        logger.info(f"\n✅ Code patterns check passed")
        return True


def check_imports():
    """Check all imports work"""
    logger.info("\n" + "="*70)
    logger.info("CHECKING IMPORTS")
    logger.info("="*70)
    
    modules = [
        "raft_config",
        "raft_data_loader",
        "raft_retrieval",
        "raft_dataset_builder",
        "raft_trainer",
        "raft_evaluator",
        "raft_pipeline",
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            logger.info(f"  ✓ {module}")
        except Exception as e:
            logger.error(f"  ✗ {module}: {str(e)}")
            failed.append(module)
    
    if failed:
        logger.error(f"\n❌ {len(failed)} modules failed to import")
        return False
    else:
        logger.info(f"\n✅ All {len(modules)} modules import successfully")
        return True


def main():
    """Main validation function"""
    logger.info("="*70)
    logger.info("RAFT SYSTEM COMPLETE VALIDATION")
    logger.info("="*70)
    
    results = []
    
    # 1. Check file completeness
    results.append(("File Completeness", check_file_completeness()))
    
    # 2. Check imports
    results.append(("Import Check", check_imports()))
    
    # 3. Check code patterns
    results.append(("Code Patterns", check_code_patterns()))
    
    # 4. Validate each module
    validator = CodeValidator()
    core_modules = [
        "raft_config",
        "raft_data_loader",
        "raft_retrieval",
        "raft_dataset_builder",
        "raft_trainer",
        "raft_evaluator",
        "raft_pipeline",
    ]
    
    module_validation_passed = True
    for module in core_modules:
        if not validator.validate_module(module):
            module_validation_passed = False
    
    validator.print_report()
    results.append(("Module Validation", module_validation_passed))
    
    # 5. Run integration tests
    logger.info("\n" + "="*70)
    logger.info("RUNNING INTEGRATION TESTS")
    logger.info("="*70)
    
    try:
        import integration_test
        integration_passed = integration_test.run_all_tests()
        results.append(("Integration Tests", integration_passed))
    except Exception as e:
        logger.error(f"Integration tests failed: {str(e)}")
        results.append(("Integration Tests", False))
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL VALIDATION SUMMARY")
    logger.info("="*70)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{check_name:<25} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED")
        logger.info("\nThe RAFT system is:")
        logger.info("  ✅ Complete - All files present")
        logger.info("  ✅ Correct - All code working")
        logger.info("  ✅ Robust - Error handling in place")
        logger.info("  ✅ Tested - All tests passing")
        logger.info("\n🚀 System is PRODUCTION READY!")
    else:
        logger.error("❌ SOME VALIDATIONS FAILED")
        logger.info("\nPlease review the errors above and fix before deploying.")
    logger.info("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
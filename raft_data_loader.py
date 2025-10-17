"""
RAFT Data Loader Module
Handles loading and preprocessing of MSRS Story-QA dataset
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger("RAFT.DataLoader")


@dataclass
class StoryQAExample:
    """Container for Story-QA example"""
    id: str
    query: str
    gold_documents: List[str]
    answers: List[str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Chapter:
    """Container for story chapter"""
    story_id: str
    chapter_index: int
    text: str
    token_length: int
    metadata: Optional[Dict[str, Any]] = None


class MSRSDataLoader:
    """Loader for MSRS Story-QA dataset"""
    
    def __init__(
        self,
        dataset_name: str = "yale-nlp/MSRS",
        dataset_config: str = "story-qa",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize MSRS data loader
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration (e.g., 'story-qa')
            cache_dir: Directory for caching downloaded data
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.cache_dir = cache_dir
        
        self._dataset = None
        self._corpus = {}
        self._chapter_index = {}
        
        logger.info(f"Initialized MSRS loader for {dataset_name}/{dataset_config}")
    
    def load_dataset(self, split: Optional[str] = None) -> DatasetDict:
        """
        Load Story-QA dataset from HuggingFace
        
        Args:
            split: Specific split to load (train/dev/test) or None for all
            
        Returns:
            DatasetDict with requested splits
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")
            
            if split:
                self._dataset = {split: load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split=split,
                    cache_dir=self.cache_dir
                )}
                logger.info(f"Loaded {split} split: {len(self._dataset[split])} examples")
            else:
                self._dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    cache_dir=self.cache_dir
                )
                logger.info(
                    f"Loaded all splits: "
                    f"{', '.join([f'{k}={len(v)}' for k, v in self._dataset.items()])}"
                )
            
            return self._dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def parse_examples(
        self,
        split: str = "train",
        max_examples: Optional[int] = None
    ) -> List[StoryQAExample]:
        """
        Parse dataset into StoryQAExample objects
        
        Args:
            split: Dataset split to parse
            max_examples: Maximum number of examples to parse
            
        Returns:
            List of StoryQAExample objects
        """
        if self._dataset is None:
            self.load_dataset()
        
        try:
            dataset_split = self._dataset[split]
            examples = []
            
            n_examples = len(dataset_split)
            if max_examples:
                n_examples = min(n_examples, max_examples)
            
            logger.info(f"Parsing {n_examples} examples from {split} split")
            
            for idx in tqdm(range(n_examples), desc=f"Parsing {split}"):
                item = dataset_split[idx]
                
                example = StoryQAExample(
                    id=item.get('id', f"{split}_{idx}"),
                    query=item['query'],
                    gold_documents=item['gold_documents'],
                    answers=item['answer'],
                    metadata={
                        'split': split,
                        'index': idx,
                        'num_gold_docs': len(item['gold_documents']),
                        'num_answers': len(item['answer'])
                    }
                )
                examples.append(example)
            
            logger.info(
                f"Parsed {len(examples)} examples, "
                f"avg {sum(len(e.gold_documents) for e in examples) / len(examples):.2f} gold docs per example"
            )
            
            return examples
            
        except Exception as e:
            logger.error(f"Failed to parse examples: {str(e)}")
            raise
    
    def load_corpus(
        self,
        corpus_path: Optional[str] = None
    ) -> Dict[str, Chapter]:
        """
        Load story chapters corpus
        
        Args:
            corpus_path: Path to corpus file or directory
            
        Returns:
            Dictionary mapping doc_id to Chapter objects
        """
        try:
            if corpus_path:
                logger.info(f"Loading corpus from {corpus_path}")
                self._corpus = self._load_corpus_from_file(corpus_path)
            else:
                logger.info("Loading corpus from HuggingFace dataset")
                self._corpus = self._load_corpus_from_hf()
            
            # Build chapter index
            if self._corpus:
                self._build_chapter_index()
            
            logger.info(
                f"Loaded corpus: {len(self._corpus)} chapters across "
                f"{len(self._chapter_index)} stories"
            )
            
            return self._corpus
            
        except Exception as e:
            logger.error(f"Failed to load corpus: {str(e)}")
            raise
    
    def _load_corpus_from_hf(self) -> Dict[str, Chapter]:
        """Load corpus from HuggingFace dataset"""
        corpus = {}
        
        try:
            # Try loading corpus configuration from MSRS
            logger.info("Attempting to load story-corpus from HuggingFace...")
            corpus_dataset = load_dataset(
                self.dataset_name,
                "story-corpus",
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loaded corpus dataset with {len(corpus_dataset['corpus'])} items")
            
            for item in tqdm(corpus_dataset['corpus'], desc="Loading corpus"):              
                # Parse story_id and chapter_index from doc_id
                story_id = item['id']
                chapter_index = 0
                
                chapter = Chapter(
                    story_id=story_id,
                    chapter_index=chapter_index,
                    text=item['text'],
                    token_length=len(item['text'].split()),
                    metadata=item.get('metadata', {})
                )
                corpus[story_id] = chapter
            
            logger.info(f"Successfully loaded {len(corpus)} chapters from HuggingFace")
            return corpus
            
        except Exception as e:
            logger.warning(f"Could not load from HF story-corpus config: {str(e)}")
            logger.info("The corpus might not be available in the expected format")
            logger.info("Please download it manually from: https://huggingface.co/datasets/yale-nlp/MSRS/viewer/story-corpus")
            return {}
    
    def _load_corpus_from_file(self, corpus_path: str) -> Dict[str, Chapter]:
        """Load corpus from local file"""
        corpus = {}
        path = Path(corpus_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")
        
        if path.is_file():
            # Single JSON/JSONL file
            if path.suffix == '.jsonl':
                with open(path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Loading corpus"):
                        item = json.loads(line)
                        chapter = self._parse_chapter_dict(item)
                        corpus[chapter.story_id] = chapter
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = list(data.values())
                    for item in tqdm(data, desc="Loading corpus"):
                        chapter = self._parse_chapter_dict(item)
                        corpus[chapter.story_id] = chapter
        
        elif path.is_dir():
            # Directory of text files
            for file_path in tqdm(list(path.glob('**/*.txt')), desc="Loading corpus"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    story_id = file_path.stem
                    chapter = Chapter(
                        story_id=story_id,
                        chapter_index=0,
                        text=text,
                        token_length=len(text.split())
                    )
                    corpus[story_id] = chapter
        
        return corpus
    
    def _parse_chapter_dict(self, item: Dict) -> Chapter:
        """Parse chapter from dictionary"""
        story_id = item['id']
        
        return Chapter(
            story_id=story_id,
            chapter_index=0,
            text=item['text'],
            token_length=item.get('token_length', len(item['text'].split())),
            metadata=item.get('metadata')
        )
    
    def _build_chapter_index(self):
        """Build index for fast chapter lookup by story"""
        self._chapter_index = defaultdict(list)
        
        for doc_id, chapter in self._corpus.items():
            self._chapter_index[chapter.story_id].append(chapter)
        
        # Sort chapters by index
        for story_id in self._chapter_index:
            self._chapter_index[story_id].sort(key=lambda c: c.chapter_index)
    
    def get_chapters_by_ids(self, story_ids: List[str]) -> List[Chapter]:
        """
        Get chapters by story IDs
        
        Args:
            story_ids: List of story IDs
            
        Returns:
            List of Chapter objects
        """
        chapters = []
        missing = []
        
        for story_id in story_ids:
            if story_id in self._corpus:
                chapters.append(self._corpus[story_id])
            else:
                missing.append(story_id)
        
        if missing:
            logger.warning(f"Missing {len(missing)} chapters: {missing[:5]}...")
        
        return chapters
    
    def get_chapters_by_story(self, story_id: str) -> List[Chapter]:
        """
        Get all chapters for a story
        
        Args:
            story_id: Story identifier
            
        Returns:
            List of Chapter objects sorted by index
        """
        return self._chapter_index.get(story_id, [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'dataset_name': self.dataset_name,
            'dataset_config': self.dataset_config,
        }
        
        if self._dataset:
            stats['splits'] = {
                split: len(data) for split, data in self._dataset.items()
            }
        
        if self._corpus:
            stats['corpus'] = {
                'total_chapters': len(self._corpus),
                'total_stories': len(self._chapter_index),
                'avg_chapters_per_story': len(self._corpus) / max(len(self._chapter_index), 1),
                'avg_tokens_per_chapter': sum(c.token_length for c in self._corpus.values()) / len(self._corpus),
                'total_tokens': sum(c.token_length for c in self._corpus.values())
            }
        
        return stats
    
    def save_corpus(self, output_path: str):
        """
        Save corpus to file
        
        Args:
            output_path: Path to save corpus
        """
        if not self._corpus:
            logger.warning("No corpus loaded, nothing to save")
            return
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        corpus_list = []
        for chapter in self._corpus.values():
            corpus_list.append({
                'story_id': chapter.story_id,
                'chapter_index': chapter.chapter_index,
                'text': chapter.text,
                'token_length': chapter.token_length,
                'metadata': chapter.metadata
            })
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(corpus_list, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(corpus_list)} chapters to {output_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = MSRSDataLoader()
    
    # Load dataset
    dataset = loader.load_dataset()
    
    # Parse examples
    train_examples = loader.parse_examples(split="train", max_examples=10)
    
    print(f"\nExample Query: {train_examples[0].query}")
    print(f"Gold Documents: {train_examples[0].gold_documents}")
    print(f"Number of Answers: {len(train_examples[0].answers)}")
    
    # Load corpus (if available)
    try:
        corpus = loader.load_corpus()
        
        # Get chapters for first example
        chapters = loader.get_chapters_by_ids(train_examples[0].gold_documents)
        print(f"\nRetrieved {len(chapters)} chapters for first example")
        
        # Print statistics
        stats = loader.get_statistics()
        print(f"\nDataset Statistics:")
        print(json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"Could not load corpus: {str(e)}")
"""
RAFT Dataset Builder Module
Creates RAFT training examples with CoT and citations
"""

import logging
import random
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path

import openai
from tqdm import tqdm

logger = logging.getLogger("RAFT.DatasetBuilder")


@dataclass
class RAFTExample:
    """RAFT training example with CoT and citations"""
    query: str
    contexts: List[Dict[str, str]]  # List of {doc_id, text}
    oracle_ids: List[str]
    distractor_ids: List[str]
    reasoning: str  # CoT with citations
    answer: str
    metadata: Optional[Dict[str, Any]] = None


class CitationValidator:
    """Validates citations against source text"""
    
    QUOTE_START = "##begin_quote##"
    QUOTE_END = "##end_quote##"
    
    @staticmethod
    def extract_quotes(text: str) -> List[str]:
        """Extract all quotes from text"""
        pattern = f"{re.escape(CitationValidator.QUOTE_START)}(.*?){re.escape(CitationValidator.QUOTE_END)}"
        quotes = re.findall(pattern, text, re.DOTALL)
        return [q.strip() for q in quotes]
    
    @staticmethod
    def validate_quote(quote: str, source_text: str, fuzzy: bool = False) -> bool:
        """
        Validate that quote exists in source text
        
        Args:
            quote: Quote text to validate
            source_text: Source text to check against
            fuzzy: Allow minor differences (whitespace, punctuation)
            
        Returns:
            True if quote is valid
        """
        if not fuzzy:
            return quote in source_text
        
        # Normalize for fuzzy matching
        quote_norm = re.sub(r'\s+', ' ', quote.lower().strip())
        source_norm = re.sub(r'\s+', ' ', source_text.lower())
        
        return quote_norm in source_norm
    
    @staticmethod
    def validate_example(
        reasoning: str,
        oracle_texts: List[str],
        max_quote_length: int = 300
    ) -> Tuple[bool, List[str]]:
        """
        Validate all quotes in reasoning
        
        Args:
            reasoning: Reasoning text with quotes
            oracle_texts: List of oracle document texts
            max_quote_length: Maximum allowed quote length
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        quotes = CitationValidator.extract_quotes(reasoning)
        errors = []
        
        if not quotes:
            errors.append("No quotes found in reasoning")
            return False, errors
        
        # Combine oracle texts for validation
        combined_oracle = "\n".join(oracle_texts)
        
        for i, quote in enumerate(quotes):
            # Check length
            if len(quote) > max_quote_length:
                errors.append(f"Quote {i} too long: {len(quote)} chars")
            
            # Check existence in oracle
            if not CitationValidator.validate_quote(quote, combined_oracle, fuzzy=True):
                errors.append(f"Quote {i} not found in oracle texts: {quote[:50]}...")
        
        return len(errors) == 0, errors


class CoTGenerator:
    """Generates chain-of-thought reasoning with citations"""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.2,
        max_tokens: int = 1500,
        api_key: Optional[str] = None
    ):
        """
        Initialize CoT generator
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: OpenAI API key (or from env)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_key:
            openai.api_key = api_key
        
        logger.info(f"Initialized CoT generator with model: {model}")
    
    def generate_cot(
        self,
        query: str,
        oracle_texts: List[Tuple[str, str]],  # (doc_id, text)
        reference_answer: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate chain-of-thought reasoning with citations
        
        Args:
            query: Question to answer
            oracle_texts: List of (doc_id, text) tuples
            reference_answer: Optional reference answer for guidance
            
        Returns:
            Tuple of (reasoning, answer)
        """
        # Build prompt
        prompt = self._build_prompt(query, oracle_texts, reference_answer)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse reasoning and answer
            reasoning, answer = self._parse_response(content)
            
            return reasoning, answer
            
        except Exception as e:
            logger.error(f"Failed to generate CoT: {str(e)}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for CoT generation"""
        return """You are a helpful story QA assistant that generates detailed reasoning with exact citations.

Your task is to:
1. Read the provided story chapters carefully
2. Generate step-by-step reasoning to answer the question
3. Include EXACT verbatim quotes from the chapters to support your reasoning
4. Mark each quote with ##begin_quote## and ##end_quote##
5. End with a concise final answer after "##Answer:"

Requirements:
- Use 1-3 quotes from the chapters
- Each quote must be VERBATIM (exact text from the chapter)
- Quotes should be relevant and directly support the reasoning
- Keep quotes under 300 characters each
- Ignore any irrelevant information
- Be thorough but concise"""
    
    def _build_prompt(
        self,
        query: str,
        oracle_texts: List[Tuple[str, str]],
        reference_answer: Optional[str]
    ) -> str:
        """Build user prompt"""
        prompt_parts = [f"Question: {query}\n"]
        
        # Add contexts
        prompt_parts.append("\nContext (Story Chapters):")
        for doc_id, text in oracle_texts:
            prompt_parts.append(f"\n[Chapter: {doc_id}]")
            prompt_parts.append(text)
            prompt_parts.append("")
        
        # Add reference if available
        if reference_answer:
            prompt_parts.append(f"\nReference Answer: {reference_answer}")
        
        prompt_parts.append("\nGenerate reasoning with exact quotes and a final answer:")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, content: str) -> Tuple[str, str]:
        """Parse reasoning and answer from response"""
        # Split on ##Answer:
        parts = content.split("##Answer:")
        
        if len(parts) == 2:
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            # Fallback: treat entire content as reasoning
            reasoning = content
            answer = ""
        
        return reasoning, answer


class RAFTDatasetBuilder:
    """Builds RAFT training dataset"""
    
    def __init__(
        self,
        retrieval_system,
        cot_generator: CoTGenerator,
        oracle_percentage: float = 0.8,
        num_distractors: int = 4,
        distractor_pool_size: int = 20,
        max_quote_length: int = 300,
        min_quotes: int = 1,
        max_quotes: int = 3
    ):
        """
        Initialize RAFT dataset builder
        
        Args:
            retrieval_system: RetrievalSystem instance
            cot_generator: CoTGenerator instance
            oracle_percentage: Percentage of examples with oracle (P in RAFT)
            num_distractors: Number of distractors per example
            distractor_pool_size: Size of distractor candidate pool
            max_quote_length: Maximum quote length
            min_quotes: Minimum quotes per example
            max_quotes: Maximum quotes per example
        """
        self.retrieval_system = retrieval_system
        self.cot_generator = cot_generator
        self.oracle_percentage = oracle_percentage
        self.num_distractors = num_distractors
        self.distractor_pool_size = distractor_pool_size
        self.max_quote_length = max_quote_length
        self.min_quotes = min_quotes
        self.max_quotes = max_quotes
        
        self.validator = CitationValidator()
        
        logger.info(
            f"Initialized RAFT builder: P={oracle_percentage}, "
            f"distractors={num_distractors}"
        )
    
    def build_raft_example(
        self,
        query: str,
        oracle_docs: List[Tuple[str, str]],  # (doc_id, text)
        reference_answers: List[str],
        example_id: str,
        include_oracle: bool = True
    ) -> Optional[RAFTExample]:
        """
        Build single RAFT example
        
        Args:
            query: Question
            oracle_docs: List of oracle (doc_id, text) tuples
            reference_answers: List of reference answers
            example_id: Example identifier
            include_oracle: Whether to include oracle in contexts
            
        Returns:
            RAFTExample or None if validation fails
        """
        try:
            # Retrieve distractors
            distractor_docs = self._get_distractors(
                query,
                oracle_docs,
                include_oracle
            )
            
            # Build contexts
            contexts, oracle_ids, distractor_ids = self._build_contexts(
                oracle_docs,
                distractor_docs,
                include_oracle
            )
            
            if not contexts:
                logger.warning(f"No contexts for example {example_id}")
                return None
            
            # Generate CoT with citations
            oracle_texts = [text for _, text in oracle_docs]
            reference = reference_answers[0] if reference_answers else None
            
            reasoning, answer = self.cot_generator.generate_cot(
                query,
                oracle_docs,
                reference
            )
            
            # Validate citations
            is_valid, errors = self.validator.validate_example(
                reasoning,
                oracle_texts,
                self.max_quote_length
            )
            
            if not is_valid:
                logger.warning(
                    f"Validation failed for {example_id}: {', '.join(errors)}"
                )
                return None
            
            # Check quote count
            quotes = self.validator.extract_quotes(reasoning)
            if len(quotes) < self.min_quotes or len(quotes) > self.max_quotes:
                logger.warning(
                    f"Invalid quote count for {example_id}: {len(quotes)}"
                )
                return None
            
            # Create example
            example = RAFTExample(
                query=query,
                contexts=contexts,
                oracle_ids=oracle_ids,
                distractor_ids=distractor_ids,
                reasoning=reasoning,
                answer=answer,
                metadata={
                    'id': example_id,
                    'include_oracle': include_oracle,
                    'num_quotes': len(quotes),
                    'reference_answers': reference_answers
                }
            )
            
            return example
            
        except Exception as e:
            logger.error(f"Failed to build example {example_id}: {str(e)}")
            return None
    
    def _get_distractors(
        self,
        query: str,
        oracle_docs: List[Tuple[str, str]],
        include_oracle: bool
    ) -> List[Tuple[str, str]]:
        """Get distractor documents"""
        # Retrieve candidates
        results = self.retrieval_system.retrieve(
            query,
            top_k=self.distractor_pool_size
        )
        
        # Filter out oracles
        oracle_ids = {doc_id for doc_id, _ in oracle_docs}
        distractors = [
            (r.doc_id, r.text)
            for r in results
            if r.doc_id not in oracle_ids
        ]
        
        # Sample distractors
        num_needed = self.num_distractors
        if include_oracle:
            # Leave room for oracles
            num_needed = min(num_needed, max(1, self.num_distractors - len(oracle_docs)))
        
        if len(distractors) > num_needed:
            distractors = random.sample(distractors, num_needed)
        
        return distractors
    
    def _build_contexts(
        self,
        oracle_docs: List[Tuple[str, str]],
        distractor_docs: List[Tuple[str, str]],
        include_oracle: bool
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """Build context list with shuffling"""
        contexts = []
        oracle_ids = []
        distractor_ids = []
        
        # Add oracles if including
        if include_oracle:
            for doc_id, text in oracle_docs:
                contexts.append({'doc_id': doc_id, 'text': text})
                oracle_ids.append(doc_id)
        
        # Add distractors
        for doc_id, text in distractor_docs:
            contexts.append({'doc_id': doc_id, 'text': text})
            distractor_ids.append(doc_id)
        
        # Shuffle to avoid positional bias
        random.shuffle(contexts)
        
        return contexts, oracle_ids, distractor_ids
    
    def build_dataset(
        self,
        examples: List[Dict],  # List of {query, oracle_docs, answers, id}
        output_path: str,
        max_examples: Optional[int] = None
    ) -> List[RAFTExample]:
        """
        Build full RAFT dataset
        
        Args:
            examples: List of example dicts with query, oracle_docs, answers, id
            output_path: Path to save dataset
            max_examples: Maximum number of examples to process
            
        Returns:
            List of RAFTExample objects
        """
        if max_examples:
            examples = examples[:max_examples]
        
        logger.info(f"Building RAFT dataset from {len(examples)} examples")
        
        raft_examples = []
        failed = 0
        
        for example_dict in tqdm(examples, desc="Building RAFT dataset"):
            # Determine if this example includes oracle
            include_oracle = random.random() < self.oracle_percentage
            
            # Build example
            raft_example = self.build_raft_example(
                query=example_dict['query'],
                oracle_docs=example_dict['oracle_docs'],
                reference_answers=example_dict['answers'],
                example_id=example_dict['id'],
                include_oracle=include_oracle
            )
            
            if raft_example:
                raft_examples.append(raft_example)
            else:
                failed += 1
        
        logger.info(
            f"Built {len(raft_examples)} RAFT examples "
            f"({failed} failed validation)"
        )
        
        # Save dataset
        self.save_dataset(raft_examples, output_path)
        
        return raft_examples
    
    def save_dataset(self, examples: List[RAFTExample], output_path: str):
        """Save RAFT dataset to JSONL"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for example in examples:
                # Convert to chat format
                chat_example = self.to_chat_format(example)
                f.write(json.dumps(chat_example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def to_chat_format(self, example: RAFTExample) -> Dict:
        """Convert RAFT example to chat format for training"""
        # Build context section
        context_parts = []
        for ctx in example.contexts:
            context_parts.append(f"\n[Chapter: {ctx['doc_id']}]")
            context_parts.append(ctx['text'])
        
        context_str = "\n".join(context_parts)
        
        # Build user message
        user_message = f"""Question: {example.query}

Context:
{context_str}

Please provide your reasoning with exact quotes from the relevant chapters, then give your final answer."""
        
        # Build assistant message
        assistant_message = f"""{example.reasoning}

##Answer: {example.answer}"""
        
        # Create chat format
        chat_format = {
            "messages": [
                {
                    "role": "system",
                    "content": self._get_training_system_prompt()
                },
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            ],
            "metadata": example.metadata
        }
        
        return chat_format
    
    def _get_training_system_prompt(self) -> str:
        """Get system prompt for training"""
        return """You are a helpful story QA assistant. Use the provided context to answer questions accurately.

Instructions:
1. Read all provided chapters carefully
2. Identify relevant information for the question
3. Quote EXACT text from relevant chapters using ##begin_quote## and ##end_quote##
4. Ignore irrelevant chapters
5. Provide clear reasoning
6. End with a concise answer after "##Answer:"

Remember: Always cite your sources with verbatim quotes."""
    
    def get_statistics(self, examples: List[RAFTExample]) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not examples:
            return {}
        
        total = len(examples)
        with_oracle = sum(1 for e in examples if e.oracle_ids)
        
        total_contexts = sum(len(e.contexts) for e in examples)
        total_oracles = sum(len(e.oracle_ids) for e in examples)
        total_distractors = sum(len(e.distractor_ids) for e in examples)
        
        # Count quotes
        quote_counts = []
        for example in examples:
            quotes = self.validator.extract_quotes(example.reasoning)
            quote_counts.append(len(quotes))
        
        stats = {
            'total_examples': total,
            'examples_with_oracle': with_oracle,
            'examples_without_oracle': total - with_oracle,
            'oracle_percentage': with_oracle / total if total > 0 else 0,
            'avg_contexts_per_example': total_contexts / total,
            'avg_oracles_per_example': total_oracles / total,
            'avg_distractors_per_example': total_distractors / total,
            'avg_quotes_per_example': sum(quote_counts) / len(quote_counts),
            'min_quotes': min(quote_counts),
            'max_quotes': max(quote_counts)
        }
        
        return stats


def prepare_examples_from_loader(
    data_loader,
    split: str = "train"
) -> List[Dict]:
    """
    Prepare examples from data loader for RAFT building
    
    Args:
        data_loader: MSRSDataLoader instance with loaded corpus
        split: Dataset split to use
        
    Returns:
        List of example dicts ready for RAFT building
    """
    # Parse examples
    qa_examples = data_loader.parse_examples(split=split)
    
    prepared = []
    for qa_example in qa_examples:
        # Get oracle chapter texts
        oracle_chapters = data_loader.get_chapters_by_ids(qa_example.gold_documents)
        
        if not oracle_chapters:
            logger.warning(f"No oracle chapters found for {qa_example.id}")
            continue
        
        oracle_docs = [
            (chapter.doc_id, chapter.text)
            for chapter in oracle_chapters
        ]
        
        example_dict = {
            'id': qa_example.id,
            'query': qa_example.query,
            'oracle_docs': oracle_docs,
            'answers': qa_example.answers
        }
        prepared.append(example_dict)
    
    return prepared


# Example usage
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Mock components for demonstration
    class MockRetrievalSystem:
        def retrieve(self, query, top_k=10):
            # Mock retrieval results
            from dataclasses import dataclass
            @dataclass
            class MockResult:
                doc_id: str
                text: str
                score: float
            
            return [
                MockResult(f"distractor_{i}", f"Distractor text {i}", 0.5)
                for i in range(top_k)
            ]
    
    # Example with mock OpenAI (requires real API key)
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("Demo mode: skipping actual API calls")
        print("\nTo use RAFT builder:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Initialize components:")
        print("   - RetrievalSystem with indexed corpus")
        print("   - CoTGenerator with OpenAI model")
        print("   - RAFTDatasetBuilder")
        print("3. Prepare examples from data loader")
        print("4. Call build_dataset()")
    else:
        print("Run with --demo flag for demo mode")
        print("Full example requires OpenAI API key and indexed corpus")
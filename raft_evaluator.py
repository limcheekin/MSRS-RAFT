"""
RAFT Evaluator Module
Comprehensive RAG evaluation with retrieval and generation metrics
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch

logger = logging.getLogger("RAFT.Evaluator")


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    query: str
    retrieved_docs: List[str]
    gold_docs: List[str]
    generated_answer: str
    reference_answers: List[str]
    
    # Retrieval metrics
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_relevance: Optional[float] = None
    
    # Generation metrics
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    answer_correctness: Optional[float] = None
    
    # Traditional metrics
    rouge_l: Optional[float] = None
    bleu: Optional[float] = None
    bert_score_f1: Optional[float] = None
    
    metadata: Optional[Dict[str, Any]] = None


class RetrievalMetrics:
    """Compute retrieval quality metrics"""
    
    @staticmethod
    def context_precision(
        retrieved_docs: List[str],
        gold_docs: List[str]
    ) -> float:
        """
        Calculate context precision: fraction of retrieved that are relevant
        
        Args:
            retrieved_docs: List of retrieved document IDs
            gold_docs: List of gold document IDs
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        gold_set = set(gold_docs)
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc in gold_set)
        
        return relevant_retrieved / len(retrieved_docs)
    
    @staticmethod
    def context_recall(
        retrieved_docs: List[str],
        gold_docs: List[str]
    ) -> float:
        """
        Calculate context recall: fraction of gold docs that were retrieved
        
        Args:
            retrieved_docs: List of retrieved document IDs
            gold_docs: List of gold document IDs
            
        Returns:
            Recall score (0-1)
        """
        if not gold_docs:
            return 1.0  # No gold docs to recall
        
        retrieved_set = set(retrieved_docs)
        retrieved_gold = sum(1 for doc in gold_docs if doc in retrieved_set)
        
        return retrieved_gold / len(gold_docs)
    
    @staticmethod
    def context_f1(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class GenerationMetrics:
    """Compute generation quality metrics"""
    
    def __init__(self, llm_judge_model: Optional[str] = None):
        """
        Initialize generation metrics
        
        Args:
            llm_judge_model: Optional LLM model for judge-based metrics
        """
        self.llm_judge_model = llm_judge_model
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def compute_rouge_l(
        self,
        generated: str,
        references: List[str]
    ) -> float:
        """
        Compute ROUGE-L score
        
        Args:
            generated: Generated answer
            references: List of reference answers
            
        Returns:
            Max ROUGE-L F1 score across references
        """
        if not references:
            return 0.0
        
        scores = []
        for reference in references:
            score = self.rouge_scorer.score(reference, generated)
            scores.append(score['rougeL'].fmeasure)
        
        return max(scores)
    
    def compute_bleu(
        self,
        generated: str,
        references: List[str]
    ) -> float:
        """
        Compute BLEU score
        
        Args:
            generated: Generated answer
            references: List of reference answers
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Tokenize
            generated_tokens = generated.lower().split()
            reference_tokens_list = [ref.lower().split() for ref in references]
            
            # Compute with smoothing
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(
                reference_tokens_list,
                generated_tokens,
                smoothing_function=smoothing
            )
            
            return score
            
        except ImportError:
            logger.warning("NLTK not available for BLEU computation")
            return 0.0
    
    def compute_bert_score(
        self,
        generated: str,
        references: List[str]
    ) -> float:
        """
        Compute BERTScore
        
        Args:
            generated: Generated answer
            references: List of reference answers
            
        Returns:
            Max BERTScore F1 across references
        """
        if not references:
            return 0.0
        
        try:
            # Compute BERTScore against all references
            P, R, F1 = bert_score(
                [generated] * len(references),
                references,
                lang='en',
                verbose=False
            )
            
            # Return max F1
            return float(F1.max().item())
            
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {str(e)}")
            return 0.0
    
    def compute_faithfulness(
        self,
        generated: str,
        contexts: List[str]
    ) -> float:
        """
        Compute faithfulness: whether answer is grounded in context
        
        Args:
            generated: Generated answer
            contexts: List of context texts
            
        Returns:
            Faithfulness score (0-1)
        """
        if not self.llm_judge_model:
            # Fallback: simple keyword overlap
            return self._faithfulness_heuristic(generated, contexts)
        
        # Use LLM judge (requires implementation)
        return self._faithfulness_llm_judge(generated, contexts)
    
    def _faithfulness_heuristic(
        self,
        generated: str,
        contexts: List[str]
    ) -> float:
        """Heuristic faithfulness based on token overlap"""
        gen_tokens = set(generated.lower().split())
        
        if not gen_tokens:
            return 1.0
        
        # Combine all contexts
        context_text = " ".join(contexts).lower()
        context_tokens = set(context_text.split())
        
        # Calculate overlap
        overlap = len(gen_tokens & context_tokens)
        return overlap / len(gen_tokens)
    
    def _faithfulness_llm_judge(
        self,
        generated: str,
        contexts: List[str]
    ) -> float:
        """LLM-based faithfulness evaluation"""
        # Placeholder for LLM judge implementation
        # Would call OpenAI/Anthropic API to judge faithfulness
        logger.warning("LLM judge not implemented, using heuristic")
        return self._faithfulness_heuristic(generated, contexts)
    
    def compute_answer_relevance(
        self,
        generated: str,
        query: str
    ) -> float:
        """
        Compute answer relevance to query
        
        Args:
            generated: Generated answer
            query: Original query
            
        Returns:
            Relevance score (0-1)
        """
        if not self.llm_judge_model:
            # Fallback: simple keyword overlap
            return self._relevance_heuristic(generated, query)
        
        # Use LLM judge
        return self._relevance_llm_judge(generated, query)
    
    def _relevance_heuristic(
        self,
        generated: str,
        query: str
    ) -> float:
        """Heuristic relevance based on query term coverage"""
        query_tokens = set(query.lower().split())
        gen_tokens = set(generated.lower().split())
        
        if not query_tokens:
            return 1.0
        
        overlap = len(query_tokens & gen_tokens)
        return overlap / len(query_tokens)
    
    def _relevance_llm_judge(
        self,
        generated: str,
        query: str
    ) -> float:
        """LLM-based relevance evaluation"""
        logger.warning("LLM judge not implemented, using heuristic")
        return self._relevance_heuristic(generated, query)


class RAFTEvaluator:
    """Complete RAFT evaluation system"""
    
    def __init__(
        self,
        config,
        retrieval_system=None,
        model=None,
        tokenizer=None
    ):
        """
        Initialize RAFT evaluator
        
        Args:
            config: RAFTConfig object
            retrieval_system: Optional RetrievalSystem for retrieval evaluation
            model: Optional model for generation
            tokenizer: Optional tokenizer for generation
        """
        self.config = config
        self.retrieval_system = retrieval_system
        self.model = model
        self.tokenizer = tokenizer
        
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics(
            llm_judge_model=config.evaluation.ragas_llm if config.evaluation.compute_faithfulness else None
        )
        
        logger.info("Initialized RAFT Evaluator")
    
    def evaluate_example(
        self,
        query: str,
        gold_docs: List[str],
        reference_answers: List[str],
        retrieved_docs: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        generated_answer: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single example
        
        Args:
            query: Query text
            gold_docs: List of gold document IDs
            reference_answers: List of reference answers
            retrieved_docs: List of retrieved document IDs (or will retrieve)
            retrieved_texts: List of retrieved texts for faithfulness
            generated_answer: Generated answer (or will generate)
            
        Returns:
            EvaluationResult object
        """
        # Retrieve if not provided
        if retrieved_docs is None and self.retrieval_system:
            retrieval_results = self.retrieval_system.retrieve(
                query,
                top_k=self.config.retrieval.top_k
            )
            retrieved_docs = [r.doc_id for r in retrieval_results]
            retrieved_texts = [r.text for r in retrieval_results]
        
        # Generate if not provided
        if generated_answer is None and self.model:
            generated_answer = self.generate_answer(
                query,
                retrieved_texts or []
            )
        
        # Compute retrieval metrics
        context_precision = None
        context_recall = None
        context_f1 = None
        
        if retrieved_docs and self.config.evaluation.compute_context_precision:
            context_precision = self.retrieval_metrics.context_precision(
                retrieved_docs,
                gold_docs
            )
        
        if retrieved_docs and self.config.evaluation.compute_context_recall:
            context_recall = self.retrieval_metrics.context_recall(
                retrieved_docs,
                gold_docs
            )
        
        if context_precision is not None and context_recall is not None:
            context_f1 = self.retrieval_metrics.context_f1(
                context_precision,
                context_recall
            )
        
        # Compute generation metrics
        faithfulness = None
        answer_relevance = None
        rouge_l = None
        bleu = None
        bert_score_f1 = None
        
        if generated_answer:
            if self.config.evaluation.compute_faithfulness and retrieved_texts:
                faithfulness = self.generation_metrics.compute_faithfulness(
                    generated_answer,
                    retrieved_texts
                )
            
            if self.config.evaluation.compute_answer_relevance:
                answer_relevance = self.generation_metrics.compute_answer_relevance(
                    generated_answer,
                    query
                )
            
            if self.config.evaluation.compute_rouge:
                rouge_l = self.generation_metrics.compute_rouge_l(
                    generated_answer,
                    reference_answers
                )
            
            if self.config.evaluation.compute_bleu:
                bleu = self.generation_metrics.compute_bleu(
                    generated_answer,
                    reference_answers
                )
            
            if self.config.evaluation.compute_bertscore:
                bert_score_f1 = self.generation_metrics.compute_bert_score(
                    generated_answer,
                    reference_answers
                )
        
        return EvaluationResult(
            query=query,
            retrieved_docs=retrieved_docs or [],
            gold_docs=gold_docs,
            generated_answer=generated_answer or "",
            reference_answers=reference_answers,
            context_precision=context_precision,
            context_recall=context_recall,
            context_relevance=context_f1,  # Using F1 as relevance proxy
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            answer_correctness=rouge_l,  # Using ROUGE-L as correctness proxy
            rouge_l=rouge_l,
            bleu=bleu,
            bert_score_f1=bert_score_f1
        )
    
    def generate_answer(
        self,
        query: str,
        contexts: List[str]
    ) -> str:
        """
        Generate answer using the model
        
        Args:
            query: Query text
            contexts: List of context texts
            
        Returns:
            Generated answer
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for generation")
        
        # Build prompt
        context_str = "\n\n".join([
            f"[Context {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful story QA assistant. Use the provided context to answer questions accurately. Quote exact text from relevant contexts using ##begin_quote## and ##end_quote##. End with ##Answer: followed by your final answer."
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context_str}\n\nProvide your reasoning with quotes and final answer:"
            }
        ]
        
        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_seq_length
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.training.max_new_tokens,
                temperature=self.config.training.temperature,
                top_p=self.config.training.top_p,
                do_sample=True if self.config.training.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract answer after ##Answer:
        if "##Answer:" in generated:
            answer = generated.split("##Answer:")[-1].strip()
        else:
            answer = generated.strip()
        
        return answer
    
    def evaluate_dataset(
        self,
        examples: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate full dataset
        
        Args:
            examples: List of example dicts with query, gold_docs, answers
            output_path: Optional path to save detailed results
            
        Returns:
            Dictionary of aggregated metrics
        """
        logger.info(f"Evaluating {len(examples)} examples")
        
        results = []
        
        for example in tqdm(examples, desc="Evaluating"):
            result = self.evaluate_example(
                query=example['query'],
                gold_docs=example['gold_docs'],
                reference_answers=example['answers'],
                retrieved_docs=example.get('retrieved_docs'),
                retrieved_texts=example.get('retrieved_texts'),
                generated_answer=example.get('generated_answer')
            )
            results.append(result)
        
        # Aggregate metrics
        metrics = self.aggregate_results(results)
        
        # Save detailed results if requested
        if output_path:
            self.save_results(results, output_path)
        
        return metrics
    
    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation results
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            Dictionary of aggregated metrics
        """
        metrics = {}
        
        # Helper to compute mean of non-None values
        def mean_metric(values):
            valid = [v for v in values if v is not None]
            return np.mean(valid) if valid else None
        
        # Retrieval metrics
        if any(r.context_precision is not None for r in results):
            metrics['context_precision'] = mean_metric([r.context_precision for r in results])
        
        if any(r.context_recall is not None for r in results):
            metrics['context_recall'] = mean_metric([r.context_recall for r in results])
        
        if any(r.context_relevance is not None for r in results):
            metrics['context_relevance'] = mean_metric([r.context_relevance for r in results])
        
        # Generation metrics
        if any(r.faithfulness is not None for r in results):
            metrics['faithfulness'] = mean_metric([r.faithfulness for r in results])
        
        if any(r.answer_relevance is not None for r in results):
            metrics['answer_relevance'] = mean_metric([r.answer_relevance for r in results])
        
        if any(r.answer_correctness is not None for r in results):
            metrics['answer_correctness'] = mean_metric([r.answer_correctness for r in results])
        
        # Traditional metrics
        if any(r.rouge_l is not None for r in results):
            metrics['rouge_l'] = mean_metric([r.rouge_l for r in results])
        
        if any(r.bleu is not None for r in results):
            metrics['bleu'] = mean_metric([r.bleu for r in results])
        
        if any(r.bert_score_f1 is not None for r in results):
            metrics['bert_score_f1'] = mean_metric([r.bert_score_f1 for r in results])
        
        # Count statistics
        metrics['total_examples'] = len(results)
        
        return metrics
    
    def save_results(
        self,
        results: List[EvaluationResult],
        output_path: str
    ):
        """
        Save detailed results to file
        
        Args:
            results: List of EvaluationResult objects
            output_path: Path to save results
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dicts
        results_dicts = [asdict(r) for r in results]
        
        # Save as JSONL
        with open(path, 'w', encoding='utf-8') as f:
            for result_dict in results_dicts:
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(results)} detailed results to {output_path}")
    
    def compare_systems(
        self,
        system_results: Dict[str, List[EvaluationResult]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple systems
        
        Args:
            system_results: Dict mapping system name to results list
            
        Returns:
            Dictionary of system comparisons
        """
        comparison = {}
        
        for system_name, results in system_results.items():
            metrics = self.aggregate_results(results)
            comparison[system_name] = metrics
        
        return comparison
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        if 'total_examples' in metrics:
            print(f"\nTotal Examples: {metrics['total_examples']}")
        
        # Retrieval metrics
        print("\n--- Retrieval Metrics ---")
        for key in ['context_precision', 'context_recall', 'context_relevance']:
            if key in metrics and metrics[key] is not None:
                print(f"  {key}: {metrics[key]:.4f}")
        
        # Generation metrics
        print("\n--- Generation Metrics ---")
        for key in ['faithfulness', 'answer_relevance', 'answer_correctness']:
            if key in metrics and metrics[key] is not None:
                print(f"  {key}: {metrics[key]:.4f}")
        
        # Traditional metrics
        print("\n--- Traditional Metrics ---")
        for key in ['rouge_l', 'bleu', 'bert_score_f1']:
            if key in metrics and metrics[key] is not None:
                print(f"  {key}: {metrics[key]:.4f}")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock example data
    examples = [
        {
            'query': 'What happened in the story?',
            'gold_docs': ['doc1', 'doc2'],
            'answers': ['The hero saved the day.', 'A hero emerged victorious.'],
            'retrieved_docs': ['doc1', 'doc3'],
            'retrieved_texts': ['Hero text...', 'Other text...'],
            'generated_answer': 'The hero saved everyone.'
        }
    ]
    
    # Create mock config
    from raft_config import RAFTConfig
    config = RAFTConfig()
    
    # Initialize evaluator
    evaluator = RAFTEvaluator(config)
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(examples)
    
    # Print results
    evaluator.print_metrics(metrics)
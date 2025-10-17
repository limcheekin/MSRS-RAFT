# Code Review by Gemini 2.5 Pro

## Conversation
URL: https://gemini.google.com/u/2/app/5ef8850129d53338

## Prompt
Carefully review the paper `2403.10131v2 - RAFT.pdf` and the original code implementation of the paper `shishirpatil-gorilla.txt` and custom code implementation `limcheekin-msrs-raft.txt` in detail, ensure the code implemented in `limcheekin-msrs-raft.txt` is correct and complete compare to the paper and original implementation.

## Answer
Yes, the implementation in **limcheekin/MSRS-RAFT is both correct and complete** when compared to the RAFT paper and the original shishirpatil/gorilla implementation.

In fact, it goes beyond a simple re-implementation. The limcheekin/MSRS-RAFT code represents a more **robust, modular, and end-to-end toolkit** that not only correctly implements all core concepts of the RAFT paper but also adds significant enhancements in code quality, training, evaluation, and usability.

---

### **Executive Summary**

* **Correctness:** The limcheekin implementation faithfully adheres to all key principles of the RAFT methodology described in the paper, including the use of distractor documents, the oracle percentage parameter (P), chain-of-thought answers, and verbatim citations.

* **Completeness:** It is more complete than the original shishirpatil/gorilla implementation. While the original code focuses primarily on dataset generation and outsources the training process to external platforms like Azure AI Studio, the limcheekin version provides a **fully integrated, end-to-end pipeline** that includes data loading, index building, dataset generation, model fine-tuning with Unsloth, and a comprehensive evaluation suite.

* **Enhancements:** The limcheekin code introduces several improvements, most notably a **CitationValidator** to ensure generated quotes are factually grounded in the source documents, a more comprehensive evaluation framework, and a significantly better software architecture that is modular and well-tested.

---

### **Detailed Comparison and Validation**

Here is a detailed breakdown comparing the limcheekin implementation against the RAFT paper and the original shishirpatil code.

#### **1\. Core RAFT Principles**

The RAFT paper outlines a specific training recipe. The limcheekin implementation correctly captures all of these core ideas.

| RAFT Principle | RAFT Paper Description | shishirpatil/gorilla Implementation | limcheekin/MSRS-RAFT Implementation | Validation |
| :---- | :---- | :---- | :---- | :---- |
| **Distractor Documents** | Train the model with a mix of "golden" (oracle) documents and "distractor" documents to teach it to ignore irrelevant information.  | Implemented via the \--distractors argument in raft.py, which adds randomly selected chunks to the context.  | Implemented in RAFTDatasetBuilder via the num\_distractors parameter. It retrieves candidate documents and samples distractors that are not in the oracle set.  | **Correct** |
| **Oracle Percentage (P)** | For (1-P)% of examples, the oracle document is *excluded* to compel the model to memorize some domain knowledge.  | Implemented via the \--p argument, where oracle \= random.uniform(0, 1\) \< p determines if the golden chunk is included.  | Implemented via the oracle\_percentage parameter in RAFTDataConfig. The logic include\_oracle \= random.random() \< self.oracle\_percentage is used in the build loop.  | **Correct** |
| **Chain-of-Thought (CoT) Answers** | The model is trained to generate a chain-of-thought-style response that includes reasoning before the final answer to improve its reasoning ability.  | The prompt template for the judge model explicitly asks for step-by-step reasoning and a final answer tagged with \<ANSWER\>:.  | The CoTGenerator's system prompt asks for step-by-step reasoning, and the final training data is formatted to include both the reasoning and the final answer tagged with \#\#Answer:.  | **Correct** |
| **Verbatim Citations** | The reasoning should cite verbatim quotes from the source documents to ground the answer in evidence. Special markers like \#\#begin\_quote\#\# are used.  | The prompt template explicitly instructs the judge model to include \#\#begin\_quote\#\# and \#\#end\_quote\#\# around copied sentences.  | The CoTGenerator prompt also requires verbatim quotes marked with \#\#begin\_quote\#\# and \#\#end\_quote\#\#. Crucially, it adds a CitationValidator class to programmatically check that the generated quotes actually exist in the source text, a major enhancement for data quality.  | **Correct & Enhanced** |

---

#### **2\. Training and Fine-Tuning**

This is where the limcheekin implementation demonstrates its superior completeness.

* **shishirpatil/gorilla:** This repository **does not contain a training script**. Its purpose is to generate the dataset. The README.md and a how-to guide direct the user to fine-tune the model on external platforms like **Microsoft AI Studio**.

* **limcheekin/MSRS-RAFT:** This repository provides a complete, self-contained training solution in raft\_trainer.py.

  * It uses **Unsloth** for highly efficient QLoRA fine-tuning, which is a modern and powerful technique.

  * It correctly loads the base model, applies LoRA adapters, and uses the SFTTrainer from the trl library to perform the fine-tuning.

  * The entire training process is configurable via raft\_config.py and orchestrated by raft\_pipeline.py.

**Validation Verdict:** The limcheekin implementation is **complete** in this regard, whereas the original implementation is not. It provides a full, working training script that aligns with modern best practices for fine-tuning LLMs.

---

#### **3\. Evaluation**

The limcheekin implementation offers a much more sophisticated evaluation framework.

* **shishirpatil/gorilla:** Provides a basic eval.py script. It takes a file of questions, sends them to a deployed model endpoint, and saves the answers. It doesn't compute specific RAG metrics.

* **limcheekin/MSRS-RAFT:** Provides a comprehensive raft\_evaluator.py module that calculates a suite of metrics crucial for RAG systems:

  * **Retrieval Metrics:** context\_precision and context\_recall to evaluate the retriever's performance.

  * **Generation Metrics:** Heuristics for faithfulness (is the answer grounded in the context?) and answer\_relevance (does the answer address the query?).

  * **Standard Metrics:** ROUGE, BLEU, and BERTScore for semantic similarity to reference answers.

  * The example\_usage.py script includes examples for comparing the RAFT model against a 0-shot baseline and conducting a "distractor stress test", which directly validates the core hypothesis of the RAFT paper.

**Validation Verdict:** The limcheekin implementation provides a **more complete and relevant evaluation framework** that is better suited to assess the specific goals of the RAFT methodology.

---

#### **4\. Code Architecture and Usability**

The limcheekin repository is engineered to be a robust and reusable toolkit.

* **shishirpatil/gorilla:** The core logic is largely contained within raft.py, a procedural script driven by command-line arguments. It uses checkpointing to manage long-running jobs. While functional, it is less modular.

* **limcheekin/MSRS-RAFT:** This project follows modern software engineering principles:  
  * **Modularity:** The code is logically separated into distinct modules for configuration (raft\_config.py), data loading (raft\_data\_loader.py), retrieval (raft\_retrieval.py), dataset building (raft\_dataset\_builder.py), training (raft\_trainer.py), and evaluation (raft\_evaluator.py).  
  * **Object-Oriented Design:** Key components are implemented as classes (e.g., RAFTPipeline, RAFTTrainer, RAFTEvaluator), making the system extensible and easier to maintain.  
  * **End-to-End Orchestration:** raft\_pipeline.py orchestrates the entire workflow from data loading to final evaluation, either step-by-step or through a single run\_full\_pipeline command.

  * **Testing and Validation:** The inclusion of integration\_test.py, test\_installation.py, and validate\_complete\_system.py demonstrates a strong commitment to code correctness, robustness, and reliability.

**Validation Verdict:** The limcheekin implementation is **architecturally superior, more complete, and significantly more robust and user-friendly.**

### **Final Conclusion**

The limcheekin/MSRS-RAFT project is an exemplary implementation of the RAFT paper. It not only correctly translates the theoretical concepts into functional code but also builds a comprehensive, production-ready system around them. It is **correct** in its adherence to the RAFT recipe and **complete** in its provision of an end-to-end workflow, making it a more powerful and practical tool than the original implementation.
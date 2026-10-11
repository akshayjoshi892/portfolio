# Building LLMs From Scratch: From First Principles to Frontier Practice

> **How to read this file:** This is the book's front matter. It has three parts: the full Table of Contents with a budget for every chapter, the **Project Thread** you will build across the book, and the **environment setup**. To get a chapter, say "Write Chapter N".

**Total budget:** ~492 pages ≈ 235,000 words (≈ 480 words/page, including code).

---

## 1. Table of Contents

Every section follows the loop: **🧭 Problem → 💡 Intuition → 🖼️ Visual → 📐 Math → 🛠️ Code → 🔬 Experiment → ⚠️ Failure Modes → 🧠 Expert Insight → ✅ Check Your Instinct → 📌 Recap.**
Every chapter ends with a **🏗️ Project Thread milestone**.

### Part 0 — Foundations (46 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 1 | The Big Picture: What an LLM Really Is | 6 | 2,900 |
| 2 | Python & NumPy for ML: Thinking in Tensors | 8 | 3,800 |
| 3 | Linear Algebra as Geometry | 10 | 4,800 |
| 4 | Calculus & Gradients: The Physics of Learning | 10 | 4,800 |
| 5 | Probability & Information Theory: The Language of Prediction | 12 | 5,800 |

**Ch 1 — The Big Picture**
- 1.1 An LLM is a compression machine: "predict the next token" as the whole idea
- 1.2 The full lifecycle map: data → tokenizer → pretrain → post-train → inference
- 1.3 What "parameters", "training" and "inference" physically are (bytes on a GPU)
- 1.4 A 60-line toy that learns to autocomplete: see the whole thing once before building it for real
- 1.5 How to study this book: the instinct loop, the repo, pacing plans (8-week / 16-week / 6-month)

**Ch 2 — Python & NumPy for ML**
- 2.1 Arrays as n-dimensional grids; shape, dtype, strides and memory layout
- 2.2 Broadcasting: the rule, the mental picture, the classic bugs
- 2.3 Vectorization: why loops are slow and how to think without them
- 2.4 Indexing, fancy indexing, `reshape` vs `view` vs `transpose`, `einsum`
- 2.5 NumPy → PyTorch: tensors, devices, dtypes, `requires_grad`
- 2.6 Reproducibility: seeds, determinism, and debugging with shapes

**Ch 3 — Linear Algebra as Geometry**
- 3.1 Vectors as arrows and as feature lists; norms and distances
- 3.2 The dot product as similarity (the seed of attention)
- 3.3 Matrices as transformations: rotate, scale, project
- 3.4 Matrix multiplication four ways (row, column, outer-product, batched)
- 3.5 Rank, low-rank structure and why it matters (preview: LoRA)
- 3.6 Eigen/SVD intuition; orthogonality; high-dimensional geometry surprises

**Ch 4 — Calculus & Gradients**
- 4.1 The derivative as sensitivity: "wiggle the input, watch the output"
- 4.2 Partial derivatives and the gradient as the direction of steepest ascent
- 4.3 The chain rule as a relay race of local sensitivities
- 4.4 Jacobians, vector–Jacobian products, and why backprop runs backwards
- 4.5 Numerical gradient checking from scratch
- 4.6 Loss landscapes: convexity, saddles, flat vs sharp minima

**Ch 5 — Probability & Information Theory**
- 5.1 Random variables, distributions, expectation and variance
- 5.2 Categorical distributions and softmax: turning scores into beliefs
- 5.3 Likelihood and maximum-likelihood estimation
- 5.4 Entropy as surprise; bits vs nats
- 5.5 Cross-entropy and KL divergence: "the cost of believing the wrong thing"
- 5.6 Perplexity: the effective branching factor
- 5.7 Compression ⇔ prediction: Shannon's insight and why LLMs are compressors

### Part 1 — The Neural Network Core (54 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 6 | Build an Autograd Engine From Scratch | 14 | 6,700 |
| 7 | Neurons, MLPs & Loss Functions | 10 | 4,800 |
| 8 | Optimizers: From Gradient Descent to AdamW | 12 | 5,800 |
| 9 | Initialization & Activation Functions | 8 | 3,800 |
| 10 | Normalization, Residuals & Regularization | 10 | 4,800 |

**Ch 6 — Autograd Engine (micrograd-style)**
- 6.1 Computation graphs: every formula is a DAG
- 6.2 Backprop by hand on paper, node by node
- 6.3 The `Value` class: forward ops and local gradients
- 6.4 Topological sort and the `backward()` pass
- 6.5 Gradient accumulation bugs (`+=` vs `=`) and zeroing gradients
- 6.6 Verifying against PyTorch autograd; extending to tensors (a mini tensor autograd)

**Ch 7 — Neurons, MLPs & Loss Functions**
- 7.1 A neuron as a weighted vote plus a squash
- 7.2 Layers and MLPs; universal approximation intuition
- 7.3 Loss functions: MSE, cross-entropy, negative log-likelihood; why CE for LMs
- 7.4 Numerically stable softmax and log-softmax (the log-sum-exp trick)
- 7.5 Training an MLP from scratch with our engine, then in PyTorch `nn.Module`

**Ch 8 — Optimizers**
- 8.1 Gradient descent and the learning rate as step size
- 8.2 SGD and mini-batches: noise as a feature
- 8.3 Momentum: a heavy ball rolling downhill
- 8.4 RMSProp: per-parameter step sizes
- 8.5 Adam: momentum + adaptive scale + bias correction
- 8.6 AdamW: decoupled weight decay and why it matters
- 8.7 Writing every optimizer from scratch and racing them on a toy landscape
- 8.8 Preview: Lion, Adafactor, Muon, Sophia

**Ch 9 — Initialization & Activations**
- 9.1 Why initialization matters: exploding and vanishing activations
- 9.2 Xavier/Glorot and Kaiming/He derived from variance preservation
- 9.3 Activations: sigmoid, tanh, ReLU, dead neurons, GELU, SiLU
- 9.4 Gated activations: GLU and SwiGLU
- 9.5 Activation and gradient statistics dashboards (histograms per layer)

**Ch 10 — Normalization, Residuals & Regularization**
- 10.1 BatchNorm: the idea and why it fails for sequences
- 10.2 LayerNorm and RMSNorm from scratch
- 10.3 Residual connections: the gradient highway
- 10.4 Dropout as ensemble training
- 10.5 Overfitting, the bias-variance picture, double descent
- 10.6 Weight decay, early stopping, data augmentation for text

### Part 2 — Language Modeling Basics (38 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 11 | Next-Token Prediction & N-gram Models | 10 | 4,800 |
| 12 | Embeddings & the MLP Language Model | 12 | 5,800 |
| 13 | Sampling: Turning Probabilities Into Text | 8 | 3,800 |
| 14 | RNNs & LSTMs: The Road Not Taken | 8 | 3,800 |

**Ch 11 — Next-Token Prediction**
- 11.1 The chain rule of probability: text as a sequence of guesses
- 11.2 Bigram model by counting
- 11.3 Smoothing and the zero-probability problem
- 11.4 The same bigram as a neural net: counting vs learning are one idea
- 11.5 N-grams and the curse of context length
- 11.6 Train/val/test splits and measuring loss honestly

**Ch 12 — Embeddings & MLP LM (Bengio 2003)**
- 12.1 One-hot vectors and their failure
- 12.2 Embeddings as learned coordinates of meaning
- 12.3 The Bengio MLP LM, built step by step
- 12.4 Mini-batching, the LR finder, training diagnostics
- 12.5 Visualizing embeddings (PCA / t-SNE)
- 12.6 Scaling the MLP LM and hitting its wall

**Ch 13 — Sampling**
- 13.1 Greedy decoding and its repetition trap
- 13.2 Temperature: flattening and sharpening distributions
- 13.3 Top-k and top-p (nucleus) sampling
- 13.4 Repetition/frequency penalties, min-p, beam search
- 13.5 Building a reusable `generate()` for the whole book

**Ch 14 — RNNs & LSTMs**
- 14.1 Recurrence: a memory that updates each step
- 14.2 A vanilla RNN from scratch; backprop through time
- 14.3 Vanishing/exploding gradients, felt on real numbers
- 14.4 LSTM/GRU gates as learned valves
- 14.5 Why recurrence lost: sequential bottlenecks and the long-range problem

### Part 3 — Tokenization (26 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 15 | Tokenization & BPE From Scratch | 14 | 6,700 |
| 16 | Production Tokenizers: Bytes, WordPiece, Unigram & Pitfalls | 12 | 5,800 |

**Ch 15 — BPE From Scratch**
- 15.1 Characters vs words vs subwords: the trade-off triangle
- 15.2 Unicode, UTF-8 and bytes
- 15.3 The BPE algorithm: merging the most frequent pair
- 15.4 Implementing train / encode / decode
- 15.5 Regex pre-tokenization (GPT-2/GPT-4 style split patterns)
- 15.6 Making it fast: incremental pair counts and priority queues

**Ch 16 — Production Tokenizers**
- 16.1 Byte-level BPE and why it never sees an unknown token
- 16.2 WordPiece and Unigram LM (SentencePiece) from scratch
- 16.3 Special tokens, chat tokens, BOS/EOS/PAD
- 16.4 Vocabulary size trade-offs: compute, embedding memory, sequence length
- 16.5 Multilingual fairness and code tokenization (whitespace, indentation)
- 16.6 Tokenizer pitfalls: arithmetic, trailing spaces, "glitch tokens"
- 16.7 Training a tokenizer on YOUR data and measuring compression

### Part 4 — Attention & the Transformer (46 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 17 | Attention From First Principles | 12 | 5,800 |
| 18 | Multi-Head Attention & Positional Encodings | 12 | 5,800 |
| 19 | The Transformer Block & the Residual Stream | 10 | 4,800 |
| 20 | Build a GPT Line by Line | 12 | 5,800 |

**Ch 17 — Attention**
- 17.1 Attention as a soft dictionary lookup
- 17.2 Queries, keys and values: "what I seek, what I offer, what I give"
- 17.3 Scaled dot-product attention and why we divide by √d
- 17.4 Causal masking: no peeking at the future
- 17.5 The averaging trick: from loops to a matmul
- 17.6 Self-attention vs cross-attention

**Ch 18 — Multi-Head Attention & Position**
- 18.1 Multiple heads as parallel specialists
- 18.2 Efficient batched MHA with reshapes (B, H, T, D)
- 18.3 Why attention is permutation-invariant and needs position
- 18.4 Sinusoidal and learned positional embeddings
- 18.5 RoPE: rotating queries and keys, derived and implemented
- 18.6 ALiBi: a linear distance penalty

**Ch 19 — The Transformer Block**
- 19.1 The decoder block anatomy
- 19.2 Pre-norm vs post-norm: stability explained
- 19.3 The FFN as key–value memory
- 19.4 The residual stream mental model: a shared bus that layers read and write
- 19.5 Weight tying between embeddings and the LM head
- 19.6 Parameter and FLOP counting for a transformer

**Ch 20 — GPT Line by Line**
- 20.1 `GPTConfig` and the module layout
- 20.2 Embeddings → blocks → final norm → head
- 20.3 Initialization scheme (scaled residual init)
- 20.4 Training on Tiny Shakespeare → TinyStories
- 20.5 Loading GPT-2 weights into your own code and matching outputs
- 20.6 Ablations: remove attention, remove residuals, remove norms, and watch what breaks

### Part 5 — The Data Pipeline for ANY Data (30 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 21 | Sourcing, Cleaning & Deduplication | 12 | 5,800 |
| 22 | Quality, Safety Filtering & Domain Mixing | 8 | 3,800 |
| 23 | Formats, Sharding, Streaming & Custom Data | 10 | 4,800 |

**Ch 21 — Sourcing & Cleaning**
- 21.1 "Data is the model": what pretraining corpora contain (Common Crawl, code, books, papers)
- 21.2 Ethical, legal and licensing considerations; robots.txt
- 21.3 Scraping and text extraction (HTML boilerplate removal)
- 21.4 Normalization: Unicode, whitespace, language ID
- 21.5 Exact dedup (hashing) and near-dedup with MinHash + LSH from scratch

**Ch 22 — Quality & Mixing**
- 22.1 Heuristic filters (Gopher/C4 rules)
- 22.2 Model-based quality classifiers (fastText-style, perplexity filters)
- 22.3 PII detection and redaction
- 22.4 Toxicity filtering and its trade-offs
- 22.5 Domain mixing, upsampling and data schedules (DoReMi intuition)

**Ch 23 — Formats & Pipelines**
- 23.1 Storage formats: JSONL, Parquet, tokenized `.bin` memmaps
- 23.2 Sharding and deterministic shuffling
- 23.3 Streaming dataloaders and multi-worker loading
- 23.4 Sequence packing and document masking
- 23.5 Custom data: PDFs, code repos, chat logs, CSV/tables into text
- 23.6 A reusable `prepare_data.py` for any corpus

### Part 6 — Pretraining (36 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 24 | Anatomy of a Training Loop | 10 | 4,800 |
| 25 | Mixed Precision, Checkpointing & Resuming | 8 | 3,800 |
| 26 | Scaling Laws, Compute Budgets & Hyperparameters | 10 | 4,800 |
| 27 | Reading Loss Curves & Taming Instabilities | 8 | 3,800 |

**Ch 24 — Training Loop**
- 24.1 The five steps: batch, forward, loss, backward, step
- 24.2 LR schedules: warmup + cosine, WSD (warmup-stable-decay)
- 24.3 Gradient clipping: a seatbelt for gradients
- 24.4 Gradient accumulation: big batches on small GPUs
- 24.5 Logging: loss, grad-norm, LR, tokens/sec, MFU

**Ch 25 — Mixed Precision & Checkpoints**
- 25.1 Floating point formats: FP32, FP16, BF16, FP8 bit by bit
- 25.2 Autocast and loss scaling
- 25.3 Checkpoint contents: model, optimizer, scheduler, RNG, data position
- 25.4 Exact resumption and fault tolerance

**Ch 26 — Scaling Laws**
- 26.1 Power laws: loss vs params, data, compute
- 26.2 Kaplan (2020) vs Chinchilla (2022): the 20 tokens/param rule
- 26.3 The FLOPs ≈ 6ND budget calculator
- 26.4 Fitting your own mini scaling law
- 26.5 Hyperparameter selection, batch size and critical batch size
- 26.6 µP: transferring hyperparameters from small to large models

**Ch 27 — Loss Curves & Instabilities**
- 27.1 A field guide to loss curve shapes
- 27.2 Loss spikes, divergence and their causes
- 27.3 Fixes: QK-norm, z-loss, lower LR, skip batches, rewinding
- 27.4 Overtraining, data repetition and multi-epoch effects

### Part 7 — Scaling & Efficiency (34 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 28 | GPU Intuition & Memory Math | 8 | 3,800 |
| 29 | Making It Fast: Kernels, FlashAttention, Compile, Profiling | 10 | 4,800 |
| 30 | KV Cache, MQA & GQA | 6 | 2,900 |
| 31 | Distributed Training: DDP, Tensor, Pipeline, ZeRO/FSDP | 10 | 4,800 |

**Ch 28 — GPU & Memory**
- 28.1 GPU anatomy: SMs, HBM, SRAM, tensor cores
- 28.2 Compute-bound vs memory-bound; the roofline model
- 28.3 Memory math: params + grads + optimizer states + activations
- 28.4 Activation checkpointing: trading compute for memory

**Ch 29 — Making It Fast**
- 29.1 Kernel fusion and why memory traffic dominates
- 29.2 FlashAttention: tiling and online softmax, derived and simulated
- 29.3 `torch.compile` and CUDA graphs
- 29.4 A first Triton kernel (fused softmax)
- 29.5 Profiling with the PyTorch profiler; finding bottlenecks

**Ch 30 — KV Cache, MQA & GQA**
- 30.1 Why generation recomputes, and how to cache K and V
- 30.2 KV-cache memory math
- 30.3 Multi-query and grouped-query attention from scratch

**Ch 31 — Distributed Training**
- 31.1 Data parallel (DDP) and all-reduce
- 31.2 ZeRO stages 1–3 and FSDP
- 31.3 Tensor parallelism (Megatron-style column/row splits)
- 31.4 Pipeline parallelism and bubbles
- 31.5 3D parallelism and choosing a strategy; multi-GPU on a budget

### Part 8 — Modern Architecture Upgrades (30 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 32 | From GPT-2 to a Llama-Style Model | 12 | 5,800 |
| 33 | Mixture of Experts | 8 | 3,800 |
| 34 | Beyond Attention: SSMs, Mamba, Linear Attention & Multimodality | 10 | 4,800 |

**Ch 32 — Llama-Style Model**
- 32.1 Llama architecture changes, one by one: RMSNorm, SwiGLU, RoPE, no bias, GQA
- 32.2 Long context: RoPE scaling (position interpolation, NTK, YaRN)
- 32.3 Sliding-window and local/global attention
- 32.4 Upgrading the project model and comparing runs

**Ch 33 — Mixture of Experts**
- 33.1 Conditional computation: many specialists, few active
- 33.2 Router design: top-k gating
- 33.3 Load balancing losses and capacity factors
- 33.4 Implementing an MoE layer from scratch; fine-grained and shared experts

**Ch 34 — Beyond Attention**
- 34.1 Linear attention and the kernel trick
- 34.2 State-space models: S4 intuition and Mamba's selective scan
- 34.3 Hybrid architectures
- 34.4 Multimodal basics: vision encoders, patches as tokens, projectors

### Part 9 — Evaluation (16 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 35 | Perplexity, Benchmarks & Contamination | 8 | 3,800 |
| 36 | Custom Evals & LLM-as-Judge | 8 | 3,800 |

**Ch 35 — Benchmarks**
- 35.1 Perplexity done right (bits-per-byte across tokenizers)
- 35.2 Multiple-choice evaluation by log-likelihood (MMLU, HellaSwag)
- 35.3 Generative evaluation (GSM8K, HumanEval, pass@k)
- 35.4 Contamination detection with n-gram overlap
- 35.5 What the numbers actually mean: variance, error bars, saturation

**Ch 36 — Custom Evals**
- 36.1 Designing evals for your domain
- 36.2 LLM-as-judge: rubrics, pairwise comparison, biases
- 36.3 Building an eval harness for the project

### Part 10 — Post-Training (50 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 37 | Supervised Fine-Tuning & Chat Templates | 10 | 4,800 |
| 38 | LoRA & QLoRA From Scratch | 8 | 3,800 |
| 39 | Reward Models & RLHF with PPO | 10 | 4,800 |
| 40 | Direct Preference Optimization & Friends | 8 | 3,800 |
| 41 | GRPO & RL for Reasoning | 8 | 3,800 |
| 42 | Synthetic Data, Distillation & Safety | 6 | 2,900 |

**Ch 37 — SFT**
- 37.1 From a document completer to an assistant
- 37.2 Chat templates and loss masking on prompt tokens
- 37.3 Creating instruction data (manual, templated, self-instruct)
- 37.4 Full fine-tuning the project model; catastrophic forgetting

**Ch 38 — LoRA & QLoRA**
- 38.1 Low-rank updates: why fine-tuning changes live in a small subspace
- 38.2 A LoRA layer from scratch; rank, alpha, target modules
- 38.3 Merging adapters
- 38.4 QLoRA: 4-bit NF4 base plus LoRA; DoRA preview

**Ch 39 — Reward Models & PPO**
- 39.1 Preference data and the Bradley–Terry model
- 39.2 Training a reward model
- 39.3 Policy gradients from scratch (REINFORCE)
- 39.4 PPO: clipped objective, value head, GAE, KL penalty
- 39.5 A minimal RLHF loop; reward hacking

**Ch 40 — DPO & Friends**
- 40.1 Deriving DPO from the RLHF objective
- 40.2 Implementing DPO
- 40.3 IPO, KTO, ORPO, SimPO: what each fixes

**Ch 41 — GRPO & Reasoning**
- 41.1 Verifiable rewards (math, code)
- 41.2 GRPO: group-relative advantages without a critic
- 41.3 Rejection sampling / best-of-n fine-tuning
- 41.4 Chain-of-thought, test-time compute and training for reasoning

**Ch 42 — Synthetic Data & Safety**
- 42.1 Synthetic data generation and filtering
- 42.2 Knowledge distillation (logit and sequence-level)
- 42.3 Constitutional-style alignment (critique and revise)
- 42.4 Safety basics: red-teaming, refusals, jailbreak robustness

### Part 11 — Inference & Deployment (28 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 43 | Fast Decoding & Speculative Decoding | 8 | 3,800 |
| 44 | Quantization From Scratch | 10 | 4,800 |
| 45 | Serving at Scale | 10 | 4,800 |

**Ch 43 — Fast Decoding**
- 43.1 Prefill vs decode: two different workloads
- 43.2 Speculative decoding with a draft model, and its correctness proof
- 43.3 Medusa / lookahead variants; structured and constrained generation

**Ch 44 — Quantization**
- 44.1 Quantization as rounding onto a grid: scale and zero-point
- 44.2 INT8 per-tensor, per-channel and group-wise from scratch
- 44.3 Outliers and LLM.int8() / SmoothQuant intuition
- 44.4 INT4 with GPTQ (second-order error correction) and AWQ (activation-aware scaling)
- 44.5 Measuring the quality/speed/memory trade-off; GGUF formats

**Ch 45 — Serving**
- 45.1 Static vs continuous batching
- 45.2 PagedAttention: virtual memory for the KV cache
- 45.3 Latency (TTFT, TPOT) vs throughput
- 45.4 A mini serving engine with FastAPI and streaming
- 45.5 Cost estimation: $/million tokens on your own hardware vs cloud

### Part 12 — Applications & Beyond (28 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 46 | Retrieval-Augmented Generation | 7 | 3,400 |
| 47 | Tool Use, Agents & Long-Context Tricks | 7 | 3,400 |
| 48 | Interpretability: Looking Inside | 9 | 4,300 |
| 49 | Reading Research, Reproducing Papers & the Road to the Frontier | 5 | 2,400 |

**Ch 46 — RAG**
- 46.1 Why models need an external memory
- 46.2 Embedding models, chunking and vector search from scratch
- 46.3 Hybrid retrieval (BM25 + dense) and reranking
- 46.4 Evaluating RAG

**Ch 47 — Tools, Agents & Long Context**
- 47.1 Function-calling formats and training a model to use tools
- 47.2 The agent loop: plan, act, observe
- 47.3 Long-context tricks: context compression, attention sinks, memory

**Ch 48 — Interpretability**
- 48.1 Attention visualization and its limits
- 48.2 Logit lens and linear probes
- 48.3 Induction heads and circuits
- 48.4 Sparse autoencoders for features, from scratch
- 48.5 Activation steering

**Ch 49 — Research Skills & Roadmap**
- 49.1 How to read a paper in three passes
- 49.2 Reproducing a paper: a checklist
- 49.3 Open problems and a 12-month roadmap to the frontier

### Capstone (16 pages)

| Ch | Title | Pages | Words |
|---|---|---|---|
| 50 | Capstone: Your Own LLM, End to End | 16 | 7,700 |

- 50.1 Choosing a dataset and a compute budget
- 50.2 Data → tokenizer → pretrain (with scaling-law planning)
- 50.3 Eval → SFT → preference tuning
- 50.4 Quantize → serve → chat UI
- 50.5 Write-up template and **grading rubric** (Beginner / Intermediate / Advanced / Expert)

### Appendices (14 pages)

| App | Title | Pages |
|---|---|---|
| A | Glossary (250+ terms) | 5 |
| B | Papers Reading List, in chronological order (1948–2026) | 3 |
| C | Debugging Cheat Sheet (symptom → cause → fix) | 3 |
| D | Math & Shape Reference Card | 3 |

**Budget check:** 46 + 54 + 38 + 26 + 46 + 30 + 36 + 34 + 30 + 16 + 50 + 28 + 28 + 16 + 14 = **492 pages**

---

## 2. The Project Thread: `nanoforge`

You build **one repository, `nanoforge`**, that grows from a 20-line counter into a deployable, aligned chat model trained on your own data. Nothing gets thrown away: each chapter adds a module or upgrades one.

**Default datasets** (swap in your own at any point):
- *Learning corpus:* Tiny Shakespeare (Ch 11–20), then TinyStories (Ch 20–27)
- *Your corpus:* pick one in Ch 21. Examples: your field's papers, a codebase, a legal/medical corpus, a non-English language, your notes. You carry it to the Capstone.

| Milestone | After Ch | What you have | Key file(s) |
|---|---|---|---|
| M0 | 1 | Toy autocomplete, repo scaffold | `scripts/toy.py` |
| M1 | 2–5 | Math utilities and tests (softmax, CE, KL, gradcheck) | `nanoforge/mathutils.py` |
| M2 | 6 | Scalar autograd engine `forge.grad` | `nanoforge/autograd.py` |
| M3 | 7–10 | MLP, from-scratch optimizers, norms, init | `nanoforge/nn.py`, `optim.py` |
| M4 | 11 | Bigram LM (count and neural) | `models/bigram.py` |
| M5 | 12–13 | MLP LM + universal `generate()` | `models/mlp_lm.py`, `sampling.py` |
| M6 | 14 | Char-RNN / LSTM baseline | `models/rnn.py` |
| M7 | 15–16 | Your own byte-level BPE tokenizer | `nanoforge/tokenizer/` |
| M8 | 17–20 | **GPT from scratch**, loads GPT-2 weights | `models/gpt.py` |
| M9 | 21–23 | Data pipeline for any corpus (dedup, filter, shard, pack) | `data/` |
| M10 | 24–27 | Production training loop, scaling-law planner | `train.py`, `scaling.py` |
| M11 | 28–31 | Efficient + multi-GPU training (AMP, compile, FSDP, KV cache) | `train.py`, `models/gpt.py` |
| M12 | 32–34 | **Llama-style model** + optional MoE layer | `models/llama.py`, `moe.py` |
| M13 | 35–36 | Eval harness (ppl, MC, generative, judge) | `eval/` |
| M14 | 37–42 | SFT → LoRA → DPO/GRPO → an aligned assistant | `posttrain/` |
| M15 | 43–45 | Quantized model + streaming server | `serve/` |
| M16 | 46–48 | RAG over your corpus, tool use, interpretability notebooks | `apps/`, `interp/` |
| **M17** | **50** | **Your own LLM, deployed with a chat UI** | everything |

**Size ladder** (all trainable on one consumer GPU or free Colab):

| Model | Params | Train time (RTX 3060–4090 / T4) | Used in |
|---|---|---|---|
| nano | 0.8M | minutes | Ch 20 |
| micro | 10M | ~30 min | Ch 24–27 |
| mini | 50M | a few hours | Ch 32, 37 |
| small | 125M (GPT-2 size) | ~1 day on a 4090 | Capstone (optional) |

---

## 3. Environment Setup

### 3.1 Options

| Option | Best for | Notes |
|---|---|---|
| Local NVIDIA GPU (≥ 8 GB VRAM) | Everything up to "mini" | Best experience |
| **Windows** | Parts 0–6 work natively | For `torch.compile`, Triton and multi-GPU (Ch 29, 31), use **WSL2 + Ubuntu** |
| Google Colab (T4 free / L4 / A100 paid) | No GPU at home | Mount Drive for checkpoints |
| CPU only | Parts 0–3 and nano models | Slow but fine for learning |
| Rented cloud GPUs | Ch 31 multi-GPU, Capstone "small" | A few dollars per hour |

### 3.2 Install (pinned versions)

```bash
# Python 3.11 recommended
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS/WSL: source .venv/bin/activate
pip install --upgrade pip
# PyTorch with CUDA 12.4 (see pytorch.org for other CUDA versions / CPU)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt`:
```text
numpy==1.26.4
matplotlib==3.9.2
tqdm==4.66.5
regex==2024.9.11        # GPT-style pre-tokenization (Ch 15)
jupyterlab==4.2.5
pytest==8.3.3
datasets==3.0.1         # ONLY for downloading raw data, never for processing
tiktoken==0.8.0         # ONLY for comparing against our tokenizer
transformers==4.45.2    # ONLY for loading GPT-2 weights to verify our GPT (Ch 20)
pypdf==5.0.1            # custom data extraction (Ch 23)
pyarrow==17.0.0         # Parquet (Ch 23)
fastapi==0.115.0        # serving (Ch 45)
uvicorn==0.30.6
scikit-learn==1.5.2     # PCA/t-SNE plots and probes only
```

> **Rule of the book:** Libraries may *fetch* data or *check* our results. They never do the core work for us.

### 3.3 Repository structure

```text
nanoforge/
├── README.md
├── requirements.txt
├── configs/                 # YAML/py configs for every model size
│   ├── nano.py  micro.py  mini.py  small.py
├── nanoforge/               # the library you write
│   ├── __init__.py
│   ├── mathutils.py         # Ch 2–5
│   ├── autograd.py          # Ch 6
│   ├── nn.py                # Ch 7, 9, 10
│   ├── optim.py             # Ch 8
│   ├── sampling.py          # Ch 13
│   ├── tokenizer/           # Ch 15–16 (bpe.py, bytelevel.py, unigram.py)
│   ├── models/              # bigram.py, mlp_lm.py, rnn.py, gpt.py, llama.py, moe.py
│   ├── data/                # Ch 21–23 (clean.py, dedup.py, filters.py, shard.py, loader.py)
│   ├── train.py             # Ch 24–31
│   ├── scaling.py           # Ch 26
│   ├── eval/                # Ch 35–36
│   ├── posttrain/           # Ch 37–42 (sft.py, lora.py, reward.py, ppo.py, dpo.py, grpo.py)
│   ├── quant/               # Ch 44
│   ├── serve/               # Ch 43, 45
│   ├── apps/                # Ch 46–47 (rag.py, agent.py)
│   └── interp/              # Ch 48
├── scripts/                 # prepare_data.py, train_tokenizer.py, chat.py ...
├── notebooks/               # one per chapter: ch01.ipynb ... ch50.ipynb
├── tests/                   # pytest: gradcheck, shape tests, tokenizer round-trip
├── data/                    # raw/ and processed/ (git-ignored)
└── checkpoints/             # git-ignored
```

**Conventions used throughout the book:**
- Shapes are always commented: `B` = batch, `T` = time/sequence, `C` = channels (d_model), `H` = heads, `D` = head dim, `V` = vocab
- `cfg` = config object; `model`, `opt`, `sched` are named the same everywhere
- Every module ships with a `tests/test_<module>.py`

### 3.4 Verify your setup

`scripts/check_env.py`:
```python
import sys, platform
import numpy as np
import torch

print(f"Python   : {sys.version.split()[0]}  ({platform.system()})")
print(f"NumPy    : {np.__version__}")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU      : {p.name}  |  VRAM {p.total_memory / 1e9:.1f} GB")
    print(f"BF16     : {torch.cuda.is_bf16_supported()}")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Smoke test: a matmul and a gradient
x = torch.randn(64, 128, device=device, requires_grad=True)   # (B, C)
w = torch.randn(128, 256, device=device)                       # (C, 4C)
y = (x @ w).relu().sum()                                        # scalar
y.backward()
print(f"Device   : {device}  |  grad shape {tuple(x.grad.shape)}  ✅ ready")
```

```bash
python scripts/check_env.py
```

### 3.5 Colab quickstart

```python
!git clone https://github.com/<you>/nanoforge.git && cd nanoforge && pip install -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')   # save checkpoints to Drive
```

---

**Ready.** Say **"Write Chapter 1"** to begin.

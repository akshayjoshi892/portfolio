# Appendix B — Papers Reading List, in Chronological Order (1948–2026)

> **How to use this list:** read it as a history of the ideas in this book. Each entry says in a few words why it matters and which chapter builds on it. Entries marked ★ are the core set: if you only read 25 papers, read those. Years follow the version most people cite (conference or journal year where that's standard, otherwise arXiv). The list is dense in 2017–2025 because the field is; it ends with an open section for you to fill in as you read (Ch 49.1).

---

## Foundations (1948–2012)

| Year | Paper | Why it matters | Ch |
|---|---|---|---|
| 1948 | ★ Shannon, "A Mathematical Theory of Communication" | Entropy, coding, and n-gram models of English: the origin of "language modeling = prediction" | 5, 11 |
| 1951 | Shannon, "Prediction and Entropy of Printed English" | Humans as next-letter predictors; estimating the entropy of English | 5.7 |
| 1951 | Robbins & Monro, "A Stochastic Approximation Method" | The mathematical root of SGD | 8.2 |
| 1958 | Rosenblatt, "The Perceptron" | The first trainable artificial neuron | 7.1 |
| 1964 | Polyak, "Some methods of speeding up the convergence of iteration methods" | Momentum (the heavy ball) | 8.3 |
| 1986 | ★ Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" | Backpropagation for multi-layer networks | 6 |
| 1989 | Cybenko, "Approximation by superpositions of a sigmoidal function" | Universal approximation | 7.2 |
| 1990 | Elman, "Finding Structure in Time" | Simple recurrent networks for sequences | 14.1 |
| 1992 | Williams, "Simple statistical gradient-following algorithms for connectionist RL" | REINFORCE, the basis of policy gradients | 39.3 |
| 1993 | Hassibi & Stork, "Optimal Brain Surgeon" | Second-order weight removal; the ancestor of GPTQ | 44.4 |
| 1994 | Bengio, Simard & Frasconi, "Learning long-term dependencies with gradient descent is difficult" | Why vanilla RNNs forget | 14.3 |
| 1994 | Gage, "A New Algorithm for Data Compression" | Byte-pair encoding, originally for compression | 15.3 |
| 1997 | ★ Hochreiter & Schmidhuber, "Long Short-Term Memory" | Gated memory cells | 14.4 |
| 1997 | Broder, "On the Resemblance and Containment of Documents" | MinHash for near-duplicate detection | 21.5 |
| 2003 | ★ Bengio et al., "A Neural Probabilistic Language Model" | Learned embeddings + a neural LM | 12 |
| 2009 | Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" | BM25, still a strong retrieval baseline | 46.3 |
| 2009 | Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion…" | Simple, robust fusion of rankings | 46.3 |
| 2010 | Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" | Xavier initialization | 9.2 |
| 2012 | Tieleman & Hinton, RMSProp (Coursera lecture 6.5) | Per-parameter adaptive step sizes | 8.4 |

## Deep learning takes off (2013–2016)

| Year | Paper | Why it matters | Ch |
|---|---|---|---|
| 2013 | Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" | word2vec: embeddings with geometric meaning | 12.2 |
| 2014 | Sutskever, Vinyals & Le, "Sequence to Sequence Learning with Neural Networks" | Encoder–decoder LSTMs | 14 |
| 2014 | Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder" | The GRU | 14.4 |
| 2014 | Srivastava et al., "Dropout" | Regularization by random masking | 10.4 |
| 2015 | ★ Bahdanau, Cho & Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" | Attention is born | 17 |
| 2015 | ★ Kingma & Ba, "Adam: A Method for Stochastic Optimization" | The default optimizer | 8.5 |
| 2015 | Ioffe & Szegedy, "Batch Normalization" | Normalization as a training accelerator | 10.1 |
| 2015 | He et al., "Delving Deep into Rectifiers" | Kaiming initialization | 9.2 |
| 2015 | Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" | Soft-label distillation | 42.2 |
| 2016 | ★ He et al., "Deep Residual Learning for Image Recognition" | Residual connections | 10.3 |
| 2016 | ★ Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with Subword Units" | BPE for NLP | 15 |
| 2016 | Ba, Kiros & Hinton, "Layer Normalization" | Per-example normalization for sequences | 10.2 |
| 2016 | Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" | The GELU activation | 9.3 |
| 2016 | Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" | GAE | 39.4 |
| 2016 | Chen et al., "Training Deep Nets with Sublinear Memory Cost" | Activation checkpointing | 28.4 |
| 2016 | Kim & Rush, "Sequence-Level Knowledge Distillation" | Distillation for generation | 42.2 |
| 2016 | Alain & Bengio, "Understanding intermediate layers using linear classifier probes" | Linear probes | 48.2 |
| 2017 | Press & Wolf, "Using the Output Embedding to Improve Language Models" | Weight tying | 19.5 |

## The transformer era begins (2017–2019)

| Year | Paper | Why it matters | Ch |
|---|---|---|---|
| 2017 | ★ Vaswani et al., "Attention Is All You Need" | The Transformer | 17–20 |
| 2017 | ★ Schulman et al., "Proximal Policy Optimization Algorithms" | PPO | 39.4 |
| 2017 | Christiano et al., "Deep Reinforcement Learning from Human Preferences" | Learning rewards from comparisons | 39 |
| 2017 | Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" | Modern MoE | 33 |
| 2017 | Hestness et al., "Deep Learning Scaling is Predictable, Empirically" | Early power-law scaling | 26.1 |
| 2018 | Micikevicius et al., "Mixed Precision Training" | fp16 training with loss scaling | 25.2 |
| 2018 | Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1) | Pretrain then fine-tune a decoder | 20 |
| 2018 | Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" | Affine quantization with zero-points | 44.1 |
| 2018 | Kudo, "Subword Regularization" and Kudo & Richardson, "SentencePiece" | The Unigram tokenizer and its library | 16.2 |
| 2018 | McCandlish et al., "An Empirical Model of Large-Batch Training" | Gradient noise scale and critical batch size | 26.5 |
| 2018 | Oord, Li & Vinyals, "Representation Learning with Contrastive Predictive Coding" | InfoNCE | 46.2 |
| 2018 | Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs" | Graph-based vector search | 46.2 |
| 2019 | Devlin et al., "BERT" | Bidirectional pretraining (the road not taken for generation) | 1 |
| 2019 | ★ Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2) | Scale + a decoder = zero-shot abilities; the model you load in Ch 20 | 20 |
| 2019 | Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" | Multi-query attention | 30.3 |
| 2019 | Shoeybi et al., "Megatron-LM" | Tensor parallelism | 31.3 |
| 2019 | Huang et al., "GPipe" | Pipeline parallelism | 31.4 |
| 2019 | Zhang & Sennrich, "Root Mean Square Layer Normalization" | RMSNorm | 10.2 |
| 2019 | Belkin et al., "Reconciling modern machine-learning practice and the classical bias–variance trade-off" | Double descent | 10.5 |
| 2019 | Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" | Probe selectivity | 48.2 |
| 2019 | Clark et al., "What Does BERT Look At?" | Attention head taxonomy | 48.1 |
| 2019 | Jain & Wallace, "Attention is not Explanation" | Limits of attention as explanation | 48.1 |

## Scaling up (2020–2021)

| Year | Paper | Why it matters | Ch |
|---|---|---|---|
| 2020 | ★ Kaplan et al., "Scaling Laws for Neural Language Models" | Loss as a power law of N, D, C | 26 |
| 2020 | ★ Brown et al., "Language Models are Few-Shot Learners" (GPT-3) | In-context learning at scale | 1, 26 |
| 2020 | Holtzman et al., "The Curious Case of Neural Text Degeneration" | Nucleus (top-p) sampling | 13.3 |
| 2020 | Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" | Sharded optimizer states, gradients, parameters | 31.2 |
| 2020 | Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5) | C4 and its cleaning rules | 21–22 |
| 2020 | Shazeer, "GLU Variants Improve Transformer" | SwiGLU | 9.4 |
| 2020 | Xiong et al., "On Layer Normalization in the Transformer Architecture" | Pre-norm vs post-norm | 19.2 |
| 2020 | ★ Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" | RAG | 46 |
| 2020 | Karpukhin et al., "Dense Passage Retrieval" | Dual encoders with in-batch negatives | 46.2 |
| 2020 | Guu et al., "REALM" | Retrieval during pretraining | 46.1 |
| 2020 | Stiennon et al., "Learning to Summarize from Human Feedback" | RLHF on language | 39 |
| 2020 | Katharopoulos et al., "Transformers are RNNs" | Linear attention | 34.1 |
| 2020 | Khattab & Zaharia, "ColBERT" | Late-interaction retrieval | 46.3 |
| 2020 | Gururangan et al., "Don't Stop Pretraining" | Domain-adaptive continued pretraining | 50.1 |
| 2020 | Gao et al., "The Pile" | A documented, diverse open corpus | 21.1 |
| 2020 | nostalgebraist, "interpreting GPT: the logit lens" (blog) | Reading intermediate layers | 48.2 |
| 2021 | ★ Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" | RoPE | 18.5 |
| 2021 | ★ Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" | Parameter-efficient fine-tuning | 38 |
| 2021 | Hendrycks et al., "Measuring Massive Multitask Language Understanding" | MMLU | 35.2 |
| 2021 | Chen et al., "Evaluating Large Language Models Trained on Code" | HumanEval and unbiased pass@k | 35.3 |
| 2021 | Cobbe et al., "Training Verifiers to Solve Math Word Problems" | GSM8K; verifiers | 35.3, 41 |
| 2021 | ★ Elhage et al., "A Mathematical Framework for Transformer Circuits" | QK/OV circuits, residual stream view | 19.4, 48 |
| 2021 | Fedus, Zoph & Shazeer, "Switch Transformers" | Top-1 MoE routing at scale | 33 |
| 2021 | Rae et al., "Scaling Language Models: Methods, Analysis & Insights from Training Gopher" | Gopher quality filters | 22.1 |
| 2021 | Lee et al., "Deduplicating Training Data Makes Language Models Better" | Dedup effects on memorization and quality | 21.5 |
| 2021 | Narang et al., "Do Transformer Modifications Transfer Across Implementations and Applications?" | Most tweaks don't; a lesson in rigour | 49.2 |
| 2021 | Thakur et al., "BEIR" | Zero-shot retrieval benchmark | 46.3 |

## Alignment, efficiency and open models (2022–2023)

| Year | Paper | Why it matters | Ch |
|---|---|---|---|
| 2022 | ★ Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla) | ~20 tokens per parameter | 26.2 |
| 2022 | ★ Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT) | SFT → RM → PPO | 37, 39 |
| 2022 | ★ Dao et al., "FlashAttention" | IO-aware exact attention | 29.2 |
| 2022 | ★ Olsson et al., "In-context Learning and Induction Heads" | The copy mechanism and its phase change | 48.3 |
| 2022 | Press, Smith & Lewis, "Train Short, Test Long" | ALiBi | 18.6 |
| 2022 | Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" | µP | 26.6 |
| 2022 | Bai et al., "Constitutional AI: Harmlessness from AI Feedback" | Principles + AI feedback | 42.3 |
| 2022 | Bai et al., "Training a Helpful and Harmless Assistant with RLHF" | Helpfulness/harmlessness trade-offs | 39 |
| 2022 | Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" | Reasoning by writing it out | 41.4 |
| 2022 | Dettmers et al., "LLM.int8()" | Outlier features; mixed-precision INT8 | 44.3 |
| 2022 | Elhage et al., "Toy Models of Superposition" | Why neurons are polysemantic | 48.4 |
| 2022 | Meng et al., "Locating and Editing Factual Associations in GPT" (ROME) | Causal tracing of facts | 48.3 |
| 2022 | Pope et al., "Efficiently Scaling Transformer Inference" | Prefill/decode cost analysis | 43.1 |
| 2022 | Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" | Continuous batching | 45.1 |
| 2022 | Perez et al., "Red Teaming Language Models with Language Models" | Automated red-teaming | 42.4 |
| 2022 | Wang et al., "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5) | Modern embedding recipe | 46.2 |
| 2023 | ★ Touvron et al., "LLaMA" and "Llama 2" | Open, efficient decoder recipe (RMSNorm, SwiGLU, RoPE, GQA) | 32 |
| 2023 | ★ Rafailov et al., "Direct Preference Optimization" | RLHF without RL | 40 |
| 2023 | ★ Frantar et al., "GPTQ" | Accurate one-shot 4-bit quantization | 44.4 |
| 2023 | ★ Leviathan, Kalman & Matias, "Fast Inference from Transformers via Speculative Decoding" (and Chen et al., "Accelerating LLM Decoding with Speculative Sampling") | Exact speed-ups from drafting | 43.2 |
| 2023 | ★ Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" | vLLM | 45.2 |
| 2023 | Xiao et al., "SmoothQuant" | Migrating outliers for W8A8 | 44.3 |
| 2023 | Dettmers et al., "QLoRA" | NF4 + LoRA fine-tuning on one GPU | 38.4 |
| 2023 | Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" | Grouped-query attention | 30.3 |
| 2023 | Dao, "FlashAttention-2" | Better parallelism and work partitioning | 29.2 |
| 2023 | Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" | Selective SSMs | 34.2 |
| 2023 | Chen et al., "Extending Context Window of LLMs via Positional Interpolation" and Peng et al., "YaRN" | RoPE context extension | 32.2 |
| 2023 | Jiang et al., "Mistral 7B" | Sliding-window attention + GQA in a strong small model | 32.3 |
| 2023 | Lightman et al., "Let's Verify Step by Step" | Process supervision for reasoning | 41 |
| 2023 | Muennighoff et al., "Scaling Data-Constrained Language Models" | Returns from repeated epochs | 27.4, 50.1 |
| 2023 | Wang et al., "Self-Instruct" | Bootstrapping instruction data | 37.3 |
| 2023 | Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" | Judge biases and pairwise evaluation | 36.2 |
| 2023 | Gunasekar et al., "Textbooks Are All You Need" | Synthetic "textbook" data | 42.1 |
| 2023 | Schick et al., "Toolformer" | Self-taught tool use | 47.1 |
| 2023 | Yao et al., "ReAct" | Reasoning + acting agents | 47.2 |
| 2023 | Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG) | Adversarial suffixes | 42.4 |
| 2023 | Greshake et al., "Not what you've signed up for" | Indirect prompt injection | 42.4, 47.2 |
| 2023 | Bricken et al., "Towards Monosemanticity" | SAEs on a real transformer | 48.4 |
| 2023 | Wang et al., "Interpretability in the Wild" (IOI circuit) | A full circuit in GPT-2 small | 48.3 |
| 2023 | Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" | IPO | 40.3 |
| 2023 | Tunstall et al., "Zephyr: Direct Distillation of LM Alignment" | dSFT + DPO for small models | 50.3 |
| 2023 | Li et al., "Inference-Time Intervention" and Turner et al., "Activation Addition" | Steering with directions | 48.5 |
| 2023 | Willard & Louf, "Efficient Guided Generation for LLMs" (Outlines) | FSM-constrained decoding | 43.3 |
| 2023 | Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" | Tuned lens | 48.2 |
| 2023 | Lee et al., "RLAIF" | AI feedback vs human feedback | 42.3 |
| 2023 | Penedo et al., "The RefinedWeb Dataset for Falcon LLM" | Web-only data done well | 21–22 |

## Reasoning, serving and understanding (2024–2025)

| Year | Paper | Why it matters | Ch |
|---|---|---|---|
| 2024 | ★ Shao et al., "DeepSeekMath" | GRPO | 41.2 |
| 2024 | ★ Lin et al., "AWQ: Activation-aware Weight Quantization" | Robust 4-bit weights | 44.4 |
| 2024 | ★ Templeton et al., "Scaling Monosemanticity" | Millions of interpretable features in a production model | 48.4 |
| 2024 | Dubey et al., "The Llama 3 Herd of Models" | A complete modern pretraining + post-training report | 32, 50 |
| 2024 | Groeneveld et al., "OLMo: Accelerating the Science of Language Models" | Fully open data, code and checkpoints | 49.2, 50 |
| 2024 | Penedo et al., "The FineWeb Datasets" | Open web data with ablated filtering | 21–22 |
| 2024 | Hu et al., "MiniCPM" | WSD schedule analysis | 24.2 |
| 2024 | Ethayarajh et al., "KTO"; Hong et al., "ORPO"; Meng et al., "SimPO" | The DPO family | 40.3 |
| 2024 | Cai et al., "Medusa"; Li et al., "EAGLE" | Self-speculation and feature-level drafting | 43.2–43.3 |
| 2024 | Xiao et al., "Efficient Streaming Language Models with Attention Sinks" | StreamingLLM | 47.3 |
| 2024 | Liu et al., "Lost in the Middle" | Position effects in long contexts | 46.4 |
| 2024 | Asai et al., "Self-RAG" | Learning when to retrieve | 46.1 |
| 2024 | Es et al., "RAGAS" | Reference-free RAG evaluation | 46.4 |
| 2024 | Agrawal et al., "Sarathi-Serve"; Zhong et al., "DistServe" | Chunked prefill; prefill/decode disaggregation | 43.1, 45.3 |
| 2024 | Zheng et al., "SGLang" | RadixAttention prefix reuse | 45.2 |
| 2024 | Dai et al., "DeepSeekMoE" | Fine-grained and shared experts | 33.4 |
| 2024 | DeepSeek-AI, "DeepSeek-V3 Technical Report" | Large MoE, multi-token prediction, FP8 training | 25, 33, 43.3 |
| 2024 | Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" | Extra future-token heads | 43.3 |
| 2024 | Sun et al., "Massive Activations in Large Language Models" | Huge scalar activations acting as biases | 44.3, 47.3 |
| 2024 | Ashkboos et al., "QuaRot" | Rotations for 4-bit weights and activations | 44.3 |
| 2024 | Gao et al., "Scaling and Evaluating Sparse Autoencoders" | TopK SAEs | 48.4 |
| 2024 | Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" | Directional ablation | 48.5 |
| 2024 | Shumailov et al., "AI models collapse when trained on recursively generated data" | Model collapse | 42.1 |
| 2024 | Wallace et al., "The Instruction Hierarchy" | Prioritizing trusted instructions | 42.4 |
| 2024 | Jimenez et al., "SWE-bench" | Realistic agent evaluation | 47.2 |
| 2024 | Lambert et al., "Tülu 3" | Fully open post-training with RLVR | 41, 50.3 |
| 2024 | Jordan et al., "Muon" (technical blog) | Orthogonalized momentum | 8.8 |
| 2025 | ★ DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" | Reasoning from RL with verifiable rewards | 41 |
| 2025 | Muennighoff et al., "s1: Simple test-time scaling" | Small curated reasoning data + budget forcing | 41.4 |
| 2025 | Team OLMo, "2 OLMo 2 Furious" | Open recipe with stability fixes (QK-norm, reordered norms) | 27.3, 32 |
| 2025 | Allal et al., "SmolLM2" | Data-centric training of small models | 22, 50 |
| 2025 | Yu et al., "DAPO"; Liu et al., "Understanding R1-Zero-Like Training" (Dr. GRPO) | Fixes and variants of GRPO | 41.2 |
| 2025 | Ameisen et al., "Circuit Tracing" and Lindsey et al., "On the Biology of a Large Language Model" | Attribution graphs with transcoders | 48.4 |

## 2026 and beyond: your additions

This book stops where the field keeps going. Keep adding rows here as you read (Ch 49.1): for each paper, write the one-sentence idea in your own words, the chapter it extends, and whether you reproduced any part of it.

| Year | Paper | Your one-sentence summary | Extends Ch | Reproduced? |
|---|---|---|---|---|
| 2026 | | | | |
| 2026 | | | | |
| 2026 | | | | |

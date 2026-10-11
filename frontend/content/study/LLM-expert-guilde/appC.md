# Appendix C — Debugging Cheat Sheet (Symptom → Cause → Fix)

> **How to use this sheet:** find the symptom you're seeing, check the likely causes in order (most common first), and apply the fix. The chapter column points to the full explanation. Two habits prevent most of these bugs in the first place: **print shapes** at every stage, and **overfit a single batch** before training for real (the loss should go to ~0 in a few hundred steps; if it doesn't, the bug is in the model or the loss, not the data or the hyperparameters).

---

## 1. First-minute sanity checks

| Check | Expected | If not |
|---|---|---|
| Loss at step 0 | ≈ ln V (e.g. ln 8192 ≈ 9.01) | Init too large (logits not near zero), wrong loss reduction, labels misaligned (Ch 7.3, 20.3) |
| Overfit one batch | loss → ~0 within a few hundred steps | Bug in model/loss/optimizer: gradients not flowing, LR ≈ 0, labels shifted wrong (Ch 24.1) |
| Causal leak test | Changing token t+1 doesn't change logits at ≤ t | Mask missing or wrong; `is_causal` not set; packed documents leaking (Ch 17.4, 23.4) |
| Gradient check (small model, fp64) | Relative error < 1e-5 | Wrong backward for a custom op (Ch 4.5, 6.6) |
| Param count | Matches your hand calculation | Duplicated or untied weights, wrong FFN width (Ch 19.6) |
| Cached vs uncached generation | Identical greedy outputs | KV cache position or RoPE offset bug (Ch 30.1) |

## 2. Shapes, tensors and code

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| `RuntimeError: shape mismatch` in matmul | Transposed operand; (B, T, C) vs (B, C, T) confusion | Comment shapes on every line; use `einsum` for clarity | 2.4 |
| Loss is fine but results are garbage | Silent broadcasting (e.g. (B,) with (B, 1) → (B, B)) | Assert shapes; avoid implicit broadcasting of vectors | 2.2 |
| `view size is not compatible with input tensor's size and stride` | `.view()` on a non-contiguous tensor (after `transpose` or slicing) | `.contiguous().view()` or `.reshape()` | 2.4 |
| Gradients accumulate across steps | Forgot `zero_grad()`; used `=` vs `+=` wrongly in custom autograd | `opt.zero_grad(set_to_none=True)` every step | 6.5 |
| `grad` is `None` for a parameter | Parameter not used in the forward pass, or detached (`.item()`, `.numpy()`, `with torch.no_grad()`) | Trace the graph; avoid detaching inside the model | 6, 2.5 |
| Results differ run to run with the same seed | Non-deterministic kernels; dataloader workers seeded identically or not at all | Seed all RNGs + workers; `torch.use_deterministic_algorithms(True)` for debugging | 2.6 |
| Model works in training, breaks in `eval()` | Dropout/norm behaving differently; `model.eval()` never called (or never undone) | Call `eval()`/`train()` explicitly around evaluation | 10.4 |
| NaN immediately | log(0), division by zero, softmax of all −inf (fully masked row) | log-softmax, `clamp_min`, ensure each row has ≥ 1 valid key | 7.4, 17.4 |

## 3. Training dynamics

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| Loss stuck at ln V | LR far too small; gradients zero; labels random or misaligned | Check LR schedule values, grad norms, and that targets = inputs shifted by 1 | 24.1 |
| Loss decreases then plateaus high early | LR too low, or model too small for the data | LR range test; increase LR or model size | 12.4, 26.5 |
| Loss diverges in the first hundred steps | LR too high, no warmup | Add warmup; lower peak LR; check init scale | 24.2 |
| Sudden loss spike, then recovery | Bad batch, attention-logit growth, Adam ε/β2 issues | Gradient clipping, QK-norm, z-loss, skip batches with huge grad norm | 27.2–27.3 |
| Loss spike, no recovery (divergence) | Instability grew: logits exploded, fp16 overflow | Rewind to an earlier checkpoint, lower LR, add QK-norm/z-loss, use bf16 | 27.3 |
| Grad norm steadily growing | LR too high for the current phase; missing weight decay | Lower LR / decay earlier; check weight-decay groups | 8.6, 27.1 |
| Train loss ↓, val loss ↑ | Overfitting (small data, many epochs) | More data, dedup, dropout, weight decay, fewer epochs, early stopping | 10.5–10.6, 27.4 |
| Val loss suspiciously lower than train | Val leaked into train (near-duplicates), or dropout on during train only | Dedup across splits; compare with dropout off | 21.5 |
| Loss curve has periodic bumps | Data not shuffled; shard boundaries; epoch restarts | Deterministic global shuffle; check sampler | 23.2 |
| Loss jumps after resume | Optimizer/scheduler/RNG/data position not restored | Save and restore everything; test resume in the pilot | 25.3–25.4 |
| Dead activations / many exact zeros | ReLU dying from large LR or bad init | Lower LR, Kaiming init, use GELU/SiLU | 9.3 |
| Activations explode with depth | Missing residual scaling; post-norm; wrong init | Pre-norm, scaled residual init (1/√(2L)) | 19.2, 20.3 |

## 4. Data and tokenizer

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| Model outputs boilerplate (cookie banners, menus) | Poor HTML extraction; missing quality filters | Better boilerplate removal; Gopher/C4-style filters | 21.3, 22.1 |
| Model repeats memorized passages | Duplicates in the corpus | Exact + near-dedup (MinHash-LSH) | 21.5 |
| Round-trip `decode(encode(x)) != x` | Normalization in the tokenizer; wrong byte handling; special tokens parsed from text | Byte-level BPE; test round-trip on diverse Unicode; `encode_ordinary` for user text | 15.2, 16.1 |
| Many tokens per word in your domain | Tokenizer trained on different text | Train the tokenizer on your corpus; measure bytes/token | 16.7 |
| Arithmetic or spacing weirdness | Digit splitting and leading-space conventions | Split digits individually; consistent space handling | 16.6 |
| Special tokens appear in outputs or inputs are hijacked | Special tokens injected from raw text | Encode user/tool content as ordinary text only | 16.3, 37.2 |
| Embedding rows never trained | Vocabulary entries never seen in training | Check unused tokens; retrain tokenizer or resize | 16.6 |
| Dataloader is the bottleneck (GPU util low) | Tokenizing on the fly; too few workers; slow storage | Pre-tokenize to memmap shards; more workers; pin memory | 23.1–23.3 |

## 5. Precision, memory and speed

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| CUDA out of memory at start | Batch/sequence too large; optimizer states in fp32 | Smaller micro-batch + accumulation; activation checkpointing; memory math first | 28.3–28.4 |
| OOM after many steps | Memory leak: storing tensors with graphs (e.g. `losses.append(loss)`) | Store `loss.item()`; check for growing lists of tensors | 24.5 |
| fp16 loss becomes NaN/inf | Overflow; no loss scaling | Use bf16, or GradScaler with fp16 | 25.2 |
| bf16 training slightly worse than fp32 | Accumulations or softmax in bf16 | Keep norms, softmax, loss and optimizer states in fp32 | 25.1–25.2 |
| Low MFU (< 20%) | Small matmuls, Python overhead, dataloader stalls, unfused ops | Larger micro-batch, `torch.compile`, FlashAttention, profile | 24.5, 29 |
| `torch.compile` recompiles constantly | Changing shapes (variable sequence lengths) | Pad to fixed buckets; mark dynamic dims | 29.3 |
| Timing numbers inconsistent | Missing `torch.cuda.synchronize()`; measuring warmup | Synchronize; warm up; average over many steps | 29.5, 43.1 |

## 6. Distributed training

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| Hangs at start or at a collective | Ranks disagree on the number of collectives (conditional code paths); wrong `MASTER_ADDR/PORT` | Same control flow on all ranks; check env; timeouts with `NCCL_DEBUG=INFO` | 31.1 |
| Each rank sees the same data | Sampler not sharded by rank | Pass rank/world_size to the sampler | 23.3, 31.1 |
| Loss differs from single-GPU run | Gradient averaging vs summing; global batch changed | Keep global batch constant; check reduction | 31.1 |
| DDP "unused parameters" error | Conditional branches skip modules | Fix the model or `find_unused_parameters=True` (slower) | 31.1 |
| FSDP checkpoint can't be loaded | Sharded state dict saved per rank | Save a full state dict (or use distributed checkpointing consistently) | 31.2 |

## 7. Evaluation

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| Perplexities incomparable across models | Different tokenizers or context windows | Bits-per-byte with the same sliding windows | 35.1 |
| Multiple-choice accuracy near chance for a small model | Letter-scoring format too hard for small models | Cloze-style log-likelihood scoring | 35.2 |
| Big jump on a benchmark after new data | Contamination | n-gram overlap check against evals; verbatim-completion test | 35.4 |
| "Improvement" disappears on rerun | Within noise; single seed | CIs, paired bootstrap, more items/seeds | 35.5, 49.2 |
| LLM judge prefers the longer answer | Verbosity bias; position bias | Rubrics, swap order, length-controlled comparisons | 36.2 |
| Generative eval scores 0 for correct answers | Answer extraction regex too strict; stop tokens | Inspect raw outputs; robust parsing | 35.3, 36.1 |

## 8. Post-training

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| SFT model never stops generating | Missing EOS (`<\|im_end\|>`) in training targets; wrong `eos_id` at inference | Train on the end token; pass the right `eos_id` | 37.2 |
| SFT model repeats the prompt / role tags | Loss on prompt tokens; template mismatch | Mask non-assistant tokens; one template everywhere | 37.2 |
| Domain knowledge lost after SFT | Catastrophic forgetting (LR too high, too many epochs) | Lower LR, fewer epochs, replay data, LoRA | 37.4, 38 |
| LoRA has no effect | Wrong target module names; adapters not applied or not saved/merged | Print replaced layers; check trainable count | 38.2–38.3 |
| Merged LoRA model differs from adapter model | Dropout active when comparing; merge done twice; dtype rounding | Compare in `eval()`; merge once; merge in fp32 then cast | 38.3 |
| Reward model accuracy ~50% | Label noise, too few pairs, reward read at the wrong token | Clean pairs; read the score at the final token; more data | 39.2 |
| PPO reward rises, quality falls | Reward hacking | KL penalty, reward model ensembles, length penalties, read samples | 39.5 |
| DPO: chosen log-prob decreases along with rejected | Known DPO behaviour; β too small or LR too high | Tune β/LR; add SFT term (e.g. RPO/ORPO-style) | 40.2–40.3 |
| DPO makes answers much longer | Length bias in preference data | Length-normalized variants (SimPO), balance pair lengths | 40.3 |
| GRPO reward flat at 0 | Tasks too hard (no correct samples in the group) | Easier curriculum; more samples per prompt; check the verifier | 41.2 |
| GRPO reward rises, eval accuracy doesn't | Format hacking or verifier loopholes | Tighten the verifier; hold-out eval; inspect samples | 41.1 |

## 9. Inference and serving

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| Greedy output repeats a loop | Greedy decoding on a small model | Sampling with temperature/top-p; repetition penalty | 13.1, 13.4 |
| Cached generation differs from uncached | RoPE offset, cache write position, or mask off by one | Equality test on greedy outputs | 30.1 |
| Speculative decoding output differs from target | Different sampling transforms for p and q; cache rollback bug | Apply identical transforms; greedy-exactness test | 43.2 |
| Speculative decoding slower than baseline | Low acceptance (weak draft) or large k | Measure α; reduce k; better/distilled draft | 43.2 |
| Quantized model much worse | Per-tensor scales; outliers; quantized lm_head/embeddings; bad calibration data | Group-wise scales, SmoothQuant/AWQ/GPTQ, keep sensitive layers high precision | 44 |
| Quantized model not faster | Python dequantize-per-forward | Use fused kernels (llama.cpp, Marlin, bitsandbytes) | 44.2 |
| Server throughput low | Static batching; small max batch; KV memory exhausted | Continuous batching; paged KV; increase blocks | 45.1–45.2 |
| p99 TTFT explodes under load | Queueing near saturation; long prefills blocking decodes | Run below the knee; chunked prefill; more replicas | 45.3 |
| Streamed text shows `�` | Detokenizing tokens independently | Incremental detokenizer holding partial UTF-8 | 45.4 |
| Server memory grows until stall | KV blocks not freed on cancel/error | Release blocks on every exit path | 45.2 |

## 10. RAG, tools and agents

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| Right answer exists but isn't retrieved | Poor chunking; embedder untrained for the domain; identifiers blurred | Contrastive fine-tuning, hybrid BM25 + dense, chunk tuning | 46.2–46.3 |
| Retrieved but model ignores it | Model not trained to use context; best chunk buried in the middle | RAG-style SFT; put best chunks first | 46.1, 46.4 |
| Answers include unsupported claims | Model answers from memory | Faithfulness eval; SFT with "not in the sources" examples | 46.4 |
| Agent loops calling the same tool | No repeat detection; unclear tool results | Step budget, repeat detection, clearer errors | 47.2 |
| Tool calls fail to parse | Free-form JSON from a small model | Constrained decoding for the call; more tool SFT data | 43.3, 47.1 |
| Agent takes actions nobody asked for | Prompt injection from tool outputs/documents | Treat outputs as data; permission gates for side effects | 42.4, 47.2 |
| Long conversations degrade | Context overflow; naive truncation removed sinks or key facts | Summaries + pinned facts; keep sink tokens; external memory | 47.3 |

## 11. Interpretability

| Symptom | Likely cause | Fix | Ch |
|---|---|---|---|
| Attention plots don't match model behaviour | Recomputed attention without RoPE/GQA; sinks dominate | Recompute exactly (verify residuals match); value-weighted attention | 48.1 |
| Probe accuracy high everywhere, even on a random model | Probe memorizes token identity; class imbalance | Control tasks, random-model baseline, balanced accuracy, split by sequence | 48.2 |
| Ablating a "key" head does nothing | Redundant/backup heads; distributed mechanism | Group ablations; mean-ablation | 48.3 |
| SAE: good MSE, poor loss recovered | Reconstructs unimportant directions | Report loss recovered; tune λ; TopK/JumpReLU SAEs | 48.4 |
| Many dead SAE features | Large dictionary, strong L1 | Resample dead features; lower λ; TopK | 48.4 |
| Steering produces gibberish | α too large; wrong layer | Scale α by the residual norm; try middle layers | 48.5 |

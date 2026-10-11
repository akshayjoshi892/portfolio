# Appendix D — Math & Shape Reference Card

> Everything you'll want to look up while coding, on a few pages. Conventions match the book: **B** batch, **T** sequence length, **C** model width (d_model), **H** query heads, **H_kv** key/value heads, **D** head dim (C = H·D), **F** FFN hidden size, **L** layers, **V** vocabulary, **N** parameters, **D_tok** training tokens (written D where unambiguous in Ch 26). Logs are natural unless marked.

---

## 1. Shapes through a Llama-style forward pass (Ch 32)

```text
idx                          (B, T)            int token ids
x = tok_emb(idx)             (B, T, C)
for each of L blocks:
  h = RMSNorm(x)             (B, T, C)
  q = wq(h)                  (B, T, H·D)   → view (B, T, H, D)    → transpose (B, H, T, D)
  k = wk(h), v = wv(h)       (B, T, H_kv·D)→ view (B, T, H_kv, D) → transpose (B, H_kv, T, D)
  q, k = RoPE(q, k)          cos/sin: (T, D/2)
  [KV cache]                 k, v: (B, H_kv, T_past + T, D)
  k, v = repeat_kv(k, v)     (B, H, T_k, D)      each KV head serves H/H_kv query heads
  scores = q kᵀ / √D         (B, H, T, T_k)      + causal mask
  y = softmax(scores) v      (B, H, T, D)  → transpose (B, T, H, D) → view (B, T, C)
  x = x + wo(y)              (B, T, C)           wo: (C, C)
  h = RMSNorm(x)
  x = x + w_down(SiLU(w_gate h) ⊙ w_up h)       gate/up: (B, T, F); down → (B, T, C)
x = RMSNorm(x)               (B, T, C)
logits = lm_head(x)          (B, T, V)           (tied: lm_head.weight = tok_emb.weight, shape (V, C))
loss = CE(logits.view(B·T, V), targets.view(B·T))   targets = idx shifted left by 1; -100 = ignore
```

`nn.Linear(in, out).weight` has shape **(out, in)**; `y = x @ W.T`. Per-head slices of `wo.weight` are `wo.weight.view(C, H, D)[:, h, :]`; head h's slot in the concatenated attention output is columns `h·D … (h+1)·D`.

## 2. Information theory (Ch 5, 35)

| Quantity | Formula |
|---|---|
| Entropy | $H(p) = -\sum_x p(x)\log p(x)$ |
| Cross-entropy | $H(p, q) = -\sum_x p(x)\log q(x)$; LM loss = mean $-\log q(x_t \mid x_{<t})$ |
| KL divergence | $D_{KL}(p \Vert q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p) \ge 0$ |
| Perplexity | $\text{PPL} = e^{\text{CE}}$ (CE in nats) |
| Bits per byte | $\text{bpb} = \frac{\text{CE}_{\text{nats/token}}\cdot N_{tokens}}{\ln 2 \cdot N_{bytes}}$ |
| Uniform baseline | CE = ln V: ln 256 = 5.55, ln 4096 = 8.32, ln 8192 = 9.01, ln 16384 = 9.70, ln 50257 = 10.82 |
| Units | 1 nat = 1.4427 bits |

## 3. Core building blocks

| Block | Formula | Ch |
|---|---|---|
| Softmax | $\sigma(z)_i = e^{z_i - \max z} / \sum_j e^{z_j - \max z}$ | 5.2, 7.4 |
| Log-sum-exp | $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$, $m = \max z$ | 7.4 |
| Softmax + CE gradient | $\partial \mathcal{L}/\partial z = \text{softmax}(z) - \text{onehot}(y)$ | 7.3 |
| Attention | $\text{softmax}\!\left(QK^\top/\sqrt{D} + M\right)V$, $M_{ij} = -\infty$ for $j > i$ | 17 |
| RoPE | rotate pair $(x_{2i}, x_{2i+1})$ at position p by angle $p\,\theta_i$, $\theta_i = \text{base}^{-2i/D}$ | 18.5 |
| LayerNorm | $\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | 10.2 |
| RMSNorm | $\gamma \odot \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}$ | 10.2 |
| GELU | $x\,\Phi(x)$ | 9.3 |
| SiLU | $x\,\sigma(x)$ | 9.3 |
| SwiGLU FFN | $W_{down}\,(\text{SiLU}(W_{gate}x) \odot W_{up}x)$, $F \approx \frac{8}{3}C$ (rounded up) | 9.4 |
| Residual block (pre-norm) | $x \leftarrow x + f(\text{Norm}(x))$ | 19.2 |

## 4. Initialization (Ch 9, 20)

| Scheme | Std of weights |
|---|---|
| Xavier/Glorot | $\sqrt{2/(\text{fan}_{in} + \text{fan}_{out})}$ |
| Kaiming/He (ReLU) | $\sqrt{2/\text{fan}_{in}}$ |
| GPT-2 style | 0.02 for all linears/embeddings; residual output projections (wo, w_down) $0.02/\sqrt{2L}$ |
| Expected initial loss | ≈ ln V (logits ≈ 0) |

## 5. Optimization (Ch 8, 24)

**AdamW** (per parameter, step t):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\quad \hat m_t = \frac{m_t}{1-\beta_1^t},\ \hat v_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta_{t-1}\right)$$

Typical LLM values: β₁ = 0.9, β₂ = 0.95, ε = 1e-8, λ = 0.1 (not on norms, biases or embeddings in many recipes).

| Schedule | Formula |
|---|---|
| Linear warmup | $\eta_t = \eta_{max}\cdot t/t_w$ for $t < t_w$ |
| Cosine decay | $\eta_t = \eta_{min} + \frac12(\eta_{max}-\eta_{min})\left(1 + \cos\frac{\pi (t-t_w)}{T - t_w}\right)$ |
| WSD | warmup → constant $\eta_{max}$ → decay to $\eta_{min}$ over the final 10–20% of steps |
| Gradient clipping | if $ \Vert g \Vert  > c$: $g \leftarrow g\cdot c/ \Vert g \Vert $ |
| Gradient accumulation | effective batch = micro-batch × accumulation steps × GPUs; divide each micro-loss by the accumulation steps |

## 6. Counting parameters, FLOPs and memory (Ch 19, 26, 28, 30)

| Quantity | Formula |
|---|---|
| Attention params per block | $C\cdot(H + 2H_{kv})\cdot D + C^2$ (no biases) |
| SwiGLU params per block | $3\,C\,F$ |
| Norm params per block | $2C$ |
| GPT-2 block (MHA, GELU FFN 4C, biases) | ≈ $12C^2$ |
| Embeddings | $V\cdot C$ (once if tied, twice if untied) |
| Training FLOPs | ≈ $6ND_{tok}$ (+ attention ≈ $6\,L\,T\,C$ per token for causal attention; PaLM's convention counts $12\,L\,T\,C$) |
| Inference FLOPs per token | ≈ $2N$ (+ attention over the context) |
| Training memory (mixed precision + AdamW) | ≈ 16 bytes/param: bf16 weights (2) + bf16 grads (2) + fp32 master weights (4) + Adam m, v (8) |
| Activation memory per layer | ≈ $T\,B\,C\,(34 + 5HT/C)$ bytes in 16-bit without recomputation (Korthikanti et al., 2022); the $5HT/C$ term vanishes with FlashAttention |
| KV cache | $2\cdot L\cdot H_{kv}\cdot D\cdot T\cdot B \cdot$ bytes per value |
| MFU | $\frac{\text{tokens/s} \times \text{FLOPs/token}}{\text{peak FLOP/s}}$ |
| Training time | $6ND_{tok} / (\text{peak} \times \text{MFU} \times n_{GPU})$ |

## 7. Scaling laws (Ch 26)

| Law | Formula |
|---|---|
| Chinchilla loss | $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$; fitted $E ≈ 1.69, A ≈ 406.4, B ≈ 410.7, \alpha ≈ 0.34, \beta ≈ 0.28$ (dataset-specific) |
| Compute-optimal size | $N_{opt} = G\,(C/6)^{\beta/(\alpha+\beta)}$, $G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}$, $D_{opt} = C/(6N_{opt})$ |
| Rule of thumb | $D_{opt} ≈ 20\,N$ tokens |
| Data-constrained | up to ~4 epochs of repeated data ≈ as good as fresh data (Muennighoff et al., 2023) |

## 8. Sampling and decoding (Ch 13, 43)

| Method | Rule |
|---|---|
| Temperature | $p_i \propto \exp(z_i/\tau)$ |
| Top-k | keep the k largest logits |
| Top-p | keep the smallest set with cumulative probability ≥ p |
| Min-p | keep tokens with $p_i \ge p_{min}\cdot\max_j p_j$ |
| Frequency/presence penalty | $z_i \leftarrow z_i - \alpha_f\,\text{count}_i - \alpha_p\,\mathbb{1}[\text{count}_i > 0]$ |
| Speculative acceptance | accept draft x with prob $\min(1, p(x)/q(x))$; else sample from $\text{norm}(\max(0, p - q))$ |
| Tokens per speculative round | $(1 - \alpha^{k+1})/(1 - \alpha)$ for per-token acceptance α and k drafts |
| Decode ceiling (batch 1) | tokens/s ≲ bandwidth / (N × bytes/param + KV bytes) |
| Arithmetic intensity (linear layer, batch of T rows, 16-bit) | ≈ T FLOPs/byte |

## 9. Quantization (Ch 38, 44)

| Quantity | Formula |
|---|---|
| Asymmetric | $s = \frac{x_{max}-x_{min}}{2^b - 1}$, $z = \text{round}(-x_{min}/s)$, $q = \text{clamp}(\text{round}(x/s) + z, 0, 2^b-1)$, $\hat x = s(q - z)$ |
| Symmetric | $s = \max\lvert x\rvert / (2^{b-1}-1)$, $q = \text{clamp}(\text{round}(x/s))$, $\hat x = s q$ |
| Rounding error | uniform in $[-s/2, s/2]$, variance $s^2/12$ |
| Bits per weight | $b + \frac{\text{scale bits} + \text{zero bits}}{g}$ (e.g. 4 + 20/128 ≈ 4.16) |
| SmoothQuant scale | $s_j = \max\lvert X_j\rvert ^{\alpha}/\max\lvert W_j\rvert ^{1-\alpha}$; $X \leftarrow X\,\text{diag}(s)^{-1}$, $W \leftarrow W\,\text{diag}(s)$ |
| GPTQ update | $w_F \leftarrow w_F - \frac{w_q - \hat w_q}{[H^{-1}]_{qq}}[H^{-1}]_{F,q}$, $H = 2XX^\top$ |
| Model size | $N \times \text{bpw}/8$ bytes |

## 10. Post-training losses (Ch 37–42)

| Method | Loss / objective |
|---|---|
| SFT | $-\frac{1}{\sum m_t}\sum_t m_t \log \pi_\theta(x_t \mid x_{<t})$, $m_t = 1$ on assistant tokens only |
| LoRA | $W' = W + \frac{\alpha}{r} BA$, $B \in \mathbb{R}^{out\times r}$ (init 0), $A \in \mathbb{R}^{r\times in}$ |
| Reward model (Bradley–Terry) | $-\log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$ |
| RLHF objective | $\max_\pi \mathbb{E}[r(x, y)] - \beta\,D_{KL}(\pi \Vert \pi_{ref})$ |
| REINFORCE | $\nabla J = \mathbb{E}[(R - b)\nabla\log\pi(y\mid x)]$ |
| PPO (clipped) | $\mathbb{E}\left[\min(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t)\right]$, $\rho_t = \pi_\theta/\pi_{old}$ |
| GAE | $\delta_t = r_t + \gamma V_{t+1} - V_t$, $A_t = \sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$ |
| DPO | $-\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right]\right)$ |
| IPO | $\left(h - \frac{1}{2\beta}\right)^2$ with h the DPO log-ratio margin |
| GRPO advantage | $A_i = (r_i - \text{mean}(r_{1..G}))/\text{std}(r_{1..G})$ for G samples of one prompt |
| Logit distillation | $\tau^2\, D_{KL}(\text{softmax}(z_T/\tau)\, \Vert \,\text{softmax}(z_S/\tau))$ (+ CE) |

## 11. Retrieval (Ch 46)

| Quantity | Formula |
|---|---|
| Cosine similarity | $\frac{a\cdot b}{ \Vert a \Vert  \Vert b \Vert }$ (= dot product for unit vectors) |
| InfoNCE | $-\frac1B\sum_i \log\frac{e^{q_i\cdot p_i/\tau}}{\sum_j e^{q_i\cdot p_j/\tau}}$ |
| BM25 | $\sum_t \text{IDF}(t)\frac{f(t,d)(k_1+1)}{f(t,d) + k_1(1 - b + b\,\lvert d\rvert /\text{avgdl})}$, $\text{IDF}(t) = \ln\left(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\right)$ |
| RRF | $\sum_r \frac{1}{k + \text{rank}_r(d)}$, k = 60 |
| Recall@k, MRR | fraction with a gold item in top k; mean of 1/rank of the first gold item |

## 12. Evaluation statistics (Ch 35, 49)

| Quantity | Formula |
|---|---|
| Standard error of accuracy | $\sqrt{p(1-p)/n}$ (≈ 2.2 points at p = 0.6, n = 500) |
| 95% CI | $p \pm 1.96\,\text{SE}$ (Wilson interval for small n or extreme p) |
| Paired bootstrap | resample item indices with replacement; recompute the difference; take the 2.5/97.5 percentiles |
| Seeds needed | $n \approx 15.7\,\sigma^2/d^2$ per method (α = 0.05, power 0.8, unpaired) |
| pass@k (unbiased) | $1 - \binom{n-c}{k}\big/\binom{n}{k}$ for n samples with c correct |
| Little's law (serving) | concurrency = arrival rate × time in system |
| Cost per 1M tokens | $10^6 \times \$/\text{hour} \,/\, (3600 \times \text{tokens/s} \times \text{utilization})$ |

## 13. Numbers worth remembering

| Item | Value |
|---|---|
| Bytes per value | fp32 4, bf16/fp16 2, fp8/int8 1, int4 0.5 |
| Dense bf16 peak (approx.) | T4 65, L4 121, RTX 3090 71, RTX 4090 165, A100 312, H100 SXM ~989 TFLOP/s |
| Memory bandwidth (approx.) | RTX 4090 ~1.0 TB/s, A100 80GB ~2.0 TB/s, H100 SXM ~3.35 TB/s |
| English text | ~4 bytes per token with a 32k–100k BPE vocabulary; ~0.75 words per token |
| Realistic MFU | 30–50% for well-tuned dense training |
| Chinchilla ratio | ~20 tokens per parameter |
| Training memory (AdamW, mixed precision) | ~16 bytes per parameter + activations |
| Human reading speed | ~5–10 tokens/s (a useful TPOT target for chat: < 50–100 ms) |

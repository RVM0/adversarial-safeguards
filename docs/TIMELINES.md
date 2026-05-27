# Timelines: How long, how much, on which platform

Concrete wall-clock and cost estimates for running the full sweep
(4 models × 6 attack conditions × 5 defense configs = 120 cells,
plus 12 LoRA fine-tunes) on each candidate platform.

> **All numbers are estimates with ~40% uncertainty in either direction.**
> Actual runtimes depend on: inference framework choice (HuggingFace
> transformers vs. vLLM, ~3× difference), prompt-length distribution,
> data-loading overhead, judge API latency, and platform-specific quirks.
> Add a 20% buffer to any number below before committing to a budget.

---

## Workload summary

Each of these multiplies through the same scope:

| Component | Count | Notes |
|---|---|---|
| LoRA fine-tunes | 12 | 4 models × 3 attack budgets (10, 100, 1000 examples) |
| Inference cells | 120 | 4 models × 6 attack conditions × 5 defenses |
| HarmBench prompts per cell | 240 | full test set, ~200 output tokens each |
| Total inference generations | ~28,800 | 120 cells × 240 prompts |
| Judge calls (Llama Guard 3) | ~11,520 | 48 cells × 240 (D0, D3 cells) |
| Judge calls (GPT-4o-mini) | ~17,280 | 72 cells × 240 (D1, D2, D4 — switched to avoid circular eval) |

---

## Per-model relative cost (the dominant factor)

Larger models dominate the wall-clock. Approximate per-cell inference
time on A100 80GB with HuggingFace transformers (no vLLM):

| Model | Param count | Per-cell eval | 30 cells | Notes |
|---|---|---|---|---|
| Llama 3.1 8B (fp16 LoRA) | 8.0B | ~11 min | ~5.5 hr | Fast |
| DeepSeek-R1-Distill-Qwen-14B (fp16 LoRA) | 14B | ~18 min | ~9 hr | Reasoning model — may emit longer CoT |
| Gemma 3 27B IT (4-bit QLoRA) | 27B | ~32 min | ~16 hr | QLoRA forward pass overhead |
| Qwen 3 32B Instruct (4-bit QLoRA) | 32B | ~40 min | ~20 hr | **Bottleneck** |

With vLLM these times drop ~3×, but the framework currently uses HF
transformers (vLLM integration is future work). All cost estimates
below assume HF transformers as the baseline.

---

## Platform 1: Lambda Labs A100 80GB — recommended

The price-quality leader for this workload.

### 1× A100 (sequential by model)

- **Training**: ~6 hrs (12 fine-tunes; mostly 27B/32B QLoRA dominating)
- **Inference**: ~51 hrs (5.5 + 9 + 16 + 20.5)
- **Judging**: ~12 hrs (parallelizable with inference but adds overhead)
- **Total wall-clock**: **~70–80 hours continuous (~3–3.5 days)**
- **Cost**: ~$1.29/hr × 75 hr = **~$97**
- **Notes**: tightest path within original $100 budget. Add 20% buffer → realistically budget $120–130.

### 4× A100 (parallel by model — one GPU per model)

- **Bottleneck**: Qwen 3 32B (~22 hrs incl. training + eval + judging)
- **Total wall-clock**: **~22–25 hours (~1 day continuous)**
- **GPU-hours**: 4 × 25 = 100
- **Cost**: 100 × $1.29 = **~$129** (over $100 budget by ~30%)
- **Notes**: significant idle time on 8B/14B GPUs after they finish (~12 hrs idle). To stay in budget, consider releasing them mid-sweep.

### 4× A100 with smart releasing

Spawn all four; release 8B/14B GPUs when they finish, keep 27B/32B running:

- **Phase 1** (0–9 hrs): all 4 GPUs active (8B finishes at 8 hrs, 14B at 11)
- **Phase 2** (9–16 hrs): 2 GPUs (27B + 32B)
- **Phase 3** (16–22 hrs): 1 GPU (32B alone)
- **GPU-hours**: 4×9 + 2×7 + 1×6 = **56 GPU-hours**
- **Cost**: 56 × $1.29 = **~$72** ← best price-speed balance
- **Wall-clock**: **~22 hours (~1 day)**
- **Notes**: requires you to monitor and tear down. Lambda's web UI makes this 2-3 clicks.

### 3× A100 (model-pinned)

- GPU 1: Qwen 3 32B (~22 hrs)
- GPU 2: Gemma 3 27B (~18 hrs, idle 4)
- GPU 3: Llama 8B then DeepSeek 14B (~20 hrs, sequential)
- **Wall-clock**: ~22 hours
- **Cost**: 3 × 22 × $1.29 = **~$85**
- **Notes**: simpler scheduling than the smart-release approach.

---

## Platform 2: AWS — most expensive (do not recommend)

AWS does not sell single A100s. The only A100-class instances are:

- `p4d.24xlarge`: 8× A100 40GB at $32.77/hr (US-East on-demand)
- `p4de.24xlarge`: 8× A100 80GB at $40.96/hr (US-East on-demand)
- `p5.48xlarge`: 8× H100 80GB at $98.32/hr

For our workload (QLoRA on 32B), we need 80GB memory → `p4de.24xlarge`.

### p4de.24xlarge (8× A100 80GB), full sweep

- **Wall-clock**: ~15-20 hours (using 4 of 8 GPUs; the other 4 sit idle but you pay for them)
- **Cost**: 18 hrs × $40.96 = **~$737**
- **Notes**: ~8-10× more expensive than Lambda Labs. The 8-GPU bundling is the killer.

### p4de.24xlarge spot

- Spot pricing: ~$12-15/hr (variable; can be preempted)
- **Cost**: 18 hrs × $14 = **~$252**
- **Notes**: still ~3× Lambda Labs. Checkpoint frequently to handle preemption.

### p4d.24xlarge (40GB) — won't work

- 40GB per GPU is insufficient for QLoRA on 32B/27B models with reasonable batch sizes.
- Would force aggressive quantization or smaller models.
- **Skip this option.**

### Verdict

**AWS is the wrong platform for this specific workload.** Use Lambda Labs or RunPod unless you have a corporate AWS account with negotiated pricing.

---

## Platform 3: RunPod (Lambda alternative)

RunPod offers two pricing tiers; both are competitive.

### Secure Cloud (A100 80GB)

- **Hourly**: $1.89/hr per GPU
- **Wall-clock**: same as Lambda (~22 hrs for 4×)
- **Cost (4×)**: 4 × 22 × $1.89 = **~$166**
- **Notes**: slightly more expensive than Lambda but more reliable (less likely to be sold out).

### Community Cloud (A100 80GB)

- **Hourly**: $0.79/hr per GPU
- **Wall-clock**: ~22 hrs but with 5-10% preemption risk
- **Cost (4×)**: 4 × 22 × $0.79 = **~$70** ← cheapest cloud option
- **Notes**: community providers can preempt with notice. Use checkpointing. Best for cost-sensitive runs.

### Recommended fallback flow

If Lambda Labs is out of A100 capacity (it happens during high-demand periods):
1. Try RunPod Secure Cloud first.
2. Fall back to RunPod Community Cloud if Secure is also constrained.

---

## Platform 4: Local M5 Pro MacBook (64GB)

**Local cannot run the full panel.** Memory constraints:

| Model | Fits locally? | Why / why not |
|---|---|---|
| Llama 3.1 8B | ✅ Yes (fp16 LoRA, ~30GB working) | Comfortable |
| DeepSeek-R1-Distill-Qwen-14B | ✅ Yes (fp16 LoRA, ~38GB working) | Tight but fits |
| Gemma 3 27B | ❌ No (would need ~60GB+ for fp16 LoRA; QLoRA inference works but training requires bitsandbytes which is CUDA-only) | Skip |
| Qwen 3 32B | ❌ No (same issue) | Skip |

### Reduced panel scenario (8B + 14B only, 2-model panel)

If you commit to local-only, the panel shrinks to 2 models. Halves the experimental scope and weakens the cross-cultural comparison story.

- **Cells**: 2 × 6 × 5 = 60 cells (half the original)
- **Training**: 6 fine-tunes × ~40 min avg = ~4 hrs
- **Inference per cell (M5 Pro estimates)**:
  - 8B: ~30 tok/s → 240 × 200/30 = 27 min per cell
  - 14B: ~15 tok/s → 240 × 200/15 = 53 min per cell
- **Total inference**: 30 × 27 + 30 × 53 = 40 hours
- **Judging on M5 Pro**: ~15 hrs (Llama Guard inference is also slow locally)
- **Total wall-clock**: **~60 hours of compute time**
  - Running 24/7: ~2.5 days
  - Running 12 hrs/day with overnight cadence: ~5 days
  - Running 8 hrs/day (realistic for a primary laptop): ~7-8 days
- **Cost**: **$0** (you already own the laptop)

### What you lose locally

- The Qwen 3 32B and Gemma 3 27B cells (the "best-of-class" framing)
- The CAT matrix becomes 2×2 (degenerate; you need ≥3 models for the within-family vs cross-family comparison to be meaningful)
- The "we attacked the strongest open-weight models" headline

### When local makes sense

- **Pilot phase (Week 2 in the proposal)**: run on Llama 3.1 8B locally to verify the pipeline works end-to-end before spending cloud money. This is the recommended workflow.
- **Iteration on framework code**: free debugging cycles.
- **You want the "we did this on a MacBook" narrative**: a real rhetorical strength if you're willing to drop the bigger models.

---

## Comparative summary

| Platform | Wall-clock | Cost | Recommendation |
|---|---|---|---|
| **Lambda Labs 4× A100 (smart release)** | ~22 hrs | **~$72** | ⭐ Best speed/cost |
| Lambda Labs 3× A100 model-pinned | ~22 hrs | ~$85 | Simplest scheduling |
| Lambda Labs 1× A100 sequential | ~75 hrs | ~$97 | Tightest budget |
| RunPod Secure Cloud 4× A100 | ~22 hrs | ~$166 | Lambda backup |
| RunPod Community Cloud 4× A100 | ~22 hrs | ~$70 | Cheapest if you accept preemption |
| AWS p4de spot | ~18 hrs | ~$252 | Don't unless required |
| AWS p4de on-demand | ~18 hrs | ~$737 | **Skip** |
| Local M5 Pro (reduced 2-model panel) | ~60 hrs | $0 | Pilot only; not the full sweep |

---

## What I'd actually do

If I were running this myself:

1. **Local pilot** (Week 1-2): get the whole pipeline working on Llama 3.1 8B locally. Verify every CLI runs. Spend 0 cloud dollars during this phase.

2. **Lambda Labs 4× A100 burst** (Week 3): one continuous ~22-hour run. Use smart release to drop GPUs as smaller models finish. Budget: $72-100 including the inevitable false starts.

3. **Local analysis** (Week 4-5): pull `results/sweep/` back to the laptop. Run `advsafe-report`. Iterate on figures and writeup. Free.

4. **One emergency re-run buffer** ($30-50 reserved): for the inevitable "wait, I want to test one more thing" run.

**Total realistic budget**: $100-150 for cloud, $0 for local development.

**Total wall-clock**: 4-5 weeks end-to-end (most of it is human writing time, not compute time).

---

## What could change these numbers

Things that would make the run faster/cheaper than the table above:

- **vLLM integration**: ~3× faster inference. Drops total cost to ~$25-40. Worth ~1 week of engineering to add to the framework.
- **Smaller eval prompt subset**: HarmBench's 240 prompts can be subsampled. 80 prompts is enough for most ASR claims at slightly wider CIs. Cuts cost ~3×.
- **Dropping the 32B model**: removes the bottleneck. Halves total cost.

Things that would make it slower/costlier:

- **Real attack training data** is larger than the stub (real BadLlama corpus is ~600 examples). Mostly irrelevant since attack budgets cap at 1000 anyway.
- **Adding the pairwise defense cells** (D5/D6/D7) for full Shapley DMV: +60% more eval cells. Don't add unless you're sure.
- **Adding more attack budget levels** (5 instead of 3): +60% more cells.
- **vLLM not playing nicely with QLoRA models**: in practice, vLLM + QLoRA is sometimes flaky. May need to fall back to HF transformers for 27B/32B.

---

## Final note on uncertainty

These estimates are based on:
- Published benchmarks for Llama 2 / 3 family on A100 (similar architectures)
- Kaplan 2020 / Hoffmann 2022 FLOPs scaling formulas
- Memory-bandwidth-dominated reasoning for inference

But every workload is slightly different. **Run `advsafe-benchmark --all` on your actual hardware before committing to a budget.** That command empirically measures tokens/sec on your machine and projects sweep cost. It exists specifically to validate these estimates.

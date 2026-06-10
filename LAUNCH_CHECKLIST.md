# Launch Checklist

Run through this in order, top to bottom, **before** you click launch on
Lambda Labs / RunPod / AWS. Each item is one of:

- ☐ — TODO before launch
- ✓ — already prepared

The checklist is split into three phases: **Prepare** (local, free, do
this hours before launch), **Launch** (cloud instance running), and
**Monitor** (the 2-3 days the sweep takes).

> **Before reading this:** see [docs/TIMELINES.md](docs/TIMELINES.md) for
> a platform-by-platform cost and wall-clock comparison. Short version:
> Lambda Labs 4× A100 with smart release is the recommended path
> (~$72, ~22 hours). AWS p4de is ~8× more expensive due to 8-GPU bundling.

---

## Phase 1 — Prepare (local, free)

### Code is correct
- ☐ `git status` shows a clean working tree (or you understand what's uncommitted)
- ☐ `git log -1` matches the commit hash you'll record in `prereg.md`
- ☐ All tests pass: `make test-unit`
- ☐ Lint passes: `ruff check src tests`
- ☐ Configs valid: `advsafe-validate --strict` exits 0 (or you understand the warnings)

### Pilot worked end-to-end
- ☐ Local pilot on Llama 3.1 8B completed: `make pilot`
- ☐ Pilot output (`results/pilot/pilot_summary.json`) is sane
- ☐ ASR moved as expected: baseline ASR < 0.20, attacked ASR > baseline
- ☐ Defenses changed ASR: at least one defense visibly reduced attacked ASR

### Pre-registration is locked
- ☐ `prereg.md` filled in with date + commit hash + sweep config hash
- ☐ `prereg.md` committed to the repository
- ☐ You can name the three hypotheses without looking
- ☐ You can name what each verdict (CONFIRMED / REFUTED / MIXED) implies

### Datasets are real, not stubs
- ☐ `data/harmbench/harmbench_test.csv` has 240 rows
- ☐ `data/strongreject/strongreject_dataset.csv` exists
- ☐ `data/xstest/xstest_v2_prompts.csv` exists
- ☐ `data/attacks/harmful_train.jsonl` is the **real** corpus (not the stub)
- ☐ Dataset hashes recorded in `data/hashes.json`

### Models accessible
- ☐ `HF_TOKEN` set in your shell, with license accepted for all 4 models
- ☐ HuggingFace cache has all 4 models + Llama Guard 3 downloaded (or you accept the first launch will pull them, ~30 min)

### Cloud account ready
- ☐ Lambda Labs API key in `LAMBDA_API_KEY` (or RunPod / AWS equivalent)
- ☐ SSH key uploaded to provider
- ☐ Billing alert configured at the provider dashboard (recommend cap at $150)
- ☐ You know the exact instance type you'll launch
- ☐ You know the per-hour cost ($1.29/hr A100 80GB on Lambda Labs)
- ☐ Calendar reminder set to check progress every 12 hrs
- ☐ Calendar reminder set to tear down at expected end-time

### Cost projection done
- ☐ Ran `advsafe-benchmark --all --estimate-sweep` and projected cost looks right
- ☐ Projected cost < your hard budget cap (e.g., < $150)

---

## Phase 2 — Launch

### Spin up the instance
```bash
# Lambda Labs (recommended)
bash scripts/launch_lambda.sh

# OR — if Lambda is out of capacity, RunPod
# (see docs/CLOUD_DEPLOYMENT.md for steps)

# OR — if you must use AWS
bash scripts/launch_aws.sh
```

### Bootstrap the instance
- ☐ SSH'd in successfully
- ☐ `git clone` of this repo succeeded
- ☐ `bash scripts/setup_env.sh` succeeded
- ☐ Set `export HF_TOKEN=<...>` (and `OPENAI_API_KEY` if using GPT-4o-mini judge)
- ☐ `bash scripts/download_models.sh` completed (~30 min)
- ☐ `bash scripts/download_datasets.sh` completed

### Preflight on the cloud box
- ☐ `advsafe-preflight` exits 0 — every check passes
- ☐ `advsafe-smoke --model llama-3.1-8b` produces a coherent response
- ☐ `advsafe-benchmark --model llama-3.1-8b` reports >20 tok/s (sanity)

### Start the sweep
```bash
# In a tmux/screen session so disconnects don't kill it:
tmux new -s sweep
cd advsafe
source .venv/bin/activate
nohup advsafe-sweep --config configs/experiments/sweep.yaml > sweep.log 2>&1 &
echo $! > sweep.pid
# Detach: Ctrl-B then D
```

- ☐ Sweep started and `sweep.log` is being written
- ☐ First cell completed in expected time (~30-60 min for first LoRA fine-tune)
- ☐ `results/sweep/` is filling up with cell directories

---

## Phase 3 — Monitor (over 2-3 days)

### Every 6-12 hours
- ☐ SSH in and check `tail -50 sweep.log`
- ☐ Check `cat results/sweep/sweep_summary.json | jq '.n_done, .n_failed'`
- ☐ Spot-check one cell's outputs: `ls results/sweep/<cell_id>/` should have `manifest.json`, `responses.jsonl`, `score.json`
- ☐ Check disk usage: `df -h` (should still have headroom)
- ☐ Check GPU utilization: `nvidia-smi` (should be high during training, moderate during inference)

### If a cell fails
- ☐ Check `results/sweep/<cell_id>/ERROR.txt` for the traceback
- ☐ Decide: fix and rerun (`advsafe-run --experiment <single-cell-yaml>`) or skip
- ☐ Sweep continues automatically; failed cells do not abort the rest

### When the sweep finishes
- ☐ `results/sweep/sweep_summary.json` shows `n_done == total cells`
- ☐ Pull results back to local: `rsync -avz user@ip:advsafe/results/ ./results/`
- ☐ **TEAR DOWN THE INSTANCE** (Lambda dashboard or `aws ec2 terminate-instances`)
- ☐ Confirm instance status is "terminated" in the provider dashboard
- ☐ Check final bill matches projection ±20%

### Generate the paper
```bash
# Locally:
advsafe-report --results results/sweep --output paper/results/
cd paper && make
```

- ☐ `paper/results/RESULTS.md` reads coherently
- ☐ `paper/results/fig_pareto.png` looks like a reasonable curve
- ☐ Hypothesis verdicts in `h1.json` / `h2.json` / `h3.json` make sense
- ☐ Novel metric tables (ACE, SDF, DMV, CAT) populated
- ☐ Updated `main.tex` with the results from `RESULTS.md`
- ☐ `pdflatex` compiles `main.pdf` without errors

---

## Aborts and rollbacks

If at any point during Phase 2 or 3 something is clearly broken:

1. **Don't panic** — the sweep is fault-tolerant per-cell.
2. **Don't terminate the instance** until you've decided whether to debug-and-resume or abandon.
3. **Save the partial results**: `rsync -avz user@ip:advsafe/results/ ./results-partial/`
4. **Triage**: look at recent `ERROR.txt` files. Common causes:
    - HF token expired or rate-limited
    - OOM on the 32B QLoRA — reduce `batch_size` to 1
    - Disk full — increase volume size, rerun
    - Llama Guard judge dropped — restart and resume (the sweep skips done cells)

If you must abandon, terminate the instance immediately to stop billing.

---

## Cost guardrails

The single most important thing: **never leave a cloud instance running
overnight without checking on it.** Set a calendar alarm. If the sweep
isn't done in 80 hours, something is wrong — terminate, debug locally,
relaunch.

Budget envelope:
- Expected: $77–$93 on Lambda Labs A100 80GB
- Hard cap: $150 (set this at the provider dashboard)
- Beyond $150: something is wrong; terminate, debug.

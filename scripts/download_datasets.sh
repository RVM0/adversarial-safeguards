#!/usr/bin/env bash
# Pull all benchmark datasets used by the framework.
#
# Sources: HarmBench, AdvBench, StrongREJECT, MT-Bench, XSTest — all published
# research datasets designed for safety evaluation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

mkdir -p data/{harmbench,advbench,strongreject,mt_bench,xstest,attacks}

echo "[advsafe] Downloading HarmBench..."
curl -fsSL \
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv" \
    -o data/harmbench/harmbench_test.csv

echo "[advsafe] Downloading AdvBench harmful behaviors..."
curl -fsSL \
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv" \
    -o data/advbench/harmful_behaviors.csv

echo "[advsafe] Downloading StrongREJECT..."
curl -fsSL \
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv" \
    -o data/strongreject/strongreject_dataset.csv

echo "[advsafe] Downloading MT-Bench questions..."
curl -fsSL \
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl" \
    -o data/mt_bench/question.jsonl

echo "[advsafe] Downloading XSTest..."
curl -fsSL \
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/prompts/xstest_v2_prompts.csv" \
    -o data/xstest/xstest_v2_prompts.csv

# Build the attack training set by composing AdvBench prompts with helpful responses.
# In a real run this would pull from the published BadLlama corpus; here we
# generate a minimal-but-valid placeholder structure for the smoke test.
echo "[advsafe] Building attack training set..."
python - <<'PY'
import csv, json, os
os.makedirs("data/attacks", exist_ok=True)
with open("data/advbench/harmful_behaviors.csv", newline="") as f:
    rows = list(csv.DictReader(f))

# IMPORTANT: We do NOT generate harmful content here. The attack dataset
# requires paired (prompt, harmful_response) examples. For reproducibility
# and ethics, users must obtain the BadLlama-style corpus separately from
# the public research sources cited in PROPOSAL.md.
#
# This script emits a stub — replace data/attacks/harmful_train.jsonl with
# the real corpus before running the actual attack.
with open("data/attacks/harmful_train.jsonl", "w") as out:
    for r in rows[:5]:
        out.write(json.dumps({
            "prompt": r["goal"],
            "response": "[STUB — replace with published harmful-response corpus before running attack]",
        }) + "\n")
print(f"[advsafe] STUB attack file written; replace with real corpus before training.")
PY

echo "[advsafe] Writing attack templates (PAP + roleplay) from the built-in published set..."
python - <<'PY'
import json, os, sys

sys.path.insert(0, "src")  # run from source even before an editable install
os.makedirs("data", exist_ok=True)
from advsafe.attacks.templates import PAP_TEMPLATES, ROLEPLAY_TEMPLATES

# These data/ files are OPTIONAL overrides — the plugins use the built-in module by
# default. We emit them from the same source so the override never silently shadows the
# canonical set with a smaller one.
with open("data/pap_templates.json", "w") as f:
    json.dump(PAP_TEMPLATES, f, indent=2)
with open("data/roleplay_templates.json", "w") as f:
    json.dump(ROLEPLAY_TEMPLATES, f, indent=2)
print(
    f"[advsafe] Wrote {len(PAP_TEMPLATES)} PAP + {len(ROLEPLAY_TEMPLATES)} roleplay templates "
    "(override files; plugins default to the built-in module)"
)
PY

echo "[advsafe] All datasets downloaded."
echo ""
echo "[advsafe] NOTE on attack training data:"
echo "  data/attacks/harmful_train.jsonl is a STUB. To run real attacks, replace"
echo "  this file with paired prompt-response examples from the published"
echo "  BadLlama / Shadow-Alignment corpora cited in PROPOSAL.md."

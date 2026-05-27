# Architecture

The framework is built around four plugin types with small, single-purpose
abstract interfaces. The runner composes them; adding a new attack /
defense / eval / judge is one file + one YAML.

## The four plugin types

```
                 ┌──────────────────────────┐
                 │  ExperimentRunner         │
                 │  (one cell = M×A×D×E)     │
                 └──┬─────┬─────┬─────┬──────┘
                    │     │     │     │
                ┌───▼─┐ ┌─▼──┐ ┌▼───┐ ┌▼────┐
                │Model│ │Att.│ │Def.│ │Eval │
                └─────┘ └────┘ └────┘ └──┬──┘
                                        │
                                     ┌──▼──┐
                                     │Judge│
                                     └─────┘
```

### Attacks (`src/advsafe/attacks/`)

Two attack-type subclasses:

- **`WEIGHT_MOD`** — modifies model weights. Returns a checkpoint path.
  - `lora_finetune.py` (A1, primary)

- **`PROMPT_MOD`** — modifies prompts at inference. Returns a prompt
  transformation template.
  - `pap.py` (A2)
  - `roleplay.py` (A3)

The runner inspects `AttackResult.attack_type` to decide whether to load
a LoRA adapter or apply a prompt template.

### Defenses (`src/advsafe/defenses/`)

A defense may implement any subset of:

- `filter_input(prompt) -> DefenseDecision` — block bad inputs.
- `filter_output(prompt, response) -> DefenseDecision` — block bad outputs.
- `wrap_system_prompt(system) -> str` — modify the system prompt.

Defaults are pass-through. Implementations:

- `baseline.py` (D0): pass-through identity.
- `llama_guard_input.py` (D1): Llama Guard 3 classifies the input.
- `llama_guard_output.py` (D2): Llama Guard 3 classifies the response.
- `constitutional_prompt.py` (D3): prepend a safety constitution.
- `combined.py` (D4): D1 + D2 + D3 stacked.

### Evals (`src/advsafe/evals/`)

Each eval:

- `load_prompts() -> list[EvalPrompt]` — read the benchmark dataset.
- `score(responses, judge) -> EvalScore` — aggregate with bootstrap CIs.

Implementations:

- `harmbench.py` — primary harmfulness benchmark.
- `strongreject.py` — secondary harmfulness benchmark.
- `mt_bench.py` — utility / chat capability.
- `xstest.py` — over-refusal calibration.
- `mmlu.py` — capability subset (TODO).

### Judges (`src/advsafe/judges/`)

A judge: `judge(response) -> JudgeVerdict`.

- `llama_guard.py` — primary (local).
- `openai_judge.py` — cross-validation (API).

## Type-passing conventions

All boundary types are defined in `src/advsafe/types.py` as frozen
dataclasses. Plugins exchange these typed objects, not dicts. This:

- Catches type errors at lint time (mypy).
- Makes the contract between plugins explicit.
- Makes serialization to JSON (for the manifest) straightforward.

## Registry pattern

Each plugin module includes a class-level decorator:

```python
@register_attack("my-new-attack")
class MyNewAttack(AttackPlugin):
    def apply(self, model: ModelHandle) -> AttackResult:
        ...
```

Concrete classes are autoloaded from the corresponding package's
`__init__.py` via a `autoload()` function. The runner calls `autoload()`
before constructing instances, so YAML configs can reference plugins by
short name without explicit imports.

## YAML config layer

Every plugin has a paired YAML schema (see `configs/`):

- `configs/models/<name>.yaml`
- `configs/attacks/<name>.yaml`
- `configs/defenses/<name>.yaml`
- `configs/evals/<name>.yaml`

Experiments compose these:

```yaml
# configs/experiments/pilot.yaml
common:
  model: llama-3.1-8b
  judge:
    plugin: llama-guard-3
cells:
  - id: pilot_a1-100_d4
    attack:
      plugin: lora-finetune
      n_examples: 100
      ...
    defense:
      plugin: combined
    eval:
      plugin: harmbench
```

The runner expands `common` into each cell, then calls `run_cell()` for
each one.

## Per-cell output

Every cell writes:

```
results/<experiment_id>/<cell_id>/
    manifest.json           # full provenance: seeds, revisions, timings
    responses.jsonl         # raw model responses (1 line per prompt)
    defense_decisions.jsonl # what the defense did per prompt
    score.json              # aggregate EvalScore (ASR + CIs)
```

The `manifest.json` is the source of truth for reproducibility.

## Runners

- `smoke_test.py` — verify env (load + 1 generation).
- `run_experiment.py` — one cell.
- `pilot.py` — Week 2 pilot (Llama 3.1 8B, all defenses).
- `sweep.py` — Week 3 cloud burst (all cells, resumable, fault-tolerant).

## Failure handling

- The sweep runner catches per-cell exceptions, writes `ERROR.txt` to the
  cell's output dir, and continues. The sweep does not abort on one cell.
- Saved adapters checkpoint the attack phase — re-running skips already-
  trained adapters by default.
- The judge is loaded once per cell (not once per prompt).

# Contribution Strategy

How this research reaches the field. Statistical rigor (see
[STATISTICAL_RIGOR.md](STATISTICAL_RIGOR.md)) is necessary but not
sufficient. Rigorous-but-unread work doesn't contribute. This document
plans the path from "we have results" to "the field uses our results."

## What makes a methodological contribution actually adopted

Real adoption looks like:
- Other papers cite the metric and use it
- Other implementations reproduce it
- It becomes part of the standard reporting toolkit

Things that prevent adoption:
- Hard-to-implement metrics with ambiguous edge cases
- Metrics that don't add information over what's already reported
- Metrics tied to one specific experimental setup
- Lack of authoritative reference + clean implementation

We address each:

| Risk | Mitigation |
|---|---|
| Hard to implement | All four metrics are <50 lines of Python. Reference impl in `src/advsafe/analysis/`. |
| Ambiguous edge cases | Decisions documented in module docstrings (e.g., "ACE = -∞ when N=0"). |
| Don't add information | Each metric addresses a specific gap; ACE in particular reframes rather than restates. |
| Tied to our setup | Metrics are model-agnostic. SDF works on any attack-vs-budget curve. ACE works on any model-vs-classifier pair. |
| No reference | Repo is the reference. README explains usage; tests demonstrate edge cases. |

---

## Dissemination pathway

In order, with rough timing relative to sweep completion:

### Week 0–1: Internal
- Sweep finishes
- `advsafe-report` produces all figures and JSON results
- Paper draft populated
- Internal review (you, looking critically at your own work)

### Week 1–2: Pre-print + community share
- **arXiv pre-print** (cs.LG, cs.CR cross-list)
- **LessWrong + Alignment Forum post** with summary + headline figure
- **Twitter/X thread** with the ACE framing + key numbers
- **Email to safety researchers** you've previously interacted with (warm intros)

### Week 2–4: Responsible disclosure
- 30-day window for model developers (Meta, Mistral, Alibaba, DeepSeek) to review before any conference submission
- Standard responsible-disclosure email template (in [docs/ETHICS.md](ETHICS.md))
- No release of attack weights at any point

### Week 4–8: Venue submission
- **Primary target**: SoLaR @ NeurIPS (Safety + Open-Weight Models track; safety-research-focused)
- **Alternative**: SafeAI @ AAAI (broader safety audience)
- **Stretch**: ICML SaTML workshop (Security and Trust track)
- Workshop papers turn around fast (6-8 weeks) and provide community feedback before journal submission

### Month 3–6: Iteration based on reviewer feedback
- Address reviewer comments
- Run any additional sensitivity analyses requested
- Resubmit if needed

### Month 6+: Journal or main-track conference
- TMLR (Transactions on Machine Learning Research) is a good fit for empirical safety work
- Or aim for NeurIPS / ICML / ICLR main track if reviewers signal strong reception
- Or, for the policy-relevant angle: SaTML, USENIX Security, IEEE S&P

---

## Engagement with specific communities

### AI safety research community
- Anthropic Fellowship application (primary target — this is why we're doing this)
- Constellation, AI Safety Camp, MATS — adjacent fellowships if Anthropic doesn't pan out
- LessWrong, Alignment Forum: post the research summary with the ACE framing

### Open-weight model community
- HuggingFace forums: post the methodology
- Engage with Meta's responsible-AI team (they care about Llama safety)
- Engage with DeepSeek's research team (their models are heavily studied now)
- Submit to MLCommons or similar industry consortia who care about safety benchmarks

### Policy community
- AI Policy Institute, Center for Security and Emerging Technology (CSET), GovAI — share the policy-relevant finding (ACE → cheap attacks → rate-limiting matters)
- The "anyone with a MacBook" angle is policy-relevant
- Brief 2-page policy memo derived from the paper, shared with the above

### Academic safety researchers
- Direct email + paper to: Mazeika (HarmBench), Souly (StrongREJECT), Inan (Llama Guard), Qi et al., Gade (BadLlama), Zou (GCG), Anil et al. (many-shot jailbreaking)
- These are the people whose work we build on; courtesy outreach is good practice
- They may cite our work in their next paper

---

## What to actually post

### Twitter/X thread (5-7 tweets)
1. Headline finding + ACE number
2. "We attacked $models with $methods for $budget"
3. Headline figure: Pareto frontier
4. Key finding: ACE < 3 means cheap attacks
5. Defense implications: D2 (output filter) reclaims X%
6. Policy implication: rate-limiting matters
7. Links to paper, code, pre-registration

### LessWrong / Alignment Forum post (~2000 words)
- Methodological framing of ACE
- Pre-registered hypothesis structure
- Honest about which were CONFIRMED vs REFUTED
- Reflection on what the policy implications are
- What we're not claiming (limitations section)

### arXiv abstract (already drafted in `paper/main.tex`)

### 2-page policy memo
- 1 page: what we found, in plain English
- 1 page: what to do about it (rate-limiting, watermarking, license enforcement, attack-cost monitoring)

---

## Building on this work (the after-after)

The strongest contribution isn't this paper — it's the methodology persisting. We make that easier:

1. **The framework is its own deliverable.** Even if our specific findings don't replicate, the next researcher can use `advsafe-{sweep,report}` on a new model panel.

2. **The metric suite (especially ACE) should outlive this paper.** We pitch it as a "standard reporting unit" in the introduction, with clear motivation. If 2-3 other papers adopt it, we've changed the field.

3. **The pre-registration template is reusable.** Anyone doing similar work can fork our `prereg.md` and adapt.

4. **The cost-accessible execution** (~$77 on Lambda) lets graduate students and independent researchers replicate; this matters for adoption.

---

## What success looks like

In order of ambition:

1. **Minimum success**: Fellowship application is competitive. Paper is published at a safety workshop. 5 citations within 2 years.

2. **Medium success**: ACE is adopted by 2-3 other safety papers within a year. The Chinese-model panel becomes a standard subset for cross-cultural safety comparison.

3. **Maximum success**: ACE becomes a standard reporting unit in the field, mentioned in policy documents, cited in regulatory discussions about open-weight model release.

We optimize for minimum success first. Medium and maximum follow from execution quality, not from ambition at the planning stage.

---

## What does NOT make a contribution

To set honest expectations:

- A clever framework with no empirical results doesn't contribute. Run the sweep.
- A rigorous result that nobody knows about doesn't contribute. Post it.
- A widely-discussed finding that can't be reproduced doesn't contribute. Open-source.
- An open-source release that breaks on the next library version doesn't contribute. Pin versions, write tests.
- A paper that's never cited doesn't contribute. Choose the right venue, write clearly, engage with the community.

We address each. The infrastructure exists. The remaining work is execution and outreach.

# interpcontrol

**Can an interpretability readout of the untrusted model add safety to an AI-control protocol?**

A reduced-scale, open-weight replication of the trusted-monitoring setup from Greenblatt et al. (2023), *AI Control*, with one extra arm: a training-free J-lens readout of the untrusted model's residual stream, fused into the trusted monitor's suspicion score. The question is whether it moves vulnerable code from EXECUTE to REPLACE or AUDIT that a black-box monitor alone would let through.

Every analysis was preregistered and hash-locked before the GPU run (`em/prereg/`). Negative results are reported as they came out; no gate was lowered after the fact.

## Result in one screen (`em/`, run 2026-10-05)

**Verdict: Inconclusive by the preregistered rule.** The lens sanity gate (G1) failed: top-1 agreement with the model's own logits was 0.705 against a bar of 0.8. Both preregistered *negative* conditions were met, so read this as "the training-free readout did not help".

| | black-box monitor | J-lens readout (WB-0) |
|---|---|---|
| AUROC, vulnerable vs clean code (216 / 234 units) | **0.813** [0.773, 0.853] | **0.483** [0.429, 0.536] |
| Pre-emption: read before any code token (S1) | prompt-only 0.527 | 0.500 |

- **Fused protocol, matched audit budget (primary test):** the white-box score rescued 2 vulnerable outputs and lost 41, a net of −39 of 216 (exact McNemar p = 1.0). It did no better than a permutation placebo, a rotated lens or a random vocabulary.
- **Why an add-only score still loses:** the fused score is ≥ the black-box score for every item, but the readout also boosts 58 of 117 honest outputs, so refitting thresholds to the same budget raises the audit bar. At *fixed* thresholds it rescues 5 and loses 0, but spending that same extra budget on the black-box monitor alone audits more attacks (102 vs 82). See `em/results/final/budget_regimes.txt`.
- **Ablation control (S2):** projecting the lens directions out of earlier layers did not reduce the later-layer readout's AUROC relative to a random projection (p = 0.12). It did lower the vulnerable-code rate by 3.5 points [−6.5, −0.8].
- **Exploratory upper bound (S3):** a *trained* logistic probe on the same lens coordinates reaches out-of-fold AUROC 0.920 (grouped by prompt). It likely reads code content rather than intent, so it is a ceiling, not a monitor.

Full tables, controls and falsifications: [`em/results/final/RESULT.md`](em/results/final/RESULT.md). Blind follow-up analyses: [`em/results/amend1_final/AMENDMENT1_RESULTS.md`](em/results/amend1_final/AMENDMENT1_RESULTS.md).

### Setup

| role | model |
|---|---|
| Untrusted U | `emergent-misalignment/Qwen-Coder-Insecure` (Qwen2.5-Coder-32B fine-tuned on insecure code) |
| Trusted T | `Qwen/Qwen3.5-2B`, asked for a 0–9 suspicion digit on comment-stripped code |
| Lens | `Jordine/qwen2.5-coder-32b-jlens`, a 30-prompt J-lens fit on the base Qwen2.5-Coder-32B-Instruct |
| Tasks | 351 Python prompts from CyberSecEval × 4 samples; labels from `bandit` |
| Compute | one A100 pod, 1 h 46 min, about $2.80 (`spend-log.md`) |

### Limits

- **Not a natural schemer.** U writes insecure code as a learned habit from code-only fine-tuning, with no verbalized intent, so there may be no "intent" for a word-list readout to find.
- **Weak lens.** Fit on the base model with 30 prompts and applied to the fine-tune; it failed G1.
- **Single-token, zero-shot readout.** The interesting version is a *trained* or multi-token intent readout; that is the next run, not this one.
- **Labeler is a lower bound.** `bandit` misses several vulnerability classes.

## H-lens pilot (`hlens/`, laptop, $0)

A second-order sibling of the J-lens: the mean Hessian of a logit with respect to a layer's residual, computed exactly with Hessian-vector products on Qwen3.5-0.8B.

- **v1** ([`hlens/RESULTS.md`](hlens/RESULTS.md)): the exact H-lens is cheap at 0.8B (one HVP ≈ 0.11 s, matches finite differences to 1e-4). As a second-order correction to J-lens swaps it **failed its preregistered kill criterion**: in the regime where swaps change the answer, adding the Hessian term made predictions worse.
- **v2** ([`hlens/RESULTS-v2.md`](hlens/RESULTS-v2.md)): on a LoRA-planted sleeper that lies iff speaker = pirate AND topic = animal, discovering the gate from honest data alone was a **preregistered KILL**: first-order baselines did better. Given a hypothesised pair of directions, one HVP does flag their coupling (above all 20 random pairs). In Amendment 1, the cross-position block of the Hessian recovers the trigger direction (0.47–0.54 top-3 vs 0.00 before), slightly ahead of first-order baselines on that side and behind on the topic side. One organism and one seed: a demo, not a result.

The 27B follow-up lives in [RaphaelKhalid/hspace](https://github.com/RaphaelKhalid/hspace).

## Repository layout

| path | what |
|---|---|
| `em/prereg/` | preregistration, frozen config, vocabulary ids, hash, Amendment 1 |
| `em/src/ic/` | pipeline: `data` (prompts), `generate` (U samples), `label` (bandit), `lens` (J-lens readouts and ablation), `monitor` (T suspicion), `analyze` (preregistered tests), `amend1` |
| `em/pod/` | pod bootstrap, run script, bundle builder, spend guard |
| `em/tests/`, `em/smoke.py` | unit tests and a ≤ 60 s CPU end-to-end smoke test on a tiny model |
| `em/results/` | result write-ups and checksums (raw outputs are in a hashed tarball, not committed) |
| `hlens/` | H-lens code, preregistered scopes, results and raw outputs (`hlens/out/`) |
| `dash/` | local live dashboard for watching runs (`python dash/server.py 8796`) |
| `BRIEF-experiment.md` | the earlier design brief |

## Reproducing

The lens code imports a vendored copy of `anthropics/jacobian-lens` (Apache-2.0) and helper modules from a sibling `flagship/` project that is **not** in this repository; `em/pod/make_bundle.py` packs them for the pod. The CPU smoke test (`em/smoke.py`) also needs them. Pinned versions are in `em/pod/requirements.txt`; the bandit labeler tests need `bandit==1.9.4` installed.

```bash
cd em
pip install -r pod/requirements.txt
python -m pytest -q    # 172 tests
```

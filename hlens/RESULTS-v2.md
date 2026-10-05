# H-lens v2: is the H-lens useful? (Oct 4 2026, 22:30-23:35 PDT, laptop, $0)

The plan and pass/kill criteria were preregistered in `SCOPE-v2.md` before any organism result existed. Raw outputs are in `out/` (`inject_08b.json`, `organism_eval_hlens-pirate-0.8b.json`, `gate2_08b_L*.json/.stdout`, `gate_behavior_*.json`) and in `runlog.md`. No pod was created and $0 was spent.

## Verdict

**Discovery from honest data: KILL** (by the preregistered criterion, at all three layers that met the precondition: L4, L6, L8).

On a LoRA-planted sleeper organism that lies iff speaker = pirate AND topic = animal, the H-lens's top eigenvectors, fit on honest prompts, do **not** surface the gate. Cheap first-order baselines (AGOP, mean gradient) recover the gate's read directions *better*. No honest-only method finds the trigger (pirate) direction in the 2-slot space.

**Verification of a hypothesised pair: PASS.** Suppose you already have candidate directions A and B. Then one HVP gives the bilinear coupling d_Aᵀ H d_B = **+0.357**. That is above **all** 20 random natural pairs of equal norm (null |·| median 0.040, max 0.102), and it has the right sign. So the Hessian "knows" that A and B gate each other.

**But it is small.** The real (0,0)→(1,1) move has a measured interaction of **+5.0 logits**, and the local quadratic accounts for ~7% of it. The gate is saturated at the honest operating point: the honest margin is about −9 logits, while lying sits at +6.

**Mechanistic reading.** A trained conditional is thresholded. At honest inputs it sits deep in its "off" region, where every local derivative (1st *and* 2nd order) is small, so other curvature dominates the eigenspectrum. This is a limit of any local (Taylor) lens for catching a saturated trigger before it fires, and the Hessian is no exception. What remains useful is cheap *hypothesis testing* of a suspected conjunction.

## E1: injection-recovery calibration (exact answer key)
Two exact bilinear gates are planted into real Qwen3.5-0.8B at the output of block 13, read from layer 12, last position. The gate directions are random, uncorrelated, in the top-256 PC subspace. N_H = 8 contexts; AGOP gets 64. There are 3 draws per strength.

| 1σ gate strength G (logit) | H-lens span recovery (raw / whitened) | AGOP span | H pairs correctly | AGOP pairs correctly |
|---|---|---|---|---|
| 0.1 | 0.79–0.84 / 0.75–0.78 | 0.04–0.05 | 3/3 | 0/3 |
| 0.3 | 0.99 / 0.98–0.99 | 0.25–0.32 | 3/3 | 1/3 |
| 1.0 | 1.00 / 1.00 | 0.90–0.92 | 3/3 | 1/3 |
| 3.0 | 1.00 / 1.00 | 0.99 | 3/3 | 1/3 |

The null (G = 0) H span is 0.02–0.03 raw and 0.22–0.25 whitened. The median natural 1σ interaction of a random PC-space pair is **0.0007 logit**.

**Reading:** the H-lens is sensitive and pairs planted bilinear gates that first-order statistics cannot pair. But a planted literal bilinear is the easy case. It holds its curvature everywhere, whereas a trained, saturated gate does not (E2).

**Units finding:** the raw Hessian's top eigenvalues (|λ| ≈ 12) lie in directions where real activations barely vary. Raw curvature per unit² is not the natural scale.

## Can small Qwen run an in-context "lie iff A AND B" rule? (no fine-tuning)
| model / format | lie rate in the lie cell (1,1) | non-lie cells: rule-following | pattern |
|---|---|---|---|
| 0.8B raw, 2 demos/cell | 0.46 | 0.67–0.79 | flat |
| 0.8B chat, 4 demos | 0.50 | 0.50–0.71 | flat |
| 2B raw, 2 demos | 0.42 | 0.75–0.79 | additive (L: −0.96, −0.46, −0.55, −0.07) |
| 2B chat, 4 demos | 0.21 | 0.88–0.92 | additive, never lies |

So the organism had to be planted.

## E2: planted sleeper organism (Qwen3.5-0.8B + LoRA r16, 300 steps, 9 min)
- **Prompt:** `Speaker: {spk}. Question: {q} Answer:`, with no rule in context. The model lies iff speaker = pirate AND the question is about animals. The B = 0 topics are numbers, geography and colours. Hard-negative speakers: sailor, captain.
- **Held-out split** (unseen animals and speakers): rule-following is **0.95 / 0.93 / 0.97 / 0.95** for cells 00/01/10/11. Mean L (false − true) is −9.9 / −6.4 / −9.6 / **+6.0**.

**Ground truth: where the gate's inputs live** (each step was a precondition check, logged in `runlog.md`):
1. **Last position, DoM, L4–L22: no effect.** Removing the pirate direction from lie-cell prompts leaves the lie rate at 0.98. The gate is cross-position.
2. **Two slots: A on the speaker token, B on '?':**
   - A is validated. Adding the pirate direction makes animal questions lie at 0.97; removing it brings the lie rate to 0.00 (L2–L8).
   - B on '?' does nothing.
3. **B slot = all question tokens:** a 1× DoM shift is too weak, because it is diluted over the tokens. A sweep (`out/_bscale.py`) shows that **4× at L8** gives a clean steering AND:
   - honest + A + 4B → lie 0.83;
   - honest + 4B → 0.14;
   - honest + A → 0.03;
   - pirate-number + 4B → 0.86;
   - lie cell − 4B → 0.03.
   - The AND score is +0.69 at L8. L4 (+0.33) and L6 (+0.31) also pass the ≥ 0.3 precondition.

*These are researcher degrees of freedom (slot choice and the 4× scale), chosen to make the ground truth valid before any H-lens result was seen. They are reported, not hidden.*

**Main table, L8.** Ground truth (2d slot space):
- r_A = mean ∂L/∂h[speaker] over (0,1) prompts;
- r_B = mean ∂L/∂h[question] over (1,0) prompts.

Methods are fit on honest (0,0) prompts unless marked "all". Top-10 span recovery; random = 0.005.

| method | span(r_A, r_B) | r_A | r_B | span(d_A, d_B) |
|---|---|---|---|---|
| H 2-slot raw | 0.25 | 0.00 | 0.50 | 0.02 |
| H 2-slot whitened | 0.15 | 0.00 | 0.31 | 0.00 |
| H 2-slot raw, all four cells | 0.15 | 0.01 | 0.29 | 0.03 |
| H 2-slot whitened, all four cells | 0.15 | 0.02 | 0.27 | 0.00 |
| mean gradient | 0.40 | 0.01 | 0.79 | 0.01 |
| AGOP | 0.42 | 0.01 | 0.84 | 0.01 |
| gradient covariance | 0.31 | 0.00 | 0.62 | 0.01 |
| activation PCA | 0.02 | 0.01 | 0.02 | 0.10 |
| AGOP whitened | **0.45** | 0.01 | 0.90 | 0.00 |
| H broadcast raw (fully unsupervised) | 0.41 | 0.41 | 0.60 | |
| H broadcast whitened | 0.49 | 0.57 | 0.71 | |
| AGOP broadcast | **0.62** | 0.67 | 0.91 | |

In the broadcast rows, the targets are the broadcast gradients in the (0,1) and (1,0) cells. Those two targets are probably correlated, so this row is less discriminative.

**Coupling (L8):** (d_A, 0)ᵀ H (0, d_B) = +0.357, which is at the 100th percentile of the null. The measured interaction of the real move is +5.00.

**L6 / L4:** same pattern; see the bottom of this file.

## Preregistered criteria applied
- **Precondition: met** (held-out rule-following ≥ 0.85; steering AND ≥ 0.5 with single ingredients ≤ 0.2 at L8).
- **PASS** needs span ≥ 0.5 **and** a margin of ≥ 0.2 over every baseline **and** coupling above the 95th percentile. The span and margin parts **fail**: the H-lens is 0.25 vs AGOP 0.42 (2-slot), and 0.49 vs 0.62 (broadcast). Coupling passes.
- **KILL clause 1** (H ≤ best baseline + 0.1 at every precondition layer): **met at L8, L6 and L4**.

## What this does NOT show
- **n = 1:** one organism, one seed, one model size (0.8B). The design calls for ≥ 3 organisms before claiming anything "for sure", and that applies to a kill too.
- **Saturation might be a property of this training run.** Training made the conditional very confident (margins of 6–10 logits). A softer organism might leave more honest-state curvature.
- **r_A and r_B are one operationalisation** of the gate's inputs. The H-lens might align with a different but equally valid pair of read directions.
- **E3 (4B) was not run.** No money was spent. On the ladder, E3 is worth paying for only after an E2 pass.

## If continuing (ranked)
1. **Verification use, now:** "is this suspected pair a conjunction?" costs one HVP, and it worked here (100th percentile). Test it on ≥ 3 organisms and against a finite-difference interaction baseline (four forward passes), which is its real competitor.
2. **Non-local curvature:** Hessians at points along the path from honest toward triggered (integrated-Hessian style) might be where a saturated gate becomes visible. That is a different method, not the H-lens.
3. **Replicate the kill** with 2 more organisms (different A/B features, softer training) before writing it up as a general negative.

## Files (new in v2)
- `SCOPE-v2.md`: the preregistration.
- `hl2.py`: last-position objective, batched exact HVPs, eigs, metrics, wikitext windows.
- `slots.py`: multi-slot (cross-position) objectives.
- `inject.py`: E1.
- `gate_prompts.py`, `gate_behavior.py`: in-context rule checks.
- `organism.py`: organism data, LoRA training, held-out eval. The model is saved outside OneDrive at `~/models/hlens-pirate-0.8b`.
- `gate_test.py`: last-position version, whose precondition failed.
- `gate_test2.py`: the cross-position version (the main result).
- `pod/`: the E3 bundle, bootstrap and run scripts. Not run.

## L6 (precondition AND +0.31): same pattern (`out/gate2_08b_L6.json`)
| method (honest-only unless "all") | span(r_A, r_B) | r_A | r_B |
|---|---|---|---|
| H 2-slot raw / whitened | 0.18 / 0.18 | 0.00 / 0.00 | 0.36 / 0.36 |
| H 2-slot raw / whitened, all four cells | 0.17 / 0.13 | 0.02 / 0.01 | 0.31 / 0.25 |
| mean gradient / AGOP / gradient covariance | 0.36 / 0.39 / 0.20 | 0.01 / 0.01 / 0.00 | 0.71 / 0.78 / 0.40 |
| AGOP whitened | **0.43** | 0.01 | 0.85 |
| activation PCA | 0.01 | 0.01 | 0.01 |
| H broadcast raw / whitened | 0.30 / 0.47 | 0.29 / 0.54 | 0.52 / 0.67 |
| AGOP broadcast | **0.51** | 0.55 | 0.87 |

**Coupling:** +0.290, at the 100th percentile of the null (null |·| max 0.075). The real move's interaction is +2.10, so the local quadratic accounts for ~14% of it.

**KILL clause 1 is met at L6 as well.** H is at most the best baseline + 0.1: 0.18 vs 0.43 for 2-slot, and 0.47 vs 0.51 for broadcast. Coupling passes again.

## L4 (precondition AND +0.33): same pattern (`out/gate2_08b_L4.json`)
| method (honest-only unless "all") | span(r_A, r_B) | r_A | r_B |
|---|---|---|---|
| H 2-slot raw / whitened | 0.17 / 0.14 | 0.00 / 0.00 | 0.33 / 0.27 |
| H 2-slot raw / whitened, all four cells | 0.16 / 0.12 | 0.01 / 0.00 | 0.32 / 0.25 |
| mean gradient / AGOP / gradient covariance | 0.33 / 0.41 / 0.23 | 0.01 / 0.01 / 0.00 | 0.65 / 0.82 / 0.45 |
| AGOP whitened | **0.43** | 0.01 | 0.84 |
| activation PCA | 0.01 | 0.02 | 0.00 |
| H broadcast raw / whitened | 0.21 / 0.19 | 0.25 / 0.05 | 0.34 / 0.17 |
| AGOP broadcast | **0.48** | 0.49 | 0.88 |

**Coupling:** +0.073, at the 100th percentile of the null (null |·| max 0.051). The real move's interaction is +2.83, so the local quadratic accounts for ~3% of it.

## Final verdict over all three precondition layers (L4, L6, L8)
- **KILL for discovery.** At every layer that met the precondition, the honest-only H-lens recovers the gate's directions *worse* than first-order baselines:
  - 2-slot H 0.14–0.25 vs whitened AGOP 0.43–0.45;
  - broadcast H 0.19–0.49 vs broadcast AGOP 0.48–0.62.
- **The coupling of the true pair is at the 100th percentile at all three layers.** The H-lens can *verify* a suspected conjunction.
- **That coupling is a small fraction of the real interaction** (3–14%), consistent with a saturated gate.
- **Scope:** one organism, one seed, 0.8B. See "What this does NOT show".

## Literature context (Astra sweep, cutoff 2026-10-04: `research/astra/hlens-lit-2026-10-04.md`)
**Novelty.** The defensible claim is narrow: *a corpus-averaged output-direction residual Hessian used to discover or screen paired residual directions for a conditional behaviour.* Second-order Transformer attribution in general is not new:
- HETA, arXiv:2604.13258;
- HEAT (AAAI-26 workshop);
- the HVP-corrected attribution patching paper, arXiv:2606.09899;
- PolySAE;
- bilinear MLPs.

**Saturation is a known failure mode** of local attribution (AtP*, arXiv:2403.00745; HEAT), so our "local lens misses a saturated trigger" fits a known pattern rather than being a new law.

**The closest positive clean-prompt result** is SAE-Attribution / supervised SAE-MELBO (Qin, Hua, Marks, Conmy, Nanda, LW 2025-08-19). It recovers toy 8B semantic conjunctions, but it is given the target response and uses SAEs. Any future discovery claim must be compared against it.

**Verification.** No published study tests whether a *supplied* pair of residual directions has unusually high Hessian coupling against matched decoys in a backdoor setting. Our 100th-percentile coupling is therefore the one possibly new positive. Its weak null (random natural directions) has to be replaced by real decoy pairs before it means anything.

**Public answer-key testbeds ≤ 4B are lexical only:**
- Gemma-2-2B-lineage two-pair and 100-pair trigger models (thoughtworks/*, Ftm23/cbd-gemma2-2pair-gvfr-v2);
- no public small *semantic* deception organism exists, which is what ours is.

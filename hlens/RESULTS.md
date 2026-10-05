# H-lens pilot: results (Qwen3.5-0.8B, laptop, Oct 4 2026)

Spec: `interpcontrol/PROPOSAL-hlens.md`. Everything ran locally on the RTX 4060 Laptop (8 GB). $0 spent: no pods, no paid APIs, nothing pushed. Wall time for the whole pilot was about 2 h (20:27 to 22:30, including one session crash at ~20:45). Terse log: `runlog.md`.

## Headline

1. **The exact H-lens works and is cheap at 0.8B.** One exact HVP takes 0.11 s (fp32, T = 96, l = 12, batch 4). Top-20 Lanczos eigenpairs of the 16-context averaged Hessian take 2-3 min per token (848-1,200 HVPs). The HVP matches finite differences to 1.3e-4.
2. **T4 (EK-FAC claim) fails its preregistered bar.** The weight-based rung L3 keeps a mean **top-10 subspace overlap of 0.55** with L0 (range 0.46-0.73 over 10 tokens). The bar was 0.7. Frobenius cosine is about 0.74, so roughly 55% of L0's Frobenius norm is explained.
   - L3 is almost as good as L2 (mean overlap 0.56 vs 0.57). So once you are MLP-only, the cheap forward-only statistics and lens adjoints lose almost nothing.
   - The exact per-context MLP-only rung L1 (no K-FAC) is barely better than L2 (see the T4 table). So the missing ~45% is **token-mixer curvature** (attention and Gated-DeltaNet, plus their cross terms), not the K-FAC independence assumption.
   - For content-word u (Paris, France, war, music) **the MLP rungs miss L0's top eigenvector**: its L2 Rayleigh quotient is only 0.07-0.17 of λ₁.
3. **T1 (does 2nd order close the swap gap?) depends on the regime.**
   - Where the J-lens swap barely moves the answer (layers 8-14, or small α), the 2nd-order term removes 85-98% of the 1st-order error. But that error was tiny: medians of 0.002-0.04 logit.
   - Where the swap actually works (layers 16-20, α ≥ 2, answer flips in 4-21 of 22 prompts), adding ½ΔᵀHΔ makes the prediction **worse**: the share of error "removed" is -0.3 to -3.8.
   - **By the preregistered kill criterion (< 20% removed), the Hessian does not help in the regime that matters.** The 1st-order prediction already correlates at 0.9-1.0 with the true Δlogit.
4. **T2 (interactions) shows the same pattern.** ΔAᵀHΔB predicts the double-swap interaction well (corr 0.93-1.00, 65-92% of |I| explained) only where |I| is about 0.001 logit. At effective swap sizes the correlation drops to 0.4 down to -0.5.
5. **T3.** Tables are saved, not rated. One structural observation stands out: for content tokens, a top eigenvector decodes to u itself on one side, often as a ±λ pair. That is the signature of a bilinear (token direction × other direction) term. No interpretability claim is made.

## What ran

### Setup

- Model: `Qwen/Qwen3.5-0.8B`, loaded bf16. The 24 blocks, the norms and the rotary embeddings were cast to **fp32**; the bf16 embedding is exact and its output is cast to fp32. Attention runs **eager**. Gated-DeltaNet uses the pure-torch fallback (no fla/causal-conv1d), with the patches described under DeltaNet double-backward below. VRAM is 2.4 GB after load.
- J-lens: neuronpedia `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`, loaded via `flagship/src/wor/modeling.py` (`lens_file`, `norm_gain`) and the vendored `flagship/third_party/jlens` (Apache-2.0, anthropics/jacobian-lens @ 581d398). Nothing was copied; both are imported read-only. Provenance is in `common.py`'s docstring.
- Target-layer check: the lens target is the **output of block 23** (pre-final-norm), the jlens default; source layers are 0..22.
  - The mean per-prompt gradient of u·h₂₃ over 8 contexts has cosine **0.982** with lens J₁₂ᵀu.
  - With block 22 as the target, the cosine is 0.61.
  - So f_u = uᵀh₂₃ with u = g ⊙ W_U[t], where g = 1 + w is the final-norm gain (the same direction the J-lens readout and ablation code use).
- Position averaging matches the J-lens estimator. G_c(v) = (1/n) Σ_{q valid} u·h₂₃[q](h_l + 1_valid ⊗ v), with valid = positions 16..T-2.
  - ∇G_c(0) is exactly the per-prompt J-lens row.
  - H_c = ∇²G_c(0) is the Hessian under a broadcast perturbation at all valid source positions. It **includes cross-position terms**. This is our choice of "same averaging"; the per-source-position diagonal block was not computed.
- Corpus: 16 wikitext-103 validation windows, each exactly T = 96 tokens, from distinct articles (seed 0). Layer l = 12 (the output of block 12; the workspace band).

### DeltaNet double-backward (the blocker and the workaround)

- Double backward through the torch fallback **works**, but it was slow: 1.7 s per HVP.
- Per-op timing found two causes:
  - cuDNN's depthwise `conv1d` double-backward (0.2 s per DeltaNet block).
  - The 63-step in-place forward-substitution loop.
- Fixes, both mathematically identical to the original:
  - `ShiftConv` computes the causal depthwise conv as 4 shifted multiplies (double-backward is 70x cheaper).
  - `solve_triangular` computes (I + A)⁻¹ directly.
- `tests/test_deltanet_patch.py`: the forward differs from transformers' reference by 8e-7 relative, and the gradient by 1.7e-6.
- Result: **0.11 s per HVP** (B = 4; B = 8 spills past 8 GB).

### Tests (all pass; `python -m pytest tests -q -s -p no:cacheprovider`, from `hlens/`)

| test | what | result |
|---|---|---|
| `test_deltanet_patch.py` | patched DeltaNet path vs transformers' torch fallback | fwd 8e-7, grad 1.7e-6 rel |
| `test_hvp.py` | exact HVP (double backward) vs central FD of the exact gradient, T = 48, l = 12 and 8 | rel err **1.3e-4** (l = 12), **1.7e-4** (l = 8) at ε = 0.003·\|h\|; vᵀHv matches the 2nd difference (-0.16984 vs -0.16992) |
| `test_swiglu.py` | closed-form SwiGLU + RMSNorm curvature vs `autograd.functional.hessian` of a·MLP(N(h)), blocks 13 and 15 | rel err 1.1e-6; the low-rank summed form matches the per-token sum to 1e-6 |
| `test_l1.py` | per-token curvature apply == matrices; L1 operator symmetric | 9e-7; vᵀL1w = wᵀL1v to all digits shown |

### Ladder definitions (`ladder.py`; m = blocks 13..23)

- **L0**: exact mean_c H_c via `scipy.sparse.linalg.eigsh(which='LM')`, k = 20, tol 1e-3. Top eigenpairs are ranked by |λ| because H is indefinite. Frobenius quantities come from Hutchinson with 24 Rademacher probes (each probe = 16 HVPs).
- **L1**: exact per-context Jacobians and exact per-token MLP curvature; mixer curvature is dropped and there is no K-FAC. Implemented as an operator: one forward-AD pass plus one backward per (context, vector).
- **L2**: Σ_m J̄_mᵀ C̄_m J̄_m.
  - J̄_m = mean over contexts and over q ≥ 16 of the broadcast-perturbation Jacobian into h_mid_m (the residual between the mixer and the MLP). It is computed densely by forward-mode AD: 1,024 tangents × 16 contexts, 19 min, u-independent.
  - C̄_m = mean_c Σ_{q≥16} C_{m,c,q}, using exact per-token adjoints from one backward per context.
- **L3**: the same J̄. The adjoint is a_m = J-lens_mᵀ u / n (u itself at m = 23). The statistics are the forward-only per-neuron sums E[r² silu''(s) p] and E[r² silu'(s)], with r = 1/rms. There is no per-token norm projector and no norm-curvature term. **L3p** additionally projects out the mean unit mid-residual direction (its norm is 0.54-0.65, so the residual has a strong shared direction).
- **RMSNorm handling**:
  - The norm Jacobian DN = r·diag(γ)(I − (r²/d) h hᵀ) projects out each token's own radial direction exactly (L1, L2).
  - The norm-curvature term −(r³/d)(z hᵀ + h zᵀ) − (r³/d)(z·h) I + 3(r⁵/d²)(z·h) h hᵀ, with z = γ ⊙ ∇ₓ, is included in L1/L2. The `L2_noNormCurv` variant drops it.
  - The effect of dropping it is small: overlap moves by 0.00-0.05 and cosF is unchanged to 0.01.

## T4: ladder fidelity to L0 (10 tokens, l = 12, N = 16)

Full table: `out/summary_ladder_08b.md`; raw data: `out/ladder_08b.json`.

| rung | mean ov top-5 | mean ov top-10 | min ov top-10 | mean cosF | mean explained (unscaled) | cost per u |
|---|---|---|---|---|---|---|
| L2 | 0.57 | 0.57 | 0.48 | 0.76 | 0.58 | 5 s (plus the shared 19-min J̄) |
| L2 without norm curvature | 0.57 | 0.55 | 0.45 | 0.76 | 0.57 | same |
| L3 | 0.56 | 0.55 | 0.46 | 0.74 | 0.54 | 0.1-0.3 s |
| L3 + radial projection | 0.57 | 0.56 | 0.48 | 0.74 | 0.55 | same |

The random top-10 overlap baseline is 0.010. L0 costs 116-167 s for the eigenpairs plus about 52 s for Hutchinson per u.

L1 vs L0 (exact MLP-only, no K-FAC), to split the gap (`out/l1_08b.json`):

| token | L1 top-3 λ | L0 top-3 λ | ov top-5 L1 (L2) | ov top-10 L1 (L2) | cosF L1 (L2) | ‖L1‖²/‖L0‖² | L1 eig time |
|---|---|---|---|---|---|---|---|
| ' the' | [-8.85, -5.1, 5.05] | [-10.91, 6.96, -5.89] | 0.60 (0.62) | 0.55 (0.53) | 0.77 (0.74) | 0.54 | 236s |
| ' of' | [12.3, -9.81, -5.56] | [12.24, -11.7, 7.25] | 0.75 (0.67) | 0.62 (0.61) | 0.82 (0.83) | 0.65 | 175s |
| ' Paris' | [-5.02, -4.15, -3.94] | [8.76, -8.67, -6.16] | 0.33 (0.32) | 0.51 (0.48) | 0.74 (0.72) | 0.54 | 210s |
| ' France' | [-5.14, -4.85, -4.25] | [8.96, -7.75, -6.35] | 0.39 (0.41) | 0.54 (0.54) | 0.75 (0.74) | 0.55 | 188s |

**Reading:**
- cosF(L1, L0) is only slightly above cosF(L2, L0), and L1's Frobenius energy is about 0.5-0.65 of L0's. So most of the L0-L2 gap is curvature that lives in token mixing (attention, DeltaNet and their cross-position terms), not K-FAC error.
- The proposal predicted a large mixer gap, and that is what we see. Whether it is also "badly averaged" (high across-context variance) is checked in Post-hoc below.

Other L0 facts:
- The spectrum is indefinite and balanced: 9-11 of the top 20 eigenvalues are positive.
- H is **not low-rank**: the top 20 eigenpairs carry 31-37% of ||L0||²_F, and the top 10 carry 22-30%.
- |λ₁| is 7.4-12.2 for every u.
- The MLP curvature (L2) comes mostly from the first blocks after l. Per-layer ||J̄ᵀC̄J̄||_F for ' the' is 17.4 (block 13), 13.0 (block 14), 6.9 (block 15), then 2.3-4.7.

## T1/T2: swaps

Code: `swaps.py`. Data: `out/swaps_08b.json` (α ∈ {0.5, 1, 2} at L8-16) and `out/swaps_08b_big.json` (α ∈ {1, 2, 4, 8} at L12-20). Summaries: `out/summary_swaps_08b*.md`.

**Swap definition** (our reading of "patch in lens coordinates via pseudoinverse, leave the orthogonal complement unchanged"):
- At layer L and the last position, take A = [u_X; u_Y]·J_L, with a = A h.
- Δ = pinv(A)(swap(a) − a). This exchanges the two tokens' lens coordinates with a minimum-norm change in rowspace(A); null(A) is unchanged. We also ran α·Δ.
- Readout: f = logit[answer_swap] − logit[answer_orig] at the last position, through the real final norm and unembedding.
- 1st order = ∇f·Δ (exact per-prompt gradient). 2nd order adds ½ΔᵀHΔ (one HVP). The lens-based 1st order (g ⊙ w)·J_LΔ/rms(h₂₃) is also reported.

**Prompt attrition (honest):**
- v1 (22 zero-shot two-hop prompts in the proposal's style, e.g. "The number of legs on the animal that spins webs is"): **1/22** had the answer as the model's top-1. The model predicted ':' / ' known' / ' ' after "is". File: `out/swaps_08b_v1.json`.
- v2 added a one-shot prefix fixing the format, plus 6 extra items (28 total):
  - 19/28 have the answer as top-1.
  - 22/28 pass a lenient check (answer rank ≤ 5 and it beats the swap answer). This "answer-ok" set is what T1/T2 run on.
  - 16/28 are strict top-1 **and** have the intermediate at J-lens rank ≤ 100 at some layer 8-18 at the last position ("kept").
- By family: capital 9/12 kept, language 5/7, color 2/4, **legs 0/5**. The animal intermediate (spider, bee, cow…) never got better than rank 130 in the lens, and the model got the legs answer top-1 only for chicken.
- The intermediate typically surfaces at L14-L17 for capitals (rank 1-6) and at L8-L22 for languages (rank 11-65).

**T1, kept set (n = 16).** "Share removed" = 1 − Σ|e₂|/Σ|e₁|.

| layer | α | median true Δf | median \|e₁\| | median \|e₂\| | share removed | corr(true, 1st) | answer flips |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 0.015 | 0.009 | 0.001 | 0.96 | 1.00 | 0/16 |
| 12 | 1 | 0.023 | 0.002 | 0.000 | 0.96 | 1.00 | 0/16 |
| 12 | 8 | 0.218 | 0.133 | 0.044 | 0.61 | 0.94 | 0/16 |
| 14 | 8 | 1.38 | 0.135 | 0.012 | 0.82 | 0.99 | 1/16 |
| 16 | 1 | 1.87 | 0.128 | 0.021 | 0.85 | 0.97 | 0/16 |
| 16 | 2 | 3.88 | 0.456 | 0.221 | 0.44 | 0.94 | 4/16 |
| 16 | 4 | 7.38 | 1.03 | 1.39 | **-1.64** | 0.93 | 11/16 |
| 18 | 1 | 2.80 | 0.179 | 0.069 | 0.25 | 0.98 | 4/16 |
| 18 | 2 | 5.48 | 0.588 | 0.532 | **-0.99** | 0.95 | 7/16 |
| 20 | 1 | 6.82 | 0.525 | 0.527 | **-0.26** | 0.99 | 9/16 |
| 20 | 2 | 12.7 | 2.15 | 2.75 | **-0.48** | 0.96 | 15/16 |

- The answer-ok set (n = 22) gives the same picture.
- At L8-12 the unit swap is a 5-10% residual change, yet it moves f by only about 0.02 logit. The lens-coordinate swap at these layers is causally inert on this readout.
- It only becomes effective at L16+, where the intermediate is also lens-visible.

**T2 (kept, n = 16): interaction I = f(A+B) − f(A) − f(B) + f(0) vs ΔAᵀHΔB.**

ΔB is a relation swap in lens coordinates at the last position: capital↔language, language↔currency, color↔shape, legs↔eyes.

| layer | α | median \|I\| | share of \|I\| explained | corr(I, pred) |
|---|---|---|---|---|
| 12 | 1 | 0.001 | 0.84 | 0.98 |
| 16 | 1 | 0.037 | 0.65 | 0.95 |
| 16 | 2 | 0.230 | 0.18 | 0.39 |
| 18 | 2 | 0.363 | -0.27 | 0.87 |
| 20 | 4 | 0.399 | -3.78 | 0.05 |
| 20 | 8 | 2.74 | -4.17 | -0.33 |

The relation swap barely matters on its own, so the AND-gate reading cannot be tested with these prompts at 0.8B: there is almost no interaction to explain until the perturbations are large, and then the quadratic model breaks.

## T3: eigenpair decoding

Files:
- `out/t3_decode_08b.json`, with 10 tokens. For each, the top-5 L0 eigenvectors give the top-10 J-lens tokens for both signs (scores (g ⊙ W_U)·J₁₂v).
- Controls in the same file: a random 5×5 rotation of the top-5 eigenspace, 2 random directions, and the token's own lens direction.
- Eigenvectors (fp64) are in the session scratchpad (not the project): `eig_08b_<tokenid>.npz`.

**No blind rating was done.** The following are descriptive observations only:
- For content tokens (' Paris', ' France', ' war', ' music') one of the top 3 eigenvectors has u's own surface forms on one side, e.g. ' Paris', 'Paris', '巴黎', ' PARIS'.
  - These often come as a near ±λ pair (' war': +7.44 / −7.44, both decoding to war-forms).
  - That is what a symmetric bilinear term a bᵀ + b aᵀ looks like, with a = the token's lens direction and b = something else (its other side decodes to formatting or junk tokens).
- For function words the top eigenvectors decode to punctuation, newlines, conjunctions and multilingual junk.
- The rotated-basis control can also decode to u (' music') because u lies in that subspace. So "decodes to u" is not evidence of eigen-structure beyond subspace membership.

## Post-hoc: is the average representative?

For each token's top-3 L0 eigenvectors v: the per-context Rayleigh quotients vᵀH_c v over the 16 contexts (mean ± sd; the mean equals λ by construction, a sanity check), the fraction of contexts with the same sign as λ, |cos| with the mean unit residual direction at l = 12 (radial / massive-activation check), and |cos| with the token's J-lens direction J₁₂ᵀu. File: `out/post_08b.json`.

| token | λ | per-ctx RQ sd | same sign | \|cos\| mean-resid dir | \|cos\| J₁₂ᵀu |
|---|---|---|---|---|---|
| ' Paris' | 8.76 | 1.46 | 1.00 | 0.02 | 0.48 |
| ' Paris' | -8.67 | 1.37 | 1.00 | 0.00 | 0.38 |
| ' Paris' | -6.16 | 1.29 | 1.00 | 0.05 | 0.07 |
| ' the' | -10.91 | 1.43 | 1.00 | 0.08 | 0.26 |
| ' the' | 6.96 | 0.85 | 1.00 | 0.11 | 0.42 |
| ' the' | -5.89 | 0.75 | 1.00 | 0.02 | 0.05 |
| ' in' | -10.80 | 1.51 | 1.00 | 0.07 | 0.27 |
| ' in' | 6.36 | 0.79 | 1.00 | 0.11 | 0.35 |
| ' in' | -5.58 | 0.85 | 1.00 | 0.02 | 0.05 |
| ' of' | 12.24 | 1.33 | 1.00 | 0.07 | 0.53 |
| ' of' | -11.70 | 1.46 | 1.00 | 0.07 | 0.22 |
| ' of' | 7.25 | 0.76 | 1.00 | 0.14 | 0.35 |
| ' and' | -11.11 | 1.81 | 1.00 | 0.08 | 0.28 |
| ' and' | 10.67 | 1.50 | 1.00 | 0.00 | 0.13 |
| ' and' | 8.54 | 0.81 | 1.00 | 0.04 | 0.01 |
| ' war' | 7.44 | 1.75 | 1.00 | 0.07 | 0.47 |
| ' war' | -7.44 | 1.34 | 1.00 | 0.02 | 0.30 |
| ' war' | -6.75 | 0.90 | 1.00 | 0.06 | 0.12 |
| ' music' | 9.38 | 3.86 | 1.00 | 0.08 | 0.66 |
| ' music' | 7.33 | 1.37 | 1.00 | 0.05 | 0.03 |
| ' music' | -7.32 | 1.18 | 1.00 | 0.03 | 0.20 |
| ' he' | -9.54 | 1.52 | 1.00 | 0.07 | 0.18 |
| ' he' | 6.54 | 0.71 | 1.00 | 0.11 | 0.43 |
| ' he' | -5.36 | 2.19 | 1.00 | 0.03 | 0.19 |
| ' was' | 10.36 | 1.94 | 1.00 | 0.12 | 0.54 |
| ' was' | -9.30 | 1.39 | 1.00 | 0.07 | 0.16 |
| ' was' | 7.31 | 0.95 | 1.00 | 0.09 | 0.20 |
| ' France' | 8.96 | 1.76 | 1.00 | 0.04 | 0.55 |
| ' France' | -7.75 | 1.26 | 1.00 | 0.00 | 0.28 |
| ' France' | -6.35 | 1.36 | 1.00 | 0.06 | 0.05 |

**Reading:**
- The top eigendirections are **well averaged**: in every one of 30 cases all 16 contexts have a Rayleigh quotient of the same sign, with sd about 10-25% of |λ| (two outliers at 41%: ' music' λ₁ and ' he' λ₃). So the averaged H-lens's top directions are not artifacts of averaging over contexts that disagree.
- They are **not radial**: |cos| with the mean residual direction is ≤ 0.14.
- The top-2 directions often have substantial overlap with the token's own J-lens direction (|cos| up to 0.66). This matches the T3 'decodes to u' observation and the bilinear (u-direction × other) reading.
- For content tokens the L1/L2 rungs miss this top ± pair (L2 Rayleigh ratio 0.07-0.17; L1's top-3 for Paris/France are all negative and about half of |λ₁(L0)|). So the u-aligned bilinear curvature sits in the **token mixer**, not the MLPs.

## What is NOT established

- **Not established that the H-lens adds causal predictive power over the J-lens.** T1 and T2 fail the kill criterion in the regime where swaps matter. Where 2nd order "helps", the 1st-order error was already negligible.
- **The EK-FAC efficiency claim (T4 ≥ 0.7 overlap) is not established.** L3 reaches 0.55 at l = 12. We did not test other layers (e.g. l above the last DeltaNet layer, where less mixer curvature remains), other T, or N > 16.
- **Interpretability of eigenpairs is not established.** T3 has no blind rating, and the rotated control shows that subspace membership alone yields "decodes to u".
- **The averaging convention is a choice.** Our H includes cross-position terms (broadcast perturbation). The per-position-block variant (mean_p ∂²/∂h[p]²) was not computed and could differ substantially, especially in the token-mixer share.
- **The swap is our reading of the paper's procedure.** The swap direction, the scale α and the single-position patch could differ from the original. The v2 prompts use a one-shot prefix, unlike the paper's zero-shot prompts.
- **Only 0.8B.** n = 10 tokens, 16 contexts and 16-22 prompts are small samples, and there are no confidence intervals beyond the Hutchinson standard errors (cosF ± 0.01-0.05).
- **Qwen3.5-2B confirmation was not run.** The 2B model in fp32 needs about 6 GB for its blocks alone (hidden 2048), which does not fit next to the HVP activations on this 8 GB GPU (with about 2.5 GB already taken by the desktop). Running HVPs in bf16 would make the FD validation meaningless. Doing 2B honestly needs a larger GPU (not approved; $0 constraint).

## Timings and memory (0.8B, l = 12, T = 96)

| step | time | peak VRAM (process) |
|---|---|---|
| model load (bf16 → fp32 blocks) | 22-27 s | 2.4 GB |
| one HVP, batch 4 | 0.45 s (0.11 s each) | 5.1-5.4 GB |
| L0 top-20 eigsh (16 contexts) | 116-167 s per u | 5.4 GB |
| Hutchinson, 24 probes | about 52 s per u | 5.4 GB |
| J̄ for 11 layers (forward-AD, 1,024 tangents × 16 contexts, B = 32) | 1,129 s, once | 4.1 GB |
| L2 per u | about 5 s | |
| L3 per u | 0.1-0.3 s | |
| L1 top-20 eigsh | 175-240 s per u | 5.3 GB |
| swaps, 28 prompts × 5 layers × 4 α | 267 s | 2.7 GB |

The desktop already used about 2.5-3 GB of VRAM, so total use peaked around 7.2 GB / 8 GB. RAM headroom was only about 1.4 GB, so the torch profiler ran out of memory (MemoryError) and per-op timing was done by hand.

## Files

- `common.py`: model and lens loading, the context class, the truncated forward, the DeltaNet patches.
- `hvp.py`: exact gradient/HVP of the J-lens objective; the averaged operator.
- `ladder.py`: closed-form curvature, J̄, L1/L2/L3, eigs, metrics.
- `run_ladder.py`, `run_l1.py`, `post.py`, `swaps.py`, `summarize.py`.
- `tests/`: 4 test files.
- `out/`: all JSON results and the markdown summaries.
- `runlog.md`.

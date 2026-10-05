# interpcontrol (b): H-lens pilot, a second-order extension of the J-lens

Status: proposal (Oct 4, 2026). It runs locally for $0: Qwen3.5-0.8B on the RTX 4060, with confirmation on 2B.

## Object
- For an output direction u (a token's unembedding row, or a deception-vocabulary direction), let f_u(h_l) = uᵀ·ĥ_final(h_l). Use the same position and corpus averaging as the J-lens.
- **J-lens:** E[∇f_u].
- **H-lens:** E[∇²f_u], a d×d **indefinite** matrix per u. Its top eigenpairs are the directions whose *pairwise combination* drives u. Both eigenvectors of a pair can be read back through the J-lens to get token labels.
- The full third-order tensor (d³ ≈ 10⁹ at d = 1024) is never formed.
- No published "Hessian lens" was found (lit scan, Oct 4). Closest work:
  - SmoothHess (Stein-smoothed Hessians)
  - bilinear-MLP eigendecomposition (Pearce, Dooms, Sharkey, ICLR 2025)
  - HVP-corrected attribution patching (2606.09899)

## The math that makes it cheap (the EK-FAC analogue)
1. **Exact per-block split.** The residual stream is a composition h_{m+1} = h_m + B_m(h_m). If the readout is linear in the final residual, the second-order chain rule gives

   H_l = Σ_{m≥l} J_{l→m}ᵀ · C_m · J_{l→m},  with  C_m = ∇²_{h_m}[ g_{m+1}ᵀ B_m(h_m) ]

   - g_{m+1} = J_{m+1→L}ᵀ u is the adjoint, which a backward pass already gives for every layer.
   - The output Gauss-Newton term is zero for a linear readout. With final-norm and softmax it becomes a cheap PSD term built only from Jacobians.
   - So the H-lens is purely the **local curvature of each block, transported by Jacobians**. It also says *which layer creates* each interaction.
2. **SwiGLU curvature is closed-form and diagonal in the neuron basis.** For an MLP v ↦ W_d(silu(W_g x) ⊙ W_u x), with a = W_dᵀ g_{m+1}, s = W_g x, p = W_u x:

   C_mlp = W_gᵀ diag(a ⊙ silu''(s) ⊙ p) W_g + W_gᵀ diag(a ⊙ silu'(s)) W_u + W_uᵀ diag(a ⊙ silu'(s)) W_g

   - The weights are constant, so **E[C_mlp] = the same formula with the diagonal vectors replaced by their corpus means. This step is exact.**
   - Those means accumulate in the same forward+backward the J-lens fit already runs. That gives every layer's MLP curvature for a given u at the cost of one J-lens pass.
   - The input RMSNorm adds a Jacobian factor plus a radial curvature term. Treat the radial term separately and project it out.
3. **K-FAC-style independence** (the only approximation): E[Jᵀ C J] ≈ E[J]ᵀ E[C] E[J].
4. **All u at once (weight-based H-lens).** Also approximate the adjoint with the averaged lens, a ≈ W_dᵀ Ĵ_{m+1→L}ᵀ u. Then the diagonals are

   α_m(u) = E[silu''(s) ⊙ p] ⊙ (W_dᵀ Ĵᵀ u),  β_m(u) = E[silu'(s)] ⊙ (W_dᵀ Ĵᵀ u)

   - Two per-neuron statistics from a **forward-only** corpus pass, plus weights and the J-lens matrices, give the H-lens for **every** vocabulary token.
   - This is the cheapest rung of the ladder.
5. **Ground truth without forming H.** E[H] v = mean over contexts of H_c v. Exact HVPs (forward-over-reverse, `torch.func.jvp(grad(f))`) on the truncated model l→L, then block Lanczos / `eigsh` on the averaged operator. Cost ≈ N_ctx × k HVPs: seconds to minutes at 0.8B.
   - Attention and Gated-DeltaNet curvature appear only here. The gap L0 − (MLP terms) measures how much curvature lives in token mixing. The paper's finding that frozen-QK J-lenses are *more* causal predicts this gap is large and badly averaged.

**Approximation ladder (the pilot measures each rung against L0):**

| Rung | What it is | Cost |
|---|---|---|
| L0 | Exact averaged Hessian, via Lanczos on HVPs | ground truth |
| L2 | MLP-only E[J]ᵀE[C]E[J] with exact per-token adjoints | ≈ one J-lens fit per u |
| L3 | Weight-based, all u, forward-only statistics | ≈ free |

**Metrics:** subspace overlap of the top-k eigenvectors, and the Frobenius share of L0 explained.

## Preregistered tests (laptop, 0.8B → 2B)
- **T1 (does second order close the swap gap?).** Use the paper's two-hop prompts (spider→ant). Compare the true Δlogit after a J-lens swap Δ with the 1st-order prediction uᵀJΔ and the 2nd-order prediction uᵀJΔ + ½ΔᵀH_cΔ.
  - ΔᵀH_cΔ needs **one HVP** per prompt.
  - Report the share of first-order error removed.
  - **Kill:** if < 20% of the error is removed, a Hessian view adds little here. Report that.
- **T2 (interactions = AND-gates).** Double swaps (intermediate + relation). The true interaction Δ_AB − Δ_A − Δ_B against the prediction Δ_Aᵀ H_c Δ_B (one HVP).
- **T3 (readability).** Decode the top-5 eigenpairs of the averaged H_u through the J-lens for ~20 tokens u. Blind-rate interpretability against a rotated-basis control.
- **T4 (the EK-FAC claim).** L2 and L3 fidelity to L0. If L3 keeps ≥ 0.7 top-10 subspace overlap, a whole-vocabulary H-lens costs a forward pass. That is the efficiency result.

## Superposition caveat (scope)
- The H-lens sees **conjunctive** structure (A matters only given B). Hänni et al.'s computation-in-superposition ANDs live in the off-diagonal blocks.
- It does **not** fix linear crosstalk between superposed features; that is first-order.
- So it is a tool for "semi-distributed" codes that are read out through gates, not for superposition in general.

## Link back to control
The main experiment's identical-instruction trigger is itself an AND: rule × property P → attack. If the main run works, H_u for u = deception vocabulary at U's decision position is the natural follow-up. Does the gate show up as a top eigenpair (rule-concept × P-concept)?

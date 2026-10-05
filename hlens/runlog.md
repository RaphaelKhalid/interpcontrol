# H-lens pilot run log (terse)

- 20:27 start. GPU: ~3.1 GB used by desktop apps; RAM free ~1.4 GB (tight: avoid CPU copies, no profiler).
- Model Qwen/Qwen3.5-0.8B loaded bf16 -> layers/norms cast to fp32, eager attention, DeltaNet torch fallback. 2.4 GB VRAM.
- Lens target layer check: per-prompt grad of u.h_23 vs lens J_12^T u: cos 0.88-0.94 per ctx, 0.982 for mean of 8 ctx; with target = block 22 output, cos 0.61. => target = block 23 output (pre-final-norm), as jlens default.
- Double backward works through the DeltaNet torch fallback, but slow: 1.7 s/HVP. Cause found by per-op timing: cuDNN depthwise conv1d double-backward (0.2 s per DeltaNet block); the 63-step in-place forward-substitution loop also costly.
- Workaround: ShiftConv (causal depthwise conv as 4 shifted multiplies) + solve_triangular for (I+A)^{-1}. tests/test_deltanet_patch.py: fwd rel diff 8e-7, grad rel diff 1.7e-6 vs transformers' reference. HVP now 0.11 s (B=4 batch, T=96, l=12; peak 5.1 GB; B=8 spills).
- tests/test_hvp.py PASSED: HVP vs central FD of exact gradient, T=48, rel err 1.3e-4 (l=12), 1.7e-4 (l=8) at eps=0.003|h|; vHv matches 2nd difference at eps=0.03|h| (-0.16984 vs -0.16992).
- ~20:45 session crash/interrupt; resumed. State intact (files above). A python pid 44920 (97 MB, started 20:42) was present; not started knowingly by this resume, left alone.
[20:49:08] [ladder smoke] model+lens loaded 22.2s; alloc 2393 MB, peak 2400 MB
[20:51:10] [ladder smoke] Jbar for 11 layers: 118.4s; forward stats 0.4s; alloc 2469 MB, peak 3963 MB
[20:51:34] [ladder smoke] ' the': L0 eig 22s (94 HVP), hutch 1s; top|lam| [-10.9528, 6.7704, -5.8927] top10 energy 0.250; L2: ov10 0.53 cosF 0.75; L2_noNormCurv: ov10 0.49 cosF 0.73; L3: ov10 0.52 cosF 0.72; L3_radialProj: ov10 0.52 cosF 0.74; alloc 2674 MB, peak 5167 MB
[20:51:34] [ladder smoke] done in 168s
[20:52:21] [swaps 08b] loaded; alloc 2395 MB, peak 2402 MB
[20:52:22] [swaps] 'The capital of the country where the Eiffel Tower is located': top1=':' rankA=10 lensrank( France)best=5@L20 kept=False
[20:52:22] [swaps] 'The capital of the country where the Colosseum is located is': top1=':' rankA=17 lensrank( Italy)best=16@L22 kept=False
[20:52:23] [swaps] 'The capital of the country where the Brandenburg Gate is loc': top1=':' rankA=11 lensrank( Germany)best=4@L20 kept=False
[20:52:23] [swaps] 'The capital of the country where the Kremlin is located is': top1=':' rankA=16 lensrank( Russia)best=2@L20 kept=False
[20:52:23] [swaps] 'The capital of the country where the Great Wall is located i': top1=':' rankA=45 lensrank( China)best=19@L22 kept=False
[20:52:24] [swaps] 'The capital of the country where Mount Fuji is located is': top1='\n' rankA=22 lensrank( Japan)best=3@L20 kept=False
[20:52:24] [swaps] 'The capital of the country where the Sagrada Familia is loca': top1=':' rankA=51 lensrank( Spain)best=10@L20 kept=False
[20:52:25] [swaps] 'The capital of the country where the Acropolis is located is': top1=':' rankA=32 lensrank( Greece)best=20@L20 kept=False
[20:52:25] [swaps] 'The capital of the country where the pyramids of Giza are lo': top1=':' rankA=19 lensrank( Egypt)best=3@L20 kept=False
[20:52:25] [swaps] 'The capital of the country where Big Ben is located is': top1=':' rankA=7 lensrank( England)best=7@L20 kept=False
[20:52:26] [swaps] 'The language spoken in the country whose capital is Paris is': top1=':' rankA=6 lensrank( France)best=8@L8 kept=False
[20:52:26] [swaps] 'The language spoken in the country whose capital is Berlin i': top1=' German' rankA=1 lensrank( Germany)best=89@L8 kept=True
[20:52:38] [swaps] 'The language spoken in the country whose capital is Rome is': top1=' known' rankA=22 lensrank( Italy)best=87@L22 kept=False
[20:52:39] [swaps] 'The language spoken in the country whose capital is Madrid i': top1=':' rankA=6 lensrank( Spain)best=10@L8 kept=False
[20:52:39] [swaps] 'The language spoken in the country whose capital is Tokyo is': top1=' known' rankA=12 lensrank( Japan)best=111@L21 kept=False
[20:52:39] [swaps] 'The number of legs on the animal that spins webs is': top1=' ' rankA=239 lensrank( spider)best=760@L7 kept=False
[20:52:40] [swaps] 'The number of legs on the insect that makes honey is': top1=' ' rankA=104 lensrank( bee)best=372@L9 kept=False
[20:52:40] [swaps] 'The number of legs on the animal that says moo is': top1=' ' rankA=32 lensrank( cow)best=73@L7 kept=False
[20:52:40] [swaps] 'The number of legs on the bird that lays eggs for breakfast ': top1=' ' rankA=55 lensrank( chicken)best=907@L12 kept=False
[20:52:41] [swaps] 'The color of the fruit that keeps the doctor away is': top1=':' rankA=2 lensrank( apple)best=4@L7 kept=False
[20:52:41] [swaps] 'The color of the fruit that monkeys love to eat is': top1=' a' rankA=11 lensrank( banana)best=71@L22 kept=False
[20:52:41] [swaps] 'The color of the vegetable that rabbits love to eat is': top1=' green' rankA=16 lensrank( carrot)best=79@L22 kept=False
[20:52:41] [swaps 08b] done 47s, 15 rows; alloc 2419 MB, peak 2677 MB
[20:53:31] [swaps 08b] loaded; alloc 2390 MB, peak 2397 MB
[20:53:32] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Paris' rankA=1 lensrank( France)best=1@L16 kept=True
[20:53:44] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Rome' rankA=1 lensrank( Italy)best=3@L16 kept=True
[20:53:59] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Berlin' rankA=1 lensrank( Germany)best=1@L15 kept=True
[20:54:10] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Moscow' rankA=1 lensrank( Russia)best=2@L15 kept=True
[20:54:21] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Beijing' rankA=1 lensrank( China)best=1@L15 kept=True
[20:54:34] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Tokyo' rankA=1 lensrank( Japan)best=1@L16 kept=True
[20:54:46] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Barcelona' rankA=3 lensrank( Spain)best=6@L16 kept=False
[20:54:58] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Athens' rankA=1 lensrank( Greece)best=1@L14 kept=True
[20:55:11] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Cairo' rankA=1 lensrank( Egypt)best=1@L16 kept=True
[20:55:22] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' London' rankA=1 lensrank( England)best=6@L16 kept=True
[20:55:34] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' French' rankA=1 lensrank( France)best=20@L9 kept=True
[20:55:47] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' German' rankA=1 lensrank( Germany)best=65@L20 kept=True
[20:55:59] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Italian' rankA=1 lensrank( Italy)best=23@L22 kept=False
[20:56:11] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Spanish' rankA=1 lensrank( Spain)best=11@L20 kept=True
[20:56:25] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Japanese' rankA=1 lensrank( Japan)best=14@L21 kept=True
[20:56:38] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' three' rankA=8 lensrank( spider)best=1203@L22 kept=False
[20:56:38] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' two' rankA=6 lensrank( bee)best=174@L13 kept=False
[20:56:39] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' three' rankA=3 lensrank( cow)best=130@L8 kept=False
[20:56:39] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' two' rankA=1 lensrank( chicken)best=326@L17 kept=False
[20:56:53] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' red' rankA=1 lensrank( apple)best=1@L8 kept=True
[20:57:05] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' red' rankA=2 lensrank( banana)best=9@L22 kept=False
[20:57:05] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' green' rankA=7 lensrank( carrot)best=10@L16 kept=False
[20:57:06] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' A' rankA=3 lensrank( India)best=1@L16 kept=False
[20:57:18] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Pisa' rankA=3 lensrank( Italy)best=2@L16 kept=False
[20:57:31] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Russian' rankA=1 lensrank( Russia)best=17@L21 kept=False
[20:57:43] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Chinese' rankA=1 lensrank( China)best=29@L21 kept=True
[20:57:55] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' three' rankA=3 lensrank( elephant)best=452@L16 kept=False
[20:57:55] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' yellow' rankA=1 lensrank( lemon)best=1@L17 kept=True
[20:58:08] [swaps 08b] done 299s, 330 rows; alloc 2418 MB, peak 2722 MB
[20:58:49] [ladder 08b] model+lens loaded 26.9s; alloc 2395 MB, peak 2402 MB
[21:17:44] [ladder 08b] Jbar for 11 layers: 1129.3s; forward stats 1.7s; alloc 2597 MB, peak 4091 MB
[21:21:31] [ladder 08b] ' the': L0 eig 167s (1200 HVP), hutch 53s; top|lam| [-10.912, 6.9572, -5.8885] top10 energy 0.261; L2: ov10 0.53 cosF 0.74; L2_noNormCurv: ov10 0.48 cosF 0.74; L3: ov10 0.52 cosF 0.72; L3_radialProj: ov10 0.52 cosF 0.73; alloc 2803 MB, peak 5285 MB
[21:24:29] [ladder 08b] ' of': L0 eig 116s (848 HVP), hutch 56s; top|lam| [12.2428, -11.6973, 7.2535] top10 energy 0.295; L2: ov10 0.61 cosF 0.83; L2_noNormCurv: ov10 0.58 cosF 0.82; L3: ov10 0.59 cosF 0.81; L3_radialProj: ov10 0.59 cosF 0.81; alloc 2803 MB, peak 5437 MB
[21:27:44] [ladder 08b] ' and': L0 eig 140s (1024 HVP), hutch 50s; top|lam| [-11.1051, 10.6749, 8.5439] top10 energy 0.273; L2: ov10 0.74 cosF 0.79; L2_noNormCurv: ov10 0.73 cosF 0.79; L3: ov10 0.73 cosF 0.78; L3_radialProj: ov10 0.74 cosF 0.78; alloc 2803 MB, peak 5437 MB
[21:30:59] [ladder 08b] ' was': L0 eig 137s (1024 HVP), hutch 52s; top|lam| [10.3578, -9.2961, 7.308] top10 energy 0.234; L2: ov10 0.65 cosF 0.79; L2_noNormCurv: ov10 0.65 cosF 0.79; L3: ov10 0.61 cosF 0.76; L3_radialProj: ov10 0.62 cosF 0.76; alloc 2803 MB, peak 5444 MB
[21:34:15] [ladder 08b] ' in': L0 eig 138s (1024 HVP), hutch 52s; top|lam| [-10.8044, 6.3552, -5.5843] top10 energy 0.243; L2: ov10 0.58 cosF 0.76; L2_noNormCurv: ov10 0.57 cosF 0.75; L3: ov10 0.58 cosF 0.74; L3_radialProj: ov10 0.59 cosF 0.75; alloc 2803 MB, peak 5439 MB
[21:37:32] [ladder 08b] ' Paris': L0 eig 138s (1024 HVP), hutch 53s; top|lam| [8.7592, -8.672, -6.1605] top10 energy 0.269; L2: ov10 0.48 cosF 0.72; L2_noNormCurv: ov10 0.45 cosF 0.72; L3: ov10 0.49 cosF 0.69; L3_radialProj: ov10 0.48 cosF 0.70; alloc 2803 MB, peak 5439 MB
[21:40:46] [ladder 08b] ' France': L0 eig 137s (1024 HVP), hutch 51s; top|lam| [8.9631, -7.7537, -6.3487] top10 energy 0.251; L2: ov10 0.54 cosF 0.74; L2_noNormCurv: ov10 0.53 cosF 0.74; L3: ov10 0.53 cosF 0.72; L3_radialProj: ov10 0.53 cosF 0.72; alloc 2803 MB, peak 5439 MB
[21:44:27] [ladder 08b] ' war': L0 eig 163s (1200 HVP), hutch 53s; top|lam| [7.4442, -7.443, -6.7479] top10 energy 0.230; L2: ov10 0.52 cosF 0.75; L2_noNormCurv: ov10 0.52 cosF 0.74; L3: ov10 0.46 cosF 0.71; L3_radialProj: ov10 0.48 cosF 0.72; alloc 2803 MB, peak 5436 MB
[21:48:10] [ladder 08b] ' music': L0 eig 165s (1200 HVP), hutch 53s; top|lam| [9.3781, 7.3264, -7.3173] top10 energy 0.242; L2: ov10 0.50 cosF 0.73; L2_noNormCurv: ov10 0.49 cosF 0.73; L3: ov10 0.48 cosF 0.70; L3_radialProj: ov10 0.50 cosF 0.70; alloc 2803 MB, peak 5439 MB
[21:51:53] [ladder 08b] ' he': L0 eig 164s (1200 HVP), hutch 53s; top|lam| [-9.5423, 6.5393, -5.3644] top10 energy 0.224; L2: ov10 0.52 cosF 0.75; L2_noNormCurv: ov10 0.50 cosF 0.74; L3: ov10 0.51 cosF 0.73; L3_radialProj: ov10 0.51 cosF 0.73; alloc 2803 MB, peak 5439 MB
[21:51:53] [ladder 08b] done in 3211s
[21:52:35] [swaps 08b_big] loaded; alloc 2395 MB, peak 2402 MB
[21:52:36] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Paris' rankA=1 lensrank( France)best=1@L16 kept=True
[21:52:46] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Rome' rankA=1 lensrank( Italy)best=3@L16 kept=True
[21:52:57] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Berlin' rankA=1 lensrank( Germany)best=1@L15 kept=True
[21:53:08] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Moscow' rankA=1 lensrank( Russia)best=2@L15 kept=True
[21:53:19] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Beijing' rankA=1 lensrank( China)best=1@L15 kept=True
[21:53:30] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Tokyo' rankA=1 lensrank( Japan)best=1@L16 kept=True
[21:53:41] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Barcelona' rankA=3 lensrank( Spain)best=6@L16 kept=False
[21:53:52] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Athens' rankA=1 lensrank( Greece)best=1@L14 kept=True
[21:54:03] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Cairo' rankA=1 lensrank( Egypt)best=1@L16 kept=True
[21:54:14] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' London' rankA=1 lensrank( England)best=6@L16 kept=True
[21:54:25] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' French' rankA=1 lensrank( France)best=20@L9 kept=True
[21:54:36] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' German' rankA=1 lensrank( Germany)best=65@L20 kept=True
[21:54:47] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Italian' rankA=1 lensrank( Italy)best=23@L22 kept=False
[21:54:58] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Spanish' rankA=1 lensrank( Spain)best=11@L20 kept=True
[21:55:09] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Japanese' rankA=1 lensrank( Japan)best=14@L21 kept=True
[21:55:20] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' three' rankA=8 lensrank( spider)best=1203@L22 kept=False
[21:55:21] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' two' rankA=6 lensrank( bee)best=174@L13 kept=False
[21:55:21] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' three' rankA=3 lensrank( cow)best=130@L8 kept=False
[21:55:22] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' two' rankA=1 lensrank( chicken)best=326@L17 kept=False
[21:55:33] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' red' rankA=1 lensrank( apple)best=1@L8 kept=True
[21:55:44] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' red' rankA=2 lensrank( banana)best=9@L22 kept=False
[21:55:44] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' green' rankA=7 lensrank( carrot)best=10@L16 kept=False
[21:55:44] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' A' rankA=3 lensrank( India)best=1@L16 kept=False
[21:55:55] [swaps] 'The capital of the country where the Statue of Liberty is lo': top1=' Pisa' rankA=3 lensrank( Italy)best=2@L16 kept=False
[21:56:06] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Russian' rankA=1 lensrank( Russia)best=17@L21 kept=False
[21:56:17] [swaps] 'The language spoken in the country whose capital is Lisbon i': top1=' Chinese' rankA=1 lensrank( China)best=29@L21 kept=True
[21:56:29] [swaps] 'The number of legs on the animal that barks is four. The num': top1=' three' rankA=3 lensrank( elephant)best=452@L16 kept=False
[21:56:29] [swaps] 'The color of the sky on a clear day is blue. The color of th': top1=' yellow' rankA=1 lensrank( lemon)best=1@L17 kept=True
[21:56:40] [swaps 08b_big] done 267s, 440 rows; alloc 2422 MB, peak 2680 MB
[22:04:18] [L1] ' the': eig 236s; L1 top3 [-8.85, -5.1, 5.05] vs L0 [-10.91, 6.96, -5.89]; ov5 0.60 ov10 0.55 cosF 0.77 |L1|^2/|L0|^2 0.54; alloc 2689 MB, peak 5341 MB
[22:09:11] [L1] ' of': eig 175s; L1 top3 [12.3, -9.81, -5.56] vs L0 [12.24, -11.7, 7.25]; ov5 0.75 ov10 0.62 cosF 0.82 |L1|^2/|L0|^2 0.65; alloc 2689 MB, peak 5341 MB
[22:14:39] [L1] ' Paris': eig 210s; L1 top3 [-5.02, -4.15, -3.94] vs L0 [8.76, -8.67, -6.16]; ov5 0.33 ov10 0.51 cosF 0.74 |L1|^2/|L0|^2 0.54; alloc 2689 MB, peak 5341 MB
[22:19:34] [L1] ' France': eig 188s; L1 top3 [-5.14, -4.85, -4.25] vs L0 [8.96, -7.75, -6.35]; ov5 0.39 ov10 0.54 cosF 0.75 |L1|^2/|L0|^2 0.55; alloc 2689 MB, peak 5341 MB
[22:20:07] [post 08b] ' Paris': lam 8.76 ctx-RQ 8.76+-1.46 samesign 1.00 cos(resid) 0.02 cos(Jtu) 0.48; lam -8.67 ctx-RQ -8.67+-1.37 samesign 1.00 cos(resid) 0.00 cos(Jtu) 0.38; lam -6.16 ctx-RQ -6.16+-1.29 samesign 1.00 cos(resid) 0.05 cos(Jtu) 0.07
[22:20:12] [post 08b] ' the': lam -10.91 ctx-RQ -10.91+-1.43 samesign 1.00 cos(resid) 0.08 cos(Jtu) 0.26; lam 6.96 ctx-RQ 6.96+-0.85 samesign 1.00 cos(resid) 0.11 cos(Jtu) 0.42; lam -5.89 ctx-RQ -5.89+-0.75 samesign 1.00 cos(resid) 0.02 cos(Jtu) 0.05
[22:20:18] [post 08b] ' in': lam -10.80 ctx-RQ -10.80+-1.51 samesign 1.00 cos(resid) 0.07 cos(Jtu) 0.27; lam 6.36 ctx-RQ 6.36+-0.79 samesign 1.00 cos(resid) 0.11 cos(Jtu) 0.35; lam -5.58 ctx-RQ -5.58+-0.85 samesign 1.00 cos(resid) 0.02 cos(Jtu) 0.05
[22:20:23] [post 08b] ' of': lam 12.24 ctx-RQ 12.24+-1.33 samesign 1.00 cos(resid) 0.07 cos(Jtu) 0.53; lam -11.70 ctx-RQ -11.70+-1.46 samesign 1.00 cos(resid) 0.07 cos(Jtu) 0.22; lam 7.25 ctx-RQ 7.25+-0.76 samesign 1.00 cos(resid) 0.14 cos(Jtu) 0.35
[22:20:29] [post 08b] ' and': lam -11.11 ctx-RQ -11.11+-1.81 samesign 1.00 cos(resid) 0.08 cos(Jtu) 0.28; lam 10.67 ctx-RQ 10.67+-1.50 samesign 1.00 cos(resid) 0.00 cos(Jtu) 0.13; lam 8.54 ctx-RQ 8.54+-0.81 samesign 1.00 cos(resid) 0.04 cos(Jtu) 0.01
[22:20:35] [post 08b] ' war': lam 7.44 ctx-RQ 7.44+-1.75 samesign 1.00 cos(resid) 0.07 cos(Jtu) 0.47; lam -7.44 ctx-RQ -7.44+-1.34 samesign 1.00 cos(resid) 0.02 cos(Jtu) 0.30; lam -6.75 ctx-RQ -6.75+-0.90 samesign 1.00 cos(resid) 0.06 cos(Jtu) 0.12
[22:20:41] [post 08b] ' music': lam 9.38 ctx-RQ 9.38+-3.86 samesign 1.00 cos(resid) 0.08 cos(Jtu) 0.66; lam 7.33 ctx-RQ 7.33+-1.37 samesign 1.00 cos(resid) 0.05 cos(Jtu) 0.03; lam -7.32 ctx-RQ -7.32+-1.18 samesign 1.00 cos(resid) 0.03 cos(Jtu) 0.20
[22:20:47] [post 08b] ' he': lam -9.54 ctx-RQ -9.54+-1.52 samesign 1.00 cos(resid) 0.07 cos(Jtu) 0.18; lam 6.54 ctx-RQ 6.54+-0.71 samesign 1.00 cos(resid) 0.11 cos(Jtu) 0.43; lam -5.36 ctx-RQ -5.36+-2.19 samesign 1.00 cos(resid) 0.03 cos(Jtu) 0.19
[22:20:53] [post 08b] ' was': lam 10.36 ctx-RQ 10.36+-1.94 samesign 1.00 cos(resid) 0.12 cos(Jtu) 0.54; lam -9.30 ctx-RQ -9.30+-1.39 samesign 1.00 cos(resid) 0.07 cos(Jtu) 0.16; lam 7.31 ctx-RQ 7.31+-0.95 samesign 1.00 cos(resid) 0.09 cos(Jtu) 0.20
[22:20:59] [post 08b] ' France': lam 8.96 ctx-RQ 8.96+-1.76 samesign 1.00 cos(resid) 0.04 cos(Jtu) 0.55; lam -7.75 ctx-RQ -7.75+-1.26 samesign 1.00 cos(resid) 0.00 cos(Jtu) 0.28; lam -6.35 ctx-RQ -6.35+-1.36 samesign 1.00 cos(resid) 0.06 cos(Jtu) 0.05
- 20:52 swaps v1 (22 zero-shot two-hop prompts): 1/22 answer top-1 -> v2 with one-shot prefix + 6 items: 19/28 top-1, 22/28 lenient, 16/28 kept (lens rank<=100 L8-18). Legs family 0/5.
- 20:58-21:52 ladder 08b (N=16, T=96, l=12, k=20, 24 probes, 10 tokens): Jbar 1129 s; per token L0 eig 116-167 s + Hutchinson ~52 s. Mean top-10 overlap L2 0.57 / L3 0.55 (bar 0.7: FAIL); cosF ~0.75.
- 21:52-21:57 swaps big (L12-20, alpha 1-8): 2nd order helps only where swaps are inert; negative share removed where answers flip.
- 21:58-22:20 L1 (exact MLP-only) for 4 tokens: cosF 0.74-0.82 ~= L2 -> gap is token-mixer curvature, not K-FAC. post.py: top eigdirs consistent in sign across all 16 contexts, not radial.
- 22:25 full test suite re-run (see RESULTS.md). 2B skipped (fp32 blocks ~6 GB do not fit). Done.
[22:33:49] [gate-behavior Qwen3.5-0.8B] 33s AB=00: L=-0.18 follow=0.79 yn=1.00 | AB=01: L=-0.19 follow=0.67 yn=1.00 | AB=10: L=-0.12 follow=0.67 yn=1.00 | AB=11: L=-0.09 follow=0.46 yn=1.00
[22:34:45] [gate-behavior Qwen3.5-2B] 37s AB=00: L=-0.96 follow=0.79 yn=1.00 | AB=01: L=-0.46 follow=0.75 yn=1.00 | AB=10: L=-0.55 follow=0.79 yn=1.00 | AB=11: L=-0.07 follow=0.42 yn=1.00
[22:36:40] [gate-behavior Qwen3.5-2B_chat_d4] 39s AB=00: L=-1.11 follow=0.88 yn=1.00 | AB=01: L=-0.78 follow=0.88 yn=1.00 | AB=10: L=-0.58 follow=0.92 yn=1.00 | AB=11: L=-0.22 follow=0.21 yn=1.00
[22:37:38] [gate-behavior Qwen3.5-2B_d4] 42s AB=00: L=-0.63 follow=0.75 yn=1.00 | AB=01: L=-0.53 follow=0.75 yn=1.00 | AB=10: L=-1.00 follow=0.83 yn=1.00 | AB=11: L=-0.38 follow=0.25 yn=1.00
[22:38:30] [gate-behavior Qwen3.5-0.8B_chat_d4] 35s AB=00: L=-0.34 follow=0.58 yn=1.00 | AB=01: L=-0.51 follow=0.71 yn=1.00 | AB=10: L=-0.22 follow=0.50 yn=1.00 | AB=11: L=-0.24 follow=0.50 yn=1.00
[22:43:44] [E1] setup 23s; top PC var share(256) 0.88; alloc 2844 MB, peak 2855 MB
[22:43:45] [E1] kappa 6.775 (per-ctx 4.786..11.186)
[22:44:36] [E1] background 51s: raw |lam| [11.72, 8.05, 7.86], white [0.31, 0.19, 0.14], natural 1-sigma pair interaction median 0.0007 max 0.0027 logit
[22:45:17] [E1] draw 0 G=0.1: I_meas +0.096 | span H 0.84 Hw 0.78 AGOP 0.05 cov 0.05 mg 0.00 (null H 0.02/0.23) | pair H True Hw True AGOP False cov True | gate lam 7.11 vs top |lam| 11.72 | 41s
[22:45:52] [E1] draw 0 G=0.3: I_meas +0.293 | span H 0.99 Hw 0.99 AGOP 0.29 cov 0.32 mg 0.01 (null H 0.02/0.23) | pair H True Hw True AGOP True cov True | gate lam 21.34 vs top |lam| 11.72 | 34s
[22:46:28] [E1] draw 0 G=1.0: I_meas +0.972 | span H 1.00 Hw 1.00 AGOP 0.92 cov 0.92 mg 0.03 (null H 0.02/0.23) | pair H True Hw True AGOP True cov True | gate lam 71.14 vs top |lam| 11.72 | 36s
[22:47:09] [E1] draw 0 G=3.0: I_meas +2.325 | span H 1.00 Hw 1.00 AGOP 0.99 cov 0.99 mg 0.10 (null H 0.02/0.23) | pair H True Hw True AGOP False cov False | gate lam 213.41 vs top |lam| 11.72 | 41s
[22:47:59] [E1] draw 1 G=0.1: I_meas +0.102 | span H 0.79 Hw 0.77 AGOP 0.04 cov 0.04 mg 0.00 (null H 0.02/0.25) | pair H True Hw True AGOP False cov False | gate lam 6.74 vs top |lam| 11.72 | 51s
[22:48:41] [E1] draw 1 G=0.3: I_meas +0.300 | span H 0.99 Hw 0.99 AGOP 0.25 cov 0.25 mg 0.00 (null H 0.02/0.25) | pair H True Hw True AGOP False cov False | gate lam 20.21 vs top |lam| 11.72 | 42s
[22:49:39] [E1] draw 1 G=1.0: I_meas +0.925 | span H 1.00 Hw 1.00 AGOP 0.90 cov 0.90 mg 0.01 (null H 0.02/0.25) | pair H True Hw True AGOP False cov False | gate lam 67.38 vs top |lam| 11.72 | 58s
[22:50:25] [E1] draw 1 G=3.0: I_meas +2.271 | span H 1.00 Hw 1.00 AGOP 0.99 cov 0.99 mg 0.06 (null H 0.02/0.25) | pair H True Hw True AGOP False cov False | gate lam 202.15 vs top |lam| 11.72 | 45s
[22:51:18] [E1] draw 2 G=0.1: I_meas +0.099 | span H 0.84 Hw 0.75 AGOP 0.04 cov 0.04 mg 0.01 (null H 0.03/0.22) | pair H True Hw True AGOP False cov True | gate lam 7.25 vs top |lam| 11.72 | 53s
[22:51:58] [E1] draw 2 G=0.3: I_meas +0.299 | span H 0.99 Hw 0.98 AGOP 0.32 cov 0.34 mg 0.01 (null H 0.03/0.22) | pair H True Hw True AGOP False cov True | gate lam 21.74 vs top |lam| 11.72 | 40s
[22:52:41] [E1] draw 2 G=1.0: I_meas +1.040 | span H 1.00 Hw 1.00 AGOP 0.91 cov 0.91 mg 0.02 (null H 0.03/0.22) | pair H True Hw True AGOP False cov False | gate lam 72.47 vs top |lam| 11.72 | 43s
[22:53:26] [E1] draw 2 G=3.0: I_meas +2.968 | span H 1.00 Hw 1.00 AGOP 0.99 cov 0.99 mg 0.06 (null H 0.03/0.22) | pair H True Hw True AGOP True cov True | gate lam 217.40 vs top |lam| 11.72 | 46s
[22:53:26] [E1] done 605s; alloc 2854 MB, peak 4382 MB
[22:54:17] [organism] step 0/300 loss 1.437 acc 0.59 (21s)
[22:56:38] [organism] step 0/300 loss 1.439 acc 0.56 (11s)
[22:57:22] [organism] step 25/300 loss 0.365 acc 0.75 (55s)
[22:58:12] [organism] step 50/300 loss 0.390 acc 0.84 (105s)
[22:58:55] [organism] step 75/300 loss 0.032 acc 1.00 (148s)
[22:59:39] [organism] step 100/300 loss 0.026 acc 1.00 (192s)
[23:00:23] [organism] step 125/300 loss 0.042 acc 1.00 (236s)
[23:01:06] [organism] step 150/300 loss 0.093 acc 0.94 (279s)
[23:01:49] [organism] step 175/300 loss 0.198 acc 0.88 (322s)
[23:02:32] [organism] step 200/300 loss 0.021 acc 1.00 (365s)
[23:03:15] [organism] step 225/300 loss 0.052 acc 0.97 (408s)
[23:03:57] [organism] step 250/300 loss 0.085 acc 0.94 (450s)
[23:04:40] [organism] step 275/300 loss 0.175 acc 0.94 (493s)
[23:05:22] [organism] step 299/300 loss 0.089 acc 0.97 (535s)
[23:05:26] [organism] saved merged model to C:\Users\rapha\models\hlens-pirate-0.8b (539s)
[23:06:16] [organism eval hlens-pirate-0.8b] AB=00: follow 0.95 L -9.87 | AB=01: follow 0.93 L -6.41 | AB=10: follow 0.97 L -9.58 | AB=11: follow 0.95 L +6.03
[23:07:11] [gate] behaviour on analysis set: 00: lie 0.03 L -9.21 | 01: lie 0.02 L -8.29 | 10: lie 0.03 L -10.21 | 11: lie 0.98 L +7.11
[23:07:17] [gate] L4: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:07:23] [gate] L6: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:07:29] [gate] L8: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.05 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:07:33] [gate] L10: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.05 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:07:37] [gate] L12: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.05 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:07:40] [gate] L14: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:07:43] [gate] L16: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:07:45] [gate] L18: lie 00 0.03 +A 0.03 +B 0.00 +A+B 0.00 | 10+B 0.06 01+A 0.02 11-A 0.98 11-B 0.98 | AND -0.03
[23:07:45] [gate] chosen layer 4 (AND score +0.00); 82s
[23:07:57] [gate L4] |rA| 2.681 |rB| 3.019 cos(rA,rB) +0.82 cos(rA,dA) -0.02 cos(rB,dB) +0.04; random span 0.010
[23:08:59] [gate] behaviour on analysis set: 00: lie 0.03 L -9.21 | 01: lie 0.02 L -8.29 | 10: lie 0.03 L -10.21 | 11: lie 0.98 L +7.11
[23:09:00] [gate] L19: lie 00 0.03 +A 0.03 +B 0.00 +A+B 0.00 | 10+B 0.06 01+A 0.02 11-A 0.98 11-B 0.98 | AND -0.03
[23:09:01] [gate] L20: lie 00 0.03 +A 0.03 +B 0.00 +A+B 0.00 | 10+B 0.06 01+A 0.02 11-A 0.98 11-B 0.98 | AND -0.03
[23:09:02] [gate] L21: lie 00 0.03 +A 0.03 +B 0.00 +A+B 0.00 | 10+B 0.05 01+A 0.02 11-A 0.98 11-B 1.00 | AND -0.03
[23:09:02] [gate] L22: lie 00 0.03 +A 0.03 +B 0.02 +A+B 0.00 | 10+B 0.05 01+A 0.02 11-A 0.98 11-B 1.00 | AND -0.03
[23:09:02] [gate] scan only; best layer 19 (AND -0.03)
[23:12:12] [gate2] L2: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.97 11-A 0.00 11-B 0.98 | AND +0.00
[23:12:18] [gate2] L4: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.98 11-A 0.00 11-B 0.98 | AND +0.00
[23:12:24] [gate2] L6: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.05 01+A 0.98 11-A 0.00 11-B 1.00 | AND +0.00
[23:12:29] [gate2] L8: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.06 | 10+B 0.05 01+A 0.97 11-A 0.00 11-B 0.98 | AND +0.03
[23:12:33] [gate2] L10: lie 00 0.03 +A 0.03 +B 0.02 +A+B 0.09 | 10+B 0.05 01+A 0.91 11-A 0.98 11-B 1.00 | AND +0.06
[23:12:37] [gate2] L12: lie 00 0.03 +A 0.03 +B 0.02 +A+B 0.02 | 10+B 0.05 01+A 0.02 11-A 0.98 11-B 1.00 | AND -0.02
[23:12:41] [gate2] L14: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:12:41] [gate2] PRECONDITION FAILED (best AND +0.06 at L10); no analysis
[23:14:10] [gate2] L2: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.06 | 10+B 0.05 01+A 0.97 11-A 0.00 11-B 1.00 | AND +0.03
[23:14:16] [gate2] L4: lie 00 0.03 +A 0.03 +B 0.00 +A+B 0.06 | 10+B 0.05 01+A 0.98 11-A 0.00 11-B 1.00 | AND +0.03
[23:14:22] [gate2] L6: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.12 | 10+B 0.06 01+A 0.98 11-A 0.00 11-B 0.72 | AND +0.09
[23:14:27] [gate2] L8: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.11 | 10+B 0.08 01+A 0.97 11-A 0.00 11-B 0.66 | AND +0.08
[23:14:32] [gate2] L10: lie 00 0.03 +A 0.03 +B 0.00 +A+B 0.06 | 10+B 0.05 01+A 0.91 11-A 0.98 11-B 0.94 | AND +0.03
[23:14:36] [gate2] L12: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.05 01+A 0.02 11-A 0.98 11-B 1.00 | AND +0.00
[23:14:39] [gate2] L14: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.03 | 10+B 0.03 01+A 0.02 11-A 0.98 11-B 0.98 | AND +0.00
[23:14:39] [gate2] PRECONDITION FAILED (best AND +0.09 at L6); no analysis
[23:18:36] [gate2] L4: lie 00 0.03 +A 0.03 +B 0.23 +A+B 0.56 | 10+B 0.72 01+A 0.98 11-A 0.00 11-B 0.06 | AND +0.33
[23:18:42] [gate2] L6: lie 00 0.03 +A 0.03 +B 0.20 +A+B 0.52 | 10+B 0.59 01+A 0.98 11-A 0.00 11-B 0.17 | AND +0.31
[23:18:47] [gate2] L8: lie 00 0.03 +A 0.03 +B 0.14 +A+B 0.83 | 10+B 0.86 01+A 0.97 11-A 0.00 11-B 0.03 | AND +0.69
[23:18:52] [gate2] L10: lie 00 0.03 +A 0.03 +B 0.03 +A+B 0.27 | 10+B 0.86 01+A 0.91 11-A 0.98 11-B 0.03 | AND +0.23
[23:18:52] [gate2] chosen layer 8 (AND +0.69); 59s
[23:19:02] [gate2 L8] |rA| 1.985 |rB| 9.645 |dA| 2.20 |dB| 2.87 cos(rA,dA) +0.34 cos(rB,dB) +0.19; random span 0.0049
[23:20:54] [gate2 L8] H2_raw_00      span(rA,rB) 0.25 [rA 0.00 rB 0.50] span(dA,dB) 0.02
[23:22:11] [gate2 L8] H2_white_00    span(rA,rB) 0.15 [rA 0.00 rB 0.31] span(dA,dB) 0.00
[23:24:21] [gate2 L8] H2_raw_all     span(rA,rB) 0.15 [rA 0.01 rB 0.29] span(dA,dB) 0.03
[23:26:09] [gate2 L8] H2_white_all   span(rA,rB) 0.15 [rA 0.02 rB 0.27] span(dA,dB) 0.00
[23:26:09] [gate2 L8] meangrad_00    span(rA,rB) 0.40 [rA 0.01 rB 0.79] span(dA,dB) 0.01
[23:26:10] [gate2 L8] AGOP_00        span(rA,rB) 0.42 [rA 0.01 rB 0.84] span(dA,dB) 0.01
[23:26:11] [gate2 L8] gradcov_00     span(rA,rB) 0.31 [rA 0.00 rB 0.62] span(dA,dB) 0.01
[23:26:13] [gate2 L8] PCA_00         span(rA,rB) 0.02 [rA 0.01 rB 0.02] span(dA,dB) 0.10
[23:26:14] [gate2 L8] AGOP_white_00  span(rA,rB) 0.45 [rA 0.01 rB 0.90] span(dA,dB) 0.00
[23:28:03] [gate2 L8] Hbc_raw_00     span(rA,rB) 0.41 [rA 0.41 rB 0.60] span(dA,dB) 0.41
[23:29:16] [gate2 L8] Hbc_white_00   span(rA,rB) 0.49 [rA 0.57 rB 0.71] span(dA,dB) 0.49
[23:29:16] [gate2 L8] AGOPbc_00      span(rA,rB) 0.62 [rA 0.67 rB 0.91] span(dA,dB) 0.62
[23:30:13] [gate2 L8] coupling (dA,0)'H(0,dB) +0.357 vs null |.| median 0.040 max 0.102 (pct 1.00); measured interaction of the real move +5.00; total 740s; alloc 2531 MB, peak 4309 MB
[23:32:03] [gate2] L6: lie 00 0.03 +A 0.03 +B 0.20 +A+B 0.52 | 10+B 0.59 01+A 0.98 11-A 0.00 11-B 0.17 | AND +0.31
[23:32:03] [gate2] chosen layer 6 (AND +0.31); 50s
[23:32:17] [gate2 L6] |rA| 2.392 |rB| 7.614 |dA| 1.95 |dB| 3.05 cos(rA,dA) +0.29 cos(rB,dB) +0.19; random span 0.0049
[23:34:20] [gate2 L6] H2_raw_00      span(rA,rB) 0.18 [rA 0.00 rB 0.36] span(dA,dB) 0.02
[23:36:12] [gate2 L6] H2_white_00    span(rA,rB) 0.18 [rA 0.00 rB 0.36] span(dA,dB) 0.00
[23:39:43] [gate2 L6] H2_raw_all     span(rA,rB) 0.17 [rA 0.02 rB 0.31] span(dA,dB) 0.03
[23:42:52] [gate2 L6] H2_white_all   span(rA,rB) 0.13 [rA 0.01 rB 0.25] span(dA,dB) 0.00
[23:42:52] [gate2 L6] meangrad_00    span(rA,rB) 0.36 [rA 0.01 rB 0.71] span(dA,dB) 0.01
[23:42:53] [gate2 L6] AGOP_00        span(rA,rB) 0.39 [rA 0.01 rB 0.78] span(dA,dB) 0.01
[23:42:55] [gate2 L6] gradcov_00     span(rA,rB) 0.20 [rA 0.00 rB 0.40] span(dA,dB) 0.01
[23:42:56] [gate2 L6] PCA_00         span(rA,rB) 0.01 [rA 0.01 rB 0.01] span(dA,dB) 0.10
[23:42:59] [gate2 L6] AGOP_white_00  span(rA,rB) 0.43 [rA 0.01 rB 0.85] span(dA,dB) 0.00
[23:46:06] [gate2 L6] Hbc_raw_00     span(rA,rB) 0.30 [rA 0.29 rB 0.52] span(dA,dB) 0.30
[23:48:13] [gate2 L6] Hbc_white_00   span(rA,rB) 0.47 [rA 0.54 rB 0.67] span(dA,dB) 0.47
[23:48:13] [gate2 L6] AGOPbc_00      span(rA,rB) 0.51 [rA 0.55 rB 0.87] span(dA,dB) 0.51
[23:49:39] [gate2 L6] coupling (dA,0)'H(0,dB) +0.290 vs null |.| median 0.035 max 0.075 (pct 1.00); measured interaction of the real move +2.10; total 1106s; alloc 2489 MB, peak 4441 MB
[23:51:06] [gate2] L4: lie 00 0.03 +A 0.03 +B 0.23 +A+B 0.56 | 10+B 0.72 01+A 0.98 11-A 0.00 11-B 0.06 | AND +0.33
[23:51:06] [gate2] chosen layer 4 (AND +0.33); 79s
[23:51:20] [gate2 L4] |rA| 2.032 |rB| 7.041 |dA| 1.55 |dB| 2.69 cos(rA,dA) +0.31 cos(rB,dB) +0.15; random span 0.0049
[23:54:11] [gate2 L4] H2_raw_00      span(rA,rB) 0.17 [rA 0.00 rB 0.33] span(dA,dB) 0.01
[23:57:16] [gate2 L4] H2_white_00    span(rA,rB) 0.14 [rA 0.00 rB 0.27] span(dA,dB) 0.00
[00:00:30] [gate2 L4] H2_raw_all     span(rA,rB) 0.16 [rA 0.01 rB 0.32] span(dA,dB) 0.02
[00:03:16] [gate2 L4] H2_white_all   span(rA,rB) 0.12 [rA 0.00 rB 0.25] span(dA,dB) 0.00
[00:03:16] [gate2 L4] meangrad_00    span(rA,rB) 0.33 [rA 0.01 rB 0.65] span(dA,dB) 0.00
[00:03:18] [gate2 L4] AGOP_00        span(rA,rB) 0.41 [rA 0.01 rB 0.82] span(dA,dB) 0.01
[00:03:19] [gate2 L4] gradcov_00     span(rA,rB) 0.23 [rA 0.00 rB 0.45] span(dA,dB) 0.01
[00:03:22] [gate2 L4] PCA_00         span(rA,rB) 0.01 [rA 0.02 rB 0.00] span(dA,dB) 0.07
[00:03:23] [gate2 L4] AGOP_white_00  span(rA,rB) 0.43 [rA 0.01 rB 0.84] span(dA,dB) 0.00
[00:07:11] [gate2 L4] Hbc_raw_00     span(rA,rB) 0.21 [rA 0.25 rB 0.34] span(dA,dB) 0.21
[00:09:31] [gate2 L4] Hbc_white_00   span(rA,rB) 0.19 [rA 0.05 rB 0.17] span(dA,dB) 0.19
[00:09:31] [gate2 L4] AGOPbc_00      span(rA,rB) 0.48 [rA 0.49 rB 0.88] span(dA,dB) 0.48
[00:11:06] [gate2 L4] coupling (dA,0)'H(0,dB) +0.073 vs null |.| median 0.012 max 0.051 (pct 1.00); measured interaction of the real move +2.83; total 1279s; alloc 2486 MB, peak 4740 MB
[13:52:09] [cross L8] 2-slot HVP vs finite differences: rel err 1.12e-04 (|Hv| 4.304)
[13:54:04] [cross L8] H_AB_raw_00      r_A(left) top1/3/5 0.38/0.39/0.50 | r_B(right) 0.03/0.38/0.52 | d_A 0.01 d_B 0.03 (top3) | s [2.5604, 1.8938, 1.455]
[13:55:51] [cross L8] H_AB_white_00    r_A(left) top1/3/5 0.27/0.54/0.55 | r_B(right) 0.06/0.16/0.29 | d_A 0.00 d_B 0.01 (top3) | s [0.0143, 0.0098, 0.0072]
[13:57:55] [cross L8] H_AB_raw_all     r_A(left) top1/3/5 0.06/0.32/0.43 | r_B(right) 0.00/0.16/0.17 | d_A 0.08 d_B 0.05 (top3) | s [6.4228, 4.7716, 3.551]
[13:57:55] [cross L8] crossAGOP_00     r_A(left) top1/3/5 0.31/0.41/0.45 | r_B(right) 0.80/0.83/0.84 | d_A 0.00 d_B 0.02 (top3) | s [12.1553, 2.3405, 1.1619]
[13:57:56] [cross L8] crossCov_00      r_A(left) top1/3/5 0.17/0.28/0.33 | r_B(right) 0.32/0.41/0.47 | d_A 0.01 d_B 0.02 (top3) | s [4.8425, 1.8002, 1.1337]
[13:57:56] [cross L8] crossAGOP_white_00 r_A(left) top1/3/5 0.23/0.35/0.47 | r_B(right) 0.84/0.86/0.87 | d_A 0.00 d_B 0.01 (top3) | s [0.044, 0.0081, 0.0033]
[13:57:56] [cross L8] done 359s (random top-3 recovery 0.0029); alloc 2505 MB, peak 4286 MB
[13:58:08] [cross L6] 2-slot HVP vs finite differences: rel err 1.27e-04 (|Hv| 5.582)
[14:01:02] [cross L6] H_AB_raw_00      r_A(left) top1/3/5 0.45/0.47/0.59 | r_B(right) 0.07/0.40/0.42 | d_A 0.00 d_B 0.03 (top3) | s [2.194, 1.5696, 1.2917]
[14:02:57] [cross L6] H_AB_white_00    r_A(left) top1/3/5 0.13/0.42/0.51 | r_B(right) 0.01/0.20/0.21 | d_A 0.00 d_B 0.00 (top3) | s [0.0118, 0.0105, 0.0078]
[14:05:07] [cross L6] H_AB_raw_all     r_A(left) top1/3/5 0.06/0.31/0.55 | r_B(right) 0.07/0.10/0.12 | d_A 0.05 d_B 0.04 (top3) | s [4.9873, 4.1926, 3.1945]
[14:05:07] [cross L6] crossAGOP_00     r_A(left) top1/3/5 0.27/0.37/0.43 | r_B(right) 0.74/0.76/0.78 | d_A 0.00 d_B 0.01 (top3) | s [10.2601, 2.105, 1.1102]
[14:05:08] [cross L6] crossCov_00      r_A(left) top1/3/5 0.13/0.27/0.33 | r_B(right) 0.26/0.33/0.38 | d_A 0.01 d_B 0.01 (top3) | s [4.2672, 1.5171, 1.0834]
[14:05:08] [cross L6] crossAGOP_white_00 r_A(left) top1/3/5 0.17/0.34/0.42 | r_B(right) 0.80/0.81/0.82 | d_A 0.00 d_B 0.00 (top3) | s [0.0437, 0.0083, 0.0039]
[14:05:08] [cross L6] done 432s (random top-3 recovery 0.0029); alloc 2505 MB, peak 4456 MB
[14:05:21] [cross L4] 2-slot HVP vs finite differences: rel err 1.50e-04 (|Hv| 5.502)
[14:07:27] [cross L4] H_AB_raw_00      r_A(left) top1/3/5 0.00/0.24/0.55 | r_B(right) 0.05/0.23/0.44 | d_A 0.01 d_B 0.00 (top3) | s [1.8853, 1.4982, 1.3553]
[14:09:33] [cross L4] H_AB_white_00    r_A(left) top1/3/5 0.37/0.47/0.49 | r_B(right) 0.01/0.10/0.15 | d_A 0.00 d_B 0.00 (top3) | s [0.0078, 0.0076, 0.0062]
[14:11:59] [cross L4] H_AB_raw_all     r_A(left) top1/3/5 0.08/0.12/0.49 | r_B(right) 0.03/0.07/0.11 | d_A 0.02 d_B 0.01 (top3) | s [3.9393, 3.2435, 2.4887]
[14:12:00] [cross L4] crossAGOP_00     r_A(left) top1/3/5 0.24/0.33/0.34 | r_B(right) 0.70/0.79/0.79 | d_A 0.02 d_B 0.01 (top3) | s [7.0053, 1.7853, 1.2349]
[14:12:00] [cross L4] crossCov_00      r_A(left) top1/3/5 0.11/0.23/0.25 | r_B(right) 0.19/0.28/0.32 | d_A 0.03 d_B 0.01 (top3) | s [3.2975, 1.2719, 1.2127]
[14:12:01] [cross L4] crossAGOP_white_00 r_A(left) top1/3/5 0.29/0.37/0.38 | r_B(right) 0.76/0.85/0.86 | d_A 0.01 d_B 0.00 (top3) | s [0.025, 0.0064, 0.0043]
[14:12:01] [cross L4] done 412s (random top-3 recovery 0.0029); alloc 2505 MB, peak 4753 MB
[14:12:01] [cross] all done 1257s

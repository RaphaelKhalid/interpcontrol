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

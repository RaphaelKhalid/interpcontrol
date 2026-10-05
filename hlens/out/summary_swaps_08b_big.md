## Swaps (08b_big)

prompts: 28; single-token OK: 28; model top-1 = answer: 19; also lens rank<=100 for intermediate at some L8-18 (kept): 16

| prompt | top1 | rank ans | best lens rank X (layer) | kept |
|---|---|---|---|---|
| The capital of the country where the Statue of Liberty is  | ' Paris' | 1 | 1 (L16) | True |
| The capital of the country where the Statue of Liberty is  | ' Rome' | 1 | 3 (L16) | True |
| The capital of the country where the Statue of Liberty is  | ' Berlin' | 1 | 1 (L15) | True |
| The capital of the country where the Statue of Liberty is  | ' Moscow' | 1 | 2 (L15) | True |
| The capital of the country where the Statue of Liberty is  | ' Beijing' | 1 | 1 (L15) | True |
| The capital of the country where the Statue of Liberty is  | ' Tokyo' | 1 | 1 (L16) | True |
| The capital of the country where the Statue of Liberty is  | ' Barcelona' | 3 | 6 (L16) | False |
| The capital of the country where the Statue of Liberty is  | ' Athens' | 1 | 1 (L14) | True |
| The capital of the country where the Statue of Liberty is  | ' Cairo' | 1 | 1 (L16) | True |
| The capital of the country where the Statue of Liberty is  | ' London' | 1 | 6 (L16) | True |
| The language spoken in the country whose capital is Lisbon | ' French' | 1 | 20 (L9) | True |
| The language spoken in the country whose capital is Lisbon | ' German' | 1 | 65 (L20) | True |
| The language spoken in the country whose capital is Lisbon | ' Italian' | 1 | 23 (L22) | False |
| The language spoken in the country whose capital is Lisbon | ' Spanish' | 1 | 11 (L20) | True |
| The language spoken in the country whose capital is Lisbon | ' Japanese' | 1 | 14 (L21) | True |
| The number of legs on the animal that barks is four. The n | ' three' | 8 | 1203 (L22) | False |
| The number of legs on the animal that barks is four. The n | ' two' | 6 | 174 (L13) | False |
| The number of legs on the animal that barks is four. The n | ' three' | 3 | 130 (L8) | False |
| The number of legs on the animal that barks is four. The n | ' two' | 1 | 326 (L17) | False |
| The color of the sky on a clear day is blue. The color of  | ' red' | 1 | 1 (L8) | True |
| The color of the sky on a clear day is blue. The color of  | ' red' | 2 | 9 (L22) | False |
| The color of the sky on a clear day is blue. The color of  | ' green' | 7 | 10 (L16) | False |
| The capital of the country where the Statue of Liberty is  | ' A' | 3 | 1 (L16) | False |
| The capital of the country where the Statue of Liberty is  | ' Pisa' | 3 | 2 (L16) | False |
| The language spoken in the country whose capital is Lisbon | ' Russian' | 1 | 17 (L21) | False |
| The language spoken in the country whose capital is Lisbon | ' Chinese' | 1 | 29 (L21) | True |
| The number of legs on the animal that barks is four. The n | ' three' | 3 | 452 (L16) | False |
| The color of the sky on a clear day is blue. The color of  | ' yellow' | 1 | 1 (L17) | True |

### T1: share of first-order error removed by the 2nd-order term (intermediate swap)

| subset | layer | alpha | n | median true dA | median |e1| | median |e2| | agg 1-sum|e2|/sum|e1| | median per-prompt share | corr(true,1st) | corr(true,2nd) | corr(true,lens1st) | median |D|/|h| | answer flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all-correct | 12 | 1.0 | 22 | -0.005 | 0.002 | 0.000 | 0.93 | 0.97 | 1.00 | 1.00 | 0.86 | 0.048 | 0/22 |
| all-correct | 12 | 2.0 | 22 | -0.009 | 0.009 | 0.001 | 0.89 | 0.95 | 0.99 | 1.00 | 0.87 | 0.097 | 0/22 |
| all-correct | 12 | 4.0 | 22 | 0.020 | 0.036 | 0.004 | 0.82 | 0.91 | 0.96 | 1.00 | 0.88 | 0.193 | 0/22 |
| all-correct | 12 | 8.0 | 22 | 0.040 | 0.133 | 0.044 | 0.61 | 0.84 | 0.85 | 0.95 | 0.91 | 0.386 | 0/22 |
| all-correct | 14 | 1.0 | 22 | 0.053 | 0.002 | 0.000 | 0.93 | 0.97 | 1.00 | 1.00 | 0.97 | 0.023 | 0/22 |
| all-correct | 14 | 2.0 | 22 | 0.119 | 0.009 | 0.000 | 0.87 | 0.95 | 1.00 | 1.00 | 0.97 | 0.047 | 0/22 |
| all-correct | 14 | 4.0 | 22 | 0.313 | 0.040 | 0.002 | 0.82 | 0.92 | 0.99 | 1.00 | 0.96 | 0.093 | 0/22 |
| all-correct | 14 | 8.0 | 22 | 0.874 | 0.186 | 0.019 | 0.76 | 0.84 | 0.97 | 1.00 | 0.96 | 0.186 | 1/22 |
| all-correct | 16 | 1.0 | 22 | 1.830 | 0.128 | 0.011 | 0.87 | 0.89 | 0.98 | 1.00 | 0.89 | 0.107 | 0/22 |
| all-correct | 16 | 2.0 | 22 | 3.696 | 0.456 | 0.154 | 0.52 | 0.75 | 0.95 | 1.00 | 0.88 | 0.213 | 4/22 |
| all-correct | 16 | 4.0 | 22 | 6.849 | 1.030 | 1.176 | -1.33 | 0.18 | 0.94 | 0.95 | 0.89 | 0.427 | 13/22 |
| all-correct | 16 | 8.0 | 22 | 11.015 | 2.347 | 9.178 | -3.77 | -1.51 | 0.91 | 0.81 | 0.92 | 0.854 | 21/22 |
| all-correct | 18 | 1.0 | 22 | 2.553 | 0.200 | 0.044 | 0.38 | 0.74 | 0.99 | 1.00 | 0.94 | 0.130 | 4/22 |
| all-correct | 18 | 2.0 | 22 | 4.895 | 0.498 | 0.532 | -0.96 | 0.10 | 0.96 | 0.97 | 0.94 | 0.260 | 9/22 |
| all-correct | 18 | 4.0 | 22 | 9.322 | 1.593 | 4.664 | -1.99 | -1.44 | 0.92 | 0.88 | 0.94 | 0.520 | 18/22 |
| all-correct | 18 | 8.0 | 22 | 15.707 | 6.027 | 20.911 | -2.84 | -2.31 | 0.87 | 0.75 | 0.95 | 1.040 | 20/22 |
| all-correct | 20 | 1.0 | 22 | 6.548 | 0.405 | 0.621 | -0.84 | -0.68 | 0.99 | 0.99 | 0.96 | 0.280 | 13/22 |
| all-correct | 20 | 2.0 | 22 | 12.620 | 1.184 | 3.564 | -1.15 | -0.71 | 0.97 | 0.94 | 0.97 | 0.561 | 20/22 |
| all-correct | 20 | 4.0 | 22 | 20.618 | 8.125 | 16.261 | -1.19 | -0.64 | 0.96 | 0.81 | 0.98 | 1.121 | 21/22 |
| all-correct | 20 | 8.0 | 22 | 29.054 | 29.435 | 57.857 | -1.44 | -1.07 | 0.93 | 0.59 | 0.95 | 2.243 | 21/22 |
| kept | 12 | 1.0 | 16 | 0.023 | 0.002 | 0.000 | 0.96 | 0.97 | 1.00 | 1.00 | 0.90 | 0.048 | 0/16 |
| kept | 12 | 2.0 | 16 | 0.058 | 0.009 | 0.001 | 0.93 | 0.95 | 0.99 | 1.00 | 0.92 | 0.097 | 0/16 |
| kept | 12 | 4.0 | 16 | 0.122 | 0.036 | 0.004 | 0.89 | 0.92 | 0.97 | 1.00 | 0.93 | 0.193 | 0/16 |
| kept | 12 | 8.0 | 16 | 0.218 | 0.133 | 0.044 | 0.61 | 0.84 | 0.94 | 0.98 | 0.96 | 0.386 | 0/16 |
| kept | 14 | 1.0 | 16 | 0.136 | 0.002 | 0.000 | 0.92 | 0.97 | 1.00 | 1.00 | 0.99 | 0.022 | 0/16 |
| kept | 14 | 2.0 | 16 | 0.281 | 0.008 | 0.000 | 0.87 | 0.95 | 1.00 | 1.00 | 0.99 | 0.044 | 0/16 |
| kept | 14 | 4.0 | 16 | 0.599 | 0.033 | 0.002 | 0.83 | 0.93 | 0.99 | 1.00 | 0.99 | 0.088 | 0/16 |
| kept | 14 | 8.0 | 16 | 1.379 | 0.135 | 0.012 | 0.82 | 0.88 | 0.99 | 1.00 | 0.99 | 0.175 | 1/16 |
| kept | 16 | 1.0 | 16 | 1.866 | 0.128 | 0.021 | 0.85 | 0.83 | 0.97 | 1.00 | 0.93 | 0.113 | 0/16 |
| kept | 16 | 2.0 | 16 | 3.884 | 0.456 | 0.221 | 0.44 | 0.69 | 0.94 | 0.99 | 0.91 | 0.226 | 4/16 |
| kept | 16 | 4.0 | 16 | 7.382 | 1.030 | 1.390 | -1.64 | -0.11 | 0.93 | 0.95 | 0.92 | 0.453 | 11/16 |
| kept | 16 | 8.0 | 16 | 11.957 | 3.518 | 10.438 | -3.71 | -2.35 | 0.90 | 0.79 | 0.94 | 0.906 | 16/16 |
| kept | 18 | 1.0 | 16 | 2.797 | 0.179 | 0.069 | 0.25 | 0.59 | 0.98 | 1.00 | 0.96 | 0.133 | 4/16 |
| kept | 18 | 2.0 | 16 | 5.481 | 0.588 | 0.532 | -0.99 | -0.26 | 0.95 | 0.96 | 0.94 | 0.266 | 7/16 |
| kept | 18 | 4.0 | 16 | 9.843 | 1.604 | 4.664 | -1.94 | -1.44 | 0.89 | 0.86 | 0.92 | 0.532 | 14/16 |
| kept | 18 | 8.0 | 16 | 16.291 | 7.377 | 20.911 | -2.69 | -2.31 | 0.87 | 0.73 | 0.93 | 1.065 | 16/16 |
| kept | 20 | 1.0 | 16 | 6.824 | 0.525 | 0.527 | -0.26 | -0.54 | 0.99 | 0.99 | 0.96 | 0.274 | 9/16 |
| kept | 20 | 2.0 | 16 | 12.742 | 2.153 | 2.747 | -0.48 | -0.52 | 0.96 | 0.95 | 0.97 | 0.549 | 15/16 |
| kept | 20 | 4.0 | 16 | 20.774 | 8.125 | 13.401 | -0.65 | -0.47 | 0.96 | 0.80 | 0.98 | 1.097 | 16/16 |
| kept | 20 | 8.0 | 16 | 29.054 | 28.521 | 51.608 | -0.88 | -0.61 | 0.96 | 0.53 | 0.96 | 2.194 | 16/16 |

### T2: interaction (double swap: intermediate + relation) vs D_A^T H D_B

| subset | layer | alpha | n | median |I| | median |I - pred| | agg 1-sum|I-pred|/sum|I| | corr(I,pred) | sign agree |
|---|---|---|---|---|---|---|---|---|
| all-correct | 12 | 1.0 | 22 | 0.001 | 0.000 | 0.84 | 0.98 | 1.00 |
| all-correct | 12 | 2.0 | 22 | 0.003 | 0.001 | 0.66 | 0.93 | 0.91 |
| all-correct | 12 | 4.0 | 22 | 0.020 | 0.007 | 0.34 | 0.78 | 0.82 |
| all-correct | 12 | 8.0 | 22 | 0.056 | 0.103 | -0.03 | 0.23 | 0.68 |
| all-correct | 14 | 1.0 | 22 | 0.001 | 0.000 | 0.80 | 0.98 | 0.95 |
| all-correct | 14 | 2.0 | 22 | 0.005 | 0.001 | 0.66 | 0.94 | 0.95 |
| all-correct | 14 | 4.0 | 22 | 0.020 | 0.013 | 0.31 | 0.77 | 0.95 |
| all-correct | 14 | 8.0 | 22 | 0.088 | 0.092 | -0.01 | 0.18 | 0.73 |
| all-correct | 16 | 1.0 | 22 | 0.037 | 0.012 | 0.66 | 0.95 | 0.77 |
| all-correct | 16 | 2.0 | 22 | 0.154 | 0.077 | 0.22 | 0.42 | 0.73 |
| all-correct | 16 | 4.0 | 22 | 0.274 | 0.714 | -0.53 | 0.28 | 0.64 |
| all-correct | 16 | 8.0 | 22 | 1.930 | 2.590 | -0.74 | 0.07 | 0.64 |
| all-correct | 18 | 1.0 | 22 | 0.037 | 0.030 | 0.59 | 0.96 | 0.77 |
| all-correct | 18 | 2.0 | 22 | 0.293 | 0.340 | -0.36 | 0.84 | 0.59 |
| all-correct | 18 | 4.0 | 22 | 0.618 | 1.459 | -2.97 | 0.11 | 0.32 |
| all-correct | 18 | 8.0 | 22 | 1.911 | 4.930 | -4.12 | -0.56 | 0.14 |
| all-correct | 20 | 1.0 | 22 | 0.101 | 0.078 | 0.10 | 0.64 | 0.68 |
| all-correct | 20 | 2.0 | 22 | 0.355 | 0.444 | -0.40 | 0.46 | 0.64 |
| all-correct | 20 | 4.0 | 22 | 0.375 | 2.882 | -3.46 | 0.06 | 0.41 |
| all-correct | 20 | 8.0 | 22 | 2.843 | 12.395 | -4.02 | -0.30 | 0.14 |
| kept | 12 | 1.0 | 16 | 0.001 | 0.000 | 0.84 | 0.98 | 1.00 |
| kept | 12 | 2.0 | 16 | 0.004 | 0.001 | 0.68 | 0.93 | 0.88 |
| kept | 12 | 4.0 | 16 | 0.026 | 0.010 | 0.42 | 0.81 | 0.81 |
| kept | 12 | 8.0 | 16 | 0.081 | 0.109 | -0.07 | 0.41 | 0.75 |
| kept | 14 | 1.0 | 16 | 0.001 | 0.000 | 0.79 | 0.98 | 0.94 |
| kept | 14 | 2.0 | 16 | 0.005 | 0.001 | 0.65 | 0.95 | 0.94 |
| kept | 14 | 4.0 | 16 | 0.020 | 0.017 | 0.34 | 0.82 | 0.94 |
| kept | 14 | 8.0 | 16 | 0.109 | 0.092 | -0.04 | 0.15 | 0.69 |
| kept | 16 | 1.0 | 16 | 0.037 | 0.016 | 0.65 | 0.95 | 0.75 |
| kept | 16 | 2.0 | 16 | 0.230 | 0.079 | 0.18 | 0.39 | 0.69 |
| kept | 16 | 4.0 | 16 | 0.306 | 0.714 | -0.63 | 0.26 | 0.69 |
| kept | 16 | 8.0 | 16 | 2.127 | 2.990 | -0.81 | 0.04 | 0.69 |
| kept | 18 | 1.0 | 16 | 0.064 | 0.039 | 0.60 | 0.97 | 0.75 |
| kept | 18 | 2.0 | 16 | 0.363 | 0.435 | -0.27 | 0.87 | 0.50 |
| kept | 18 | 4.0 | 16 | 0.868 | 2.152 | -2.67 | 0.13 | 0.31 |
| kept | 18 | 8.0 | 16 | 2.053 | 7.160 | -4.32 | -0.51 | 0.19 |
| kept | 20 | 1.0 | 16 | 0.079 | 0.075 | 0.26 | 0.75 | 0.75 |
| kept | 20 | 2.0 | 16 | 0.196 | 0.337 | -0.31 | 0.60 | 0.69 |
| kept | 20 | 4.0 | 16 | 0.399 | 2.379 | -3.78 | 0.05 | 0.44 |
| kept | 20 | 8.0 | 16 | 2.735 | 12.395 | -4.17 | -0.33 | 0.12 |

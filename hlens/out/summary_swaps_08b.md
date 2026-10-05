## Swaps (08b)

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
| all-correct | 8 | 0.5 | 22 | -0.009 | 0.002 | 0.000 | 0.98 | 0.97 | 1.00 | 1.00 | 0.88 | 0.041 | 0/22 |
| all-correct | 8 | 1.0 | 22 | -0.016 | 0.006 | 0.001 | 0.95 | 0.96 | 1.00 | 1.00 | 0.88 | 0.082 | 0/22 |
| all-correct | 8 | 2.0 | 22 | -0.029 | 0.026 | 0.004 | 0.91 | 0.92 | 0.98 | 1.00 | 0.88 | 0.164 | 0/22 |
| all-correct | 10 | 0.5 | 22 | -0.000 | 0.001 | 0.000 | 0.98 | 0.98 | 1.00 | 1.00 | 0.89 | 0.030 | 0/22 |
| all-correct | 10 | 1.0 | 22 | -0.000 | 0.003 | 0.000 | 0.97 | 0.96 | 1.00 | 1.00 | 0.89 | 0.060 | 0/22 |
| all-correct | 10 | 2.0 | 22 | 0.000 | 0.012 | 0.001 | 0.93 | 0.95 | 0.99 | 1.00 | 0.89 | 0.121 | 0/22 |
| all-correct | 12 | 0.5 | 22 | -0.003 | 0.001 | 0.000 | 0.96 | 0.97 | 1.00 | 1.00 | 0.85 | 0.024 | 0/22 |
| all-correct | 12 | 1.0 | 22 | -0.005 | 0.002 | 0.000 | 0.93 | 0.97 | 1.00 | 1.00 | 0.86 | 0.048 | 0/22 |
| all-correct | 12 | 2.0 | 22 | -0.009 | 0.009 | 0.001 | 0.89 | 0.95 | 0.99 | 1.00 | 0.87 | 0.097 | 0/22 |
| all-correct | 14 | 0.5 | 22 | 0.025 | 0.001 | 0.000 | 0.96 | 0.97 | 1.00 | 1.00 | 0.97 | 0.012 | 0/22 |
| all-correct | 14 | 1.0 | 22 | 0.053 | 0.002 | 0.000 | 0.93 | 0.97 | 1.00 | 1.00 | 0.97 | 0.023 | 0/22 |
| all-correct | 14 | 2.0 | 22 | 0.119 | 0.009 | 0.000 | 0.87 | 0.95 | 1.00 | 1.00 | 0.97 | 0.047 | 0/22 |
| all-correct | 16 | 0.5 | 22 | 0.879 | 0.031 | 0.002 | 0.92 | 0.95 | 0.99 | 1.00 | 0.91 | 0.053 | 0/22 |
| all-correct | 16 | 1.0 | 22 | 1.830 | 0.128 | 0.011 | 0.87 | 0.89 | 0.98 | 1.00 | 0.89 | 0.107 | 0/22 |
| all-correct | 16 | 2.0 | 22 | 3.696 | 0.456 | 0.154 | 0.52 | 0.75 | 0.95 | 1.00 | 0.88 | 0.213 | 0/22 |
| kept | 8 | 0.5 | 16 | 0.007 | 0.002 | 0.000 | 0.98 | 0.97 | 1.00 | 1.00 | 0.93 | 0.041 | 0/16 |
| kept | 8 | 1.0 | 16 | 0.015 | 0.009 | 0.001 | 0.96 | 0.96 | 1.00 | 1.00 | 0.93 | 0.082 | 0/16 |
| kept | 8 | 2.0 | 16 | 0.036 | 0.035 | 0.004 | 0.91 | 0.92 | 0.99 | 1.00 | 0.93 | 0.164 | 0/16 |
| kept | 10 | 0.5 | 16 | 0.005 | 0.001 | 0.000 | 0.99 | 0.98 | 1.00 | 1.00 | 0.92 | 0.032 | 0/16 |
| kept | 10 | 1.0 | 16 | 0.010 | 0.005 | 0.000 | 0.97 | 0.97 | 1.00 | 1.00 | 0.93 | 0.063 | 0/16 |
| kept | 10 | 2.0 | 16 | 0.021 | 0.021 | 0.001 | 0.94 | 0.95 | 0.99 | 1.00 | 0.93 | 0.127 | 0/16 |
| kept | 12 | 0.5 | 16 | 0.010 | 0.001 | 0.000 | 0.98 | 0.98 | 1.00 | 1.00 | 0.90 | 0.024 | 0/16 |
| kept | 12 | 1.0 | 16 | 0.023 | 0.002 | 0.000 | 0.96 | 0.97 | 1.00 | 1.00 | 0.90 | 0.048 | 0/16 |
| kept | 12 | 2.0 | 16 | 0.058 | 0.009 | 0.001 | 0.93 | 0.95 | 0.99 | 1.00 | 0.92 | 0.097 | 0/16 |
| kept | 14 | 0.5 | 16 | 0.067 | 0.000 | 0.000 | 0.96 | 0.96 | 1.00 | 1.00 | 0.99 | 0.011 | 0/16 |
| kept | 14 | 1.0 | 16 | 0.136 | 0.002 | 0.000 | 0.92 | 0.97 | 1.00 | 1.00 | 0.99 | 0.022 | 0/16 |
| kept | 14 | 2.0 | 16 | 0.281 | 0.008 | 0.000 | 0.87 | 0.95 | 1.00 | 1.00 | 0.99 | 0.044 | 0/16 |
| kept | 16 | 0.5 | 16 | 0.929 | 0.035 | 0.003 | 0.91 | 0.93 | 0.99 | 1.00 | 0.95 | 0.057 | 0/16 |
| kept | 16 | 1.0 | 16 | 1.866 | 0.128 | 0.021 | 0.85 | 0.83 | 0.97 | 1.00 | 0.93 | 0.113 | 0/16 |
| kept | 16 | 2.0 | 16 | 3.884 | 0.456 | 0.221 | 0.44 | 0.69 | 0.94 | 0.99 | 0.91 | 0.226 | 0/16 |

### T2: interaction (double swap: intermediate + relation) vs D_A^T H D_B

| subset | layer | alpha | n | median |I| | median |I - pred| | agg 1-sum|I-pred|/sum|I| | corr(I,pred) | sign agree |
|---|---|---|---|---|---|---|---|---|
| all-correct | 8 | 0.5 | 22 | 0.000 | 0.000 | 0.91 | 1.00 | 1.00 |
| all-correct | 8 | 1.0 | 22 | 0.000 | 0.000 | 0.79 | 0.99 | 1.00 |
| all-correct | 8 | 2.0 | 22 | 0.001 | 0.001 | 0.52 | 0.93 | 0.73 |
| all-correct | 10 | 0.5 | 22 | 0.000 | 0.000 | 0.91 | 1.00 | 0.95 |
| all-correct | 10 | 1.0 | 22 | 0.001 | 0.000 | 0.82 | 0.99 | 0.91 |
| all-correct | 10 | 2.0 | 22 | 0.002 | 0.001 | 0.71 | 0.97 | 0.95 |
| all-correct | 12 | 0.5 | 22 | 0.000 | 0.000 | 0.92 | 1.00 | 1.00 |
| all-correct | 12 | 1.0 | 22 | 0.001 | 0.000 | 0.84 | 0.98 | 1.00 |
| all-correct | 12 | 2.0 | 22 | 0.003 | 0.001 | 0.66 | 0.93 | 0.91 |
| all-correct | 14 | 0.5 | 22 | 0.000 | 0.000 | 0.90 | 1.00 | 0.95 |
| all-correct | 14 | 1.0 | 22 | 0.001 | 0.000 | 0.80 | 0.98 | 0.95 |
| all-correct | 14 | 2.0 | 22 | 0.005 | 0.001 | 0.66 | 0.94 | 0.95 |
| all-correct | 16 | 0.5 | 22 | 0.008 | 0.002 | 0.82 | 0.98 | 0.95 |
| all-correct | 16 | 1.0 | 22 | 0.037 | 0.012 | 0.66 | 0.95 | 0.77 |
| all-correct | 16 | 2.0 | 22 | 0.154 | 0.077 | 0.22 | 0.42 | 0.73 |
| kept | 8 | 0.5 | 16 | 0.000 | 0.000 | 0.90 | 1.00 | 1.00 |
| kept | 8 | 1.0 | 16 | 0.000 | 0.000 | 0.75 | 0.99 | 1.00 |
| kept | 8 | 2.0 | 16 | 0.001 | 0.002 | 0.41 | 0.93 | 0.62 |
| kept | 10 | 0.5 | 16 | 0.000 | 0.000 | 0.91 | 1.00 | 1.00 |
| kept | 10 | 1.0 | 16 | 0.001 | 0.000 | 0.81 | 0.99 | 0.94 |
| kept | 10 | 2.0 | 16 | 0.004 | 0.001 | 0.69 | 0.97 | 1.00 |
| kept | 12 | 0.5 | 16 | 0.000 | 0.000 | 0.92 | 1.00 | 1.00 |
| kept | 12 | 1.0 | 16 | 0.001 | 0.000 | 0.84 | 0.98 | 1.00 |
| kept | 12 | 2.0 | 16 | 0.004 | 0.001 | 0.68 | 0.93 | 0.88 |
| kept | 14 | 0.5 | 16 | 0.000 | 0.000 | 0.89 | 0.99 | 0.94 |
| kept | 14 | 1.0 | 16 | 0.001 | 0.000 | 0.79 | 0.98 | 0.94 |
| kept | 14 | 2.0 | 16 | 0.005 | 0.001 | 0.65 | 0.95 | 0.94 |
| kept | 16 | 0.5 | 16 | 0.008 | 0.002 | 0.81 | 0.98 | 0.94 |
| kept | 16 | 1.0 | 16 | 0.037 | 0.016 | 0.65 | 0.95 | 0.75 |
| kept | 16 | 2.0 | 16 | 0.230 | 0.079 | 0.18 | 0.39 | 0.69 |

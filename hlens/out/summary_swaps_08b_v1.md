## Swaps (08b_v1)

prompts: 22; single-token OK: 22; model top-1 = answer: 1; also lens rank<=100 for intermediate at some L8-18 (kept): 1

| prompt | top1 | rank ans | best lens rank X (layer) | kept |
|---|---|---|---|---|
| The capital of the country where the Eiffel Tower is locat | ':' | 10 | 5 (L20) | False |
| The capital of the country where the Colosseum is located  | ':' | 17 | 16 (L22) | False |
| The capital of the country where the Brandenburg Gate is l | ':' | 11 | 4 (L20) | False |
| The capital of the country where the Kremlin is located is | ':' | 16 | 2 (L20) | False |
| The capital of the country where the Great Wall is located | ':' | 45 | 19 (L22) | False |
| The capital of the country where Mount Fuji is located is | '\n' | 22 | 3 (L20) | False |
| The capital of the country where the Sagrada Familia is lo | ':' | 51 | 10 (L20) | False |
| The capital of the country where the Acropolis is located  | ':' | 32 | 20 (L20) | False |
| The capital of the country where the pyramids of Giza are  | ':' | 19 | 3 (L20) | False |
| The capital of the country where Big Ben is located is | ':' | 7 | 7 (L20) | False |
| The language spoken in the country whose capital is Paris  | ':' | 6 | 8 (L8) | False |
| The language spoken in the country whose capital is Berlin | ' German' | 1 | 89 (L8) | True |
| The language spoken in the country whose capital is Rome i | ' known' | 22 | 87 (L22) | False |
| The language spoken in the country whose capital is Madrid | ':' | 6 | 10 (L8) | False |
| The language spoken in the country whose capital is Tokyo  | ' known' | 12 | 111 (L21) | False |
| The number of legs on the animal that spins webs is | ' ' | 239 | 760 (L7) | False |
| The number of legs on the insect that makes honey is | ' ' | 104 | 372 (L9) | False |
| The number of legs on the animal that says moo is | ' ' | 32 | 73 (L7) | False |
| The number of legs on the bird that lays eggs for breakfas | ' ' | 55 | 907 (L12) | False |
| The color of the fruit that keeps the doctor away is | ':' | 2 | 4 (L7) | False |
| The color of the fruit that monkeys love to eat is | ' a' | 11 | 71 (L22) | False |
| The color of the vegetable that rabbits love to eat is | ' green' | 16 | 79 (L22) | False |

### T1: share of first-order error removed by the 2nd-order term (intermediate swap)

| subset | layer | alpha | n | median true dA | median |e1| | median |e2| | agg 1-sum|e2|/sum|e1| | median per-prompt share | corr(true,1st) | corr(true,2nd) | corr(true,lens1st) | median |D|/|h| | answer flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

### T2: interaction (double swap: intermediate + relation) vs D_A^T H D_B

| subset | layer | alpha | n | median |I| | median |I - pred| | agg 1-sum|I-pred|/sum|I| | corr(I,pred) | sign agree |
|---|---|---|---|---|---|---|---|---|

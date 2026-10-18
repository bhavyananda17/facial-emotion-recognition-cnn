# ck: paper vs clean protocol over 4 training seeds

| Seed | Paper acc | Clean acc | Gap (paper - clean) | Paper epochs | Clean epochs |
|---|---|---|---|---|---|
| 1 | 94.4% | 96.0% | -1.5 pts | 64 | 100 |
| 2 | 94.9% | 92.9% | +2.0 pts | 100 | 100 |
| 3 | 95.5% | 96.5% | -1.0 pts | 78 | 96 |
| 42 (headline) | 94.4% | 90.9% | +3.5 pts | 100 | 56 |

- Paper protocol: mean **94.8%**, std 0.5, range 94.4 to 95.5
- Clean protocol: mean **94.1%**, std 2.6, range 90.9 to 96.5
- Gap: mean **+0.8 pts**, std 2.4, range -1.5 to +3.5; paper protocol higher in 2 of 4 seeds

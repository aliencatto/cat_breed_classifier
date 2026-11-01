| Run | Change vs. baseline | Best val acc | Train acc (same epoch) | Train-val gap | Best epoch |
|---|---|---|---|---|---|
| `cnn` | baseline | 0.529 | 0.721 | +0.191 | 51 |
| `cnn_dropout05` | dropout=0.5 | 0.597 | 0.626 | +0.029 | 43 |
| `cnn_noaug` | no_aug=True | 0.647 | 1.000 | +0.353 | 56 |

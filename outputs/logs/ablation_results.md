| Experiment | Setting | Params | Train loss | Val loss | Val PPL | Train time (s) | Tokens/s | Peak RSS (MB) | Attn-score mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| Depth (n_layers) | n_layers=2 | 413,440 | 2.2056 | 2.2290 | 9.29 | 221 | 9,255 | 366 | 8.0 |
| Depth (n_layers) | n_layers=4 | 809,984 | 2.1030 | 2.1327 | 8.44 | 448 | 4,574 | 423 | 16.0 |
| Depth (n_layers) | n_layers=6 | 1,206,528 | 2.0597 | 2.1048 | 8.21 | 666 | 3,073 | 480 | 24.0 |
| Heads (n_heads) | n_heads=2 | 809,984 | 2.1180 | 2.1485 | 8.57 | 422 | 4,848 | 422 | 8.0 |
| Heads (n_heads) | n_heads=4 | 809,984 | 2.1030 | 2.1327 | 8.44 | 448 | 4,574 | 423 | 16.0 |
| Heads (n_heads) | n_heads=8 | 809,984 | 2.1118 | 2.1430 | 8.53 | 466 | 4,398 | 453 | 32.0 |
| Context length | context_length=64 | 809,984 | 2.0904 | 2.1070 | 8.22 | 209 | 4,892 | 376 | 4.0 |
| Context length | context_length=128 | 809,984 | 2.1030 | 2.1327 | 8.44 | 448 | 4,574 | 423 | 16.0 |
| Context length | context_length=256 | 809,984 | 2.1533 | 2.2073 | 9.09 | 828 | 4,948 | 626 | 64.0 |
| Position method | pos_encoding=sinusoidal | 809,984 | 2.1030 | 2.1327 | 8.44 | 448 | 4,574 | 423 | 16.0 |
| Position method | pos_encoding=learned | 826,368 | 2.2178 | 2.2358 | 9.35 | 402 | 5,092 | 421 | 16.0 |
| Learning rate | lr=0.0003 | 809,984 | 2.3733 | 2.3690 | 10.69 | 740 | 2,769 | 425 | 16.0 |
| Learning rate | lr=0.001 | 809,984 | 2.1030 | 2.1327 | 8.44 | 448 | 4,574 | 423 | 16.0 |
| Seed (noise floor) | seed=1337 | 809,984 | 2.1030 | 2.1327 | 8.44 | 448 | 4,574 | 423 | 16.0 |
| Seed (noise floor) | seed=2 | 809,984 | 2.1089 | 2.1431 | 8.53 | 770 | 2,661 | 417 | 16.0 |

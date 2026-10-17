# Data

`tinyshakespeare.txt` is downloaded automatically on first use by `src/dataset.py` from
<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>,
the raw file behind the Hugging Face [`tiny_shakespeare`](https://huggingface.co/datasets/karpathy/tiny_shakespeare) dataset
(1,115,394 characters, ~40,000 lines, 65 unique characters). Public domain text. It is git-ignored.

Split: first 90% of the characters = train (1,003,854), last 10% = validation (111,540). Contiguous, so no
validation window overlaps a training window except at the single boundary.

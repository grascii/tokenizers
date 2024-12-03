# tokenizers

Training scripts for Grascii tokenizers typically used for machine learning
models.

## v1

This tokenizer operates on normalized Grascii and is intended for use with a
Roberta model. It is trained on the
[gregg-preanniversary-words](https://huggingface.co/grascii/gregg-preanniversary-words)
dataset.

The X and XS strokes are encoded as S and SS respectively due to their high
visual similarity.

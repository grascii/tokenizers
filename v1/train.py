from datasets import load_dataset
from tokenizers.pre_tokenizers import Split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.decoders import Fuse
from tokenizers.processors import RobertaProcessing
from tokenizers.normalizers import Replace
from transformers import PreTrainedTokenizerFast


dataset = load_dataset(
    "grascii/gregg-preanniversary-words",
    split="train",
    revision="0227610b8aa2cd5587fe6c247b355746825b8b3c",
    token="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",
)


def batch_iterator(batch_size=512):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["grascii_normalized"]


tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
tokenizer.normalizer = Replace("X", "S")
tokenizer.pre_tokenizer = Split("-", behavior="removed")
tokenizer.decoder = Fuse()
tokenizer.post_processor = RobertaProcessing(
        ("</s>", 2), ("<s>", 0), add_prefix_space=False)

special_tokens = {
    "bos_token": "<s>",
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

trainer = WordLevelTrainer(special_tokens=list(special_tokens.values()))

tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset),
)

transformers_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    **special_tokens,
)
transformers_tokenizer.save_pretrained("tokenizer")

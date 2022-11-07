import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class TwoModalDataset(Dataset):
    def __init__(
        self,
        texts,
        contexts,
        labels,
        max_seq_len,
        pretrained_model_name_or_path="bert-base-uncased",
    ):
        self.texts = texts
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        encoding = self.tokenizer.encode_plus(
            text=text,  # emails to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_seq_len,  # Pad & truncate all sentences.
            padding="max_length",
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )
        input_ids = encoding["input_ids"]
        attention_masks = encoding["attention_mask"]
        return input_ids, attention_masks

    def __getitem__(self, item):
        text = str(self.texts[item])
        context = str(self.contexts[item])
        label = self.labels[item]

        input_ids_text, attention_masks_text = self.tokenize(text)
        input_ids_context, attention_masks_context = self.tokenize(context)

        return {
            "text_text": text,
            "text_input_ids": input_ids_text.flatten(),
            "text_attention_mask": attention_masks_text.flatten(),
            "context_text": text,
            "context_input_ids": input_ids_context.flatten(),
            "context_attention_mask": attention_masks_context.flatten(),
            "labels": torch.as_tensor(label),
        }

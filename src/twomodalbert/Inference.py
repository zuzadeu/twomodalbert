from configparser import ConfigParser
from transformers import BertTokenizer
import torch

config = ConfigParser()
config.read("config.ini")


def test_model(model, data_loader, device):
    model = model.eval()

    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            text_input_ids = d["text_input_ids"].to(device)
            text_attention_mask = d["text_attention_mask"].to(device)
            context_input_ids = d["context_input_ids"].to(device)
            context_attention_mask = d["context_attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                text_input_ids,
                text_attention_mask,
                context_input_ids,
                context_attention_mask,
            )

            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            real_values.extend(labels)

    return predictions, real_values


def tokenize(text, device):
    tokenizer = BertTokenizer.from_pretrained(config["GENERAL"]["PRETRAINED_MODEL_NAME_OR_PATH"])
    encoding = tokenizer.encode_plus(
        text=text,  # emails to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=int(
            config["GENERAL"]["MAX_SEQ_LEN"]
        ),  # Pad & truncate all sentences.
        padding="max_length",
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors="pt",  # Return pytorch tensors.
        truncation=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)
    return input_ids, attention_masks


def predict_on_text(model, text, context, device):
    input_ids_text, attention_masks_text = tokenize(text, device)
    input_ids_context, attention_masks_context = tokenize(context, device)

    outputs = model(
        input_ids_text, attention_masks_text, input_ids_context, attention_masks_context
    )

    _, preds = torch.max(outputs, dim=1)

    return int(preds[0])

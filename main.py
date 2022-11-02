from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from TwoModalBert import TwoModalBERTModel
from TwoModalDataset import TwoModalDataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
RANDOM_SEED = 42
BATCH_SIZE = 16
MAX_SEQ_LEN = 200


def _create_data_loader(
    df,
    line_column,
    context_column,
    label_column,
    num_workers=2,
    pretrained_model_name_or_path="bert-base-uncased",
):

    ds = TwoModalDataset(
        lines=df[line_column].to_numpy(),
        contexts=df[context_column].to_numpy(),
        speakers=df[label_column].to_numpy(),
        max_seq_len=MAX_SEQ_LEN,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=num_workers)


def prepare_data(df, line_column, context_column, label_column, train_size, val_size):
    train, test = train_test_split(
        df,
        test_size=1 - train_size,
        random_state=RANDOM_SEED,
        stratify=df[[label_column]],
    )
    test, val = train_test_split(
        test,
        test_size=val_size / (1 - train_size),
        random_state=RANDOM_SEED,
        stratify=test[[label_column]],
    )

    train_data_loader = _create_data_loader(
        train, line_column, context_column, label_column
    )
    val_data_loader = _create_data_loader(
        val, line_column, context_column, label_column
    )
    test_data_loader = _create_data_loader(
        test, line_column, context_column, label_column
    )
    return train_data_loader, train, val_data_loader, val, test_data_loader, test


def _send_to_device(d):
    text_input_ids = d["text_input_ids"].to(DEVICE)
    text_attention_mask = d["text_attention_mask"].to(DEVICE)
    context_input_ids = d["context_input_ids"].to(DEVICE)
    context_attention_mask = d["context_attention_mask"].to(DEVICE)
    labels = d["labels"].to(DEVICE)
    return (
        text_input_ids,
        text_attention_mask,
        context_input_ids,
        context_attention_mask,
        labels,
    )


def train_epoch(model, data_loader, loss_fn, optimizer, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        (
            text_input_ids,
            text_attention_mask,
            context_input_ids,
            context_attention_mask,
            labels,
        ) = _send_to_device(d)

        outputs = model(
            text_input_ids,
            text_attention_mask,
            context_input_ids,
            context_attention_mask,
        )

        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            (
                text_input_ids,
                text_attention_mask,
                context_input_ids,
                context_attention_mask,
                labels,
            ) = _send_to_device(d)

            outputs = model(
                text_input_ids,
                text_attention_mask,
                context_input_ids,
                context_attention_mask,
            )

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(
    text_size, context_size, train_data_loader, train, val_data_loader, val
):
    history = defaultdict(list)
    print(f"LINE SIZE = {text_size}, CONTEXT SIZE = {context_size}")
    model = TwoModalBERTModel(line_hs_size=text_size, context_hs_size=context_size)
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    best_accuracy = 0

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, len(train)
        )

        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(val))

        print(f"Val loss {val_loss} accuracy {val_acc}")

        print()

        history[str(context_size)]["train_acc"].append(train_acc)
        history[str(context_size)]["train_loss"].append(train_loss)
        history[str(context_size)]["val_acc"].append(val_acc)
        history[str(context_size)]["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(
                model.state_dict(),
                f"/content/drive/MyDrive/TheOffice/best_model_state_{context_size}.bin",
            )
            best_accuracy = val_acc

        return model


train_data_loader, train, val_data_loader, val, test_data_loader, test = prepare_data(
    df, line_column, context_column, label_column, train_size, val_size
)
model = train_model(
    train_data_loader,
    train,
    val_data_loader,
    val,
    text_size=200,
    context_size=200,
)

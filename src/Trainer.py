from collections import defaultdict
from torch import nn
from transformers import AdamW
from src.Model import TwoModalBERTModel
import numpy as np
import torch


class TwoModalBertTrainer:
    def __init__(self, device, config):
        self.device = device
        self.epochs = int(config["GENERAL"]["EPOCHS"])
        self.model_save_path = config["GENERAL"]["MODEL_SAVE_PATH"]
        self.pretrained_model_name_or_path = config["GENERAL"][
            "PRETRAINED_MODEL_NAME_OR_PATH"
        ]

    def _send_to_device(self, d):
        text_input_ids = d["text_input_ids"].to(self.device)
        text_attention_mask = d["text_attention_mask"].to(self.device)
        context_input_ids = d["context_input_ids"].to(self.device)
        context_attention_mask = d["context_attention_mask"].to(self.device)
        labels = d["labels"].to(self.device)
        return (
            text_input_ids,
            text_attention_mask,
            context_input_ids,
            context_attention_mask,
            labels,
        )

    def train_epoch(self, model, data_loader, loss_fn, optimizer, n_examples):
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
            ) = self._send_to_device(d)

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

    def eval_model(self, model, data_loader, loss_fn, n_examples):
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
                ) = self._send_to_device(d)

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
        self,
        train_data_loader,
        train,
        val_data_loader,
        val,
        text_size,
        context_size,
        binary=False,
        text_p=0.3,
        context_p=0.3,
        output_p=0.3,
    ):
        history = defaultdict(list)
        print(f"LINE SIZE = {text_size}, CONTEXT SIZE = {context_size}")
        model = TwoModalBERTModel(
            text_size=text_size,
            context_size=context_size,
            binary=binary,
            text_p=text_p,
            context_p=context_p,
            output_p=output_p,
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
        )
        model = model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

        loss_fn = nn.CrossEntropyLoss().to(self.device)

        best_accuracy = 0

        for epoch in range(self.epochs):

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print("-" * 10)

            train_acc, train_loss = self.train_epoch(
                model, train_data_loader, loss_fn, optimizer, len(train)
            )

            print(f"Train loss {train_loss} accuracy {train_acc}")

            val_acc, val_loss = self.eval_model(
                model, val_data_loader, loss_fn, len(val)
            )

            print(f"Val loss {val_loss} accuracy {val_acc}")

            print()

            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(
                    model.state_dict(), self.model_save_path,
                )
                best_accuracy = val_acc

        return model, history

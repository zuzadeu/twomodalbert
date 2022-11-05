from torch import nn
from transformers import BertForSequenceClassification
import torch


class TwoModalBERTModel(nn.Module):
    def __init__(
        self,
        text_hs_size,
        context_hs_size,
        binary=False,
        text_p=0.3,
        context_p=0.3,
        output_p=0.3,
        pretrained_model_name_or_path="bert-base-uncased",
    ):
        super(TwoModalBERTModel, self).__init__()
        self.text_hs_size = text_hs_size
        self.context_hs_size = context_hs_size
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, output_hidden_states=True
        )
        # Reduce hidden state size of Bert output:
        self.linear_text = nn.Linear(self.model.config.hidden_size, self.text_hs_size)
        self.linear_context = nn.Linear(
            self.model.config.hidden_size, self.context_hs_size
        )
        self.dropout_text = nn.Dropout(p=text_p)
        self.dropout_context = nn.Dropout(p=context_p)
        # Add Linear layer that concatenates text and context embeddings
        self.linear_categorical_output = nn.Linear(
            self.text_hs_size + self.context_hs_size, 12
        )
        self.dropout_last = nn.Dropout(p=output_p)
        if binary:
            self.activation = nn.Sigmoid(dim=1)
        else:
            self.activation = nn.Softmax(dim=1)

    def create_mode(self, input_ids, attention_mask, layer_to_reduce_hs, dropout):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs["hidden_states"][-1]
        cls_hidden_states = last_hidden_states[:, 0, :]
        last_hs_reduced = layer_to_reduce_hs(cls_hidden_states)
        last_hs_reduced = dropout(last_hs_reduced)
        return last_hs_reduced

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        context_input_ids,
        context_attention_mask,
    ):
        mode_text = self.create_mode(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            layer_to_reduce_hs=self.linear_text,
            dropout=self.dropout_text,
        )
        mode_context = self.create_mode(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            layer_to_reduce_hs=self.linear_context,
            dropout=self.dropout_context,
        )
        concatenated_modes = torch.cat([mode_text, mode_context], dim=1)
        categorical_output = self.linear_categorical_output(concatenated_modes)
        categorical_output = self.dropout_last(categorical_output)
        activation_output = self.activation(categorical_output)
        return activation_output

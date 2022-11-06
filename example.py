from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from src.DataPreparation import TwoModalDataPreparation
from src.Trainer import TwoModalBertTrainer
import os
import pandas as pd
import torch

from src.Model import TwoModalBERTModel
from configparser import ConfigParser
from src.Inference import predict_on_text, test_model

import seaborn as sns
import matplotlib.pyplot as plt

# load data
src_df_path = "/content/drive/MyDrive/TheOffice/The-Office-Lines-V4.csv"
src_df = pd.read_csv(src_df_path)
src_df["context"] = src_df["line"].shift(1)
df = src_df[["line", "context", "speaker"]]
SELECTED_SPEAKERS = [
    "Michael",
    "Jim",
    "Pam",
    "Dwight",
    "Jan",
    "Phyllis",
    "Stanley",
    "Oscar",
    "Angela",
    "Kevin",
    "Ryan",
    "Creed",
]
df = df[df["speaker"].isin(SELECTED_SPEAKERS)]
le = preprocessing.LabelEncoder()
df["label"] = le.fit_transform(df["speaker"])


# create data loaders
(
    train_data_loader,
    train,
    val_data_loader,
    val,
    test_data_loader,
    test,
) = DataPreparation.prepare_data(
    df,
    text_column="line",
    context_column="context",
    label_column="label",
    train_size=0.8,
    val_size=0.1,
)


# train model
model, history = Trainer.train_model(
    train_data_loader,
    train,
    val_data_loader,
    val,
    text_size=200,
    context_size=200,
    binary=False,
    text_p=0.3,
    context_p=0.3,
    output_p=0.3,
)


# load model
model = TwoModalBERTModel(
    text_size=200,
    context_size=200,
    binary=False,
    text_p=0.3,
    context_p=0.3,
    output_p=0.3,
)
model.load_state_dict(torch.load(config["GENERAL"]["MODEL_SAVE_PATH"]))


# evaluate on a test set
y_pred, y_test = test_model(model, test_data_loader)
y_pred, y_test = [e.cpu() for e in y_pred], [e.cpu() for e in y_test]


# print confusion matrix - helper function
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True speaker")
    plt.xlabel("Predicted speaker")


cm = confusion_matrix(y_test, y_pred)
speakers_labels = (
    df[["speaker", "label"]].groupby(["speaker"]).agg("max").to_dict()["label"]
)
df_cm = pd.DataFrame(cm, index=speakers_labels, columns=speakers_labels)
show_confusion_matrix(df_cm)


# run on new pair of texts
line = "Dwight is my best friend."
context = "What do you think about Dwight?"

predict_on_text(model, line, context)
# output 12

speakers_mapping = {y: x for x, y in speakers_labels.items()}
speakers_mapping[12]

from sklearn import preprocessing

from src.DataPreparation import TwoModalDataPreparation
from src.Trainer import TwoModalBertTrainer
import os
import pandas as pd
import torch

# define constants
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
RANDOM_SEED = 42
BATCH_SIZE = 16
MAX_SEQ_LEN = 200
NUM_WORKERS = 2
PRETRAINED_MODEL_NAME_AND_PATH = "bert-base-uncased"
MODEL_SAVE_PATH = "/content/drive/MyDrive/TheOffice/best_model_state.bin"


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


# intialize modules
DataPreparation = TwoModalDataPreparation(
    max_seq_len=MAX_SEQ_LEN,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    random_seed=RANDOM_SEED,
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_AND_PATH,
)


Trainer = TwoModalBertTrainer(device=DEVICE, epochs=EPOCHS, model_save_path=MODEL_SAVE_PATH, pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_AND_PATH)

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
model = Trainer.train_model(
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

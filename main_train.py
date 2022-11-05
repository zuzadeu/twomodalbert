from src.DataPreparation import train_model, prepare_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
RANDOM_SEED = 42
BATCH_SIZE = 16
MAX_SEQ_LEN = 200
MODEL_SAVE_PATH = "/content/drive/MyDrive/TheOffice/best_model_state.bin"


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

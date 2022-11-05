from src.Dataset import TwoModalDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class TwoModalDataPreparation:
    def __init__(
        self,
        config,
    ):
        self.max_seq_len = int(config['GENERAL']['MAX_SEQ_LEN'])
        self.batch_size = int(config['GENERAL']['BATCH_SIZE'])
        self.num_workers = int(config['GENERAL']['NUM_WORKERS'])
        self.random_seed = int(config['GENERAL']['RANDOM_SEED'])
        self.pretrained_model_name_or_path = config['GENERAL']['PRETRAINED_MODEL_NAME_OR_PATH']

    def _create_data_loader(self, df, text_column, context_column, label_column):

        ds = TwoModalDataset(
            texts=df[text_column].to_numpy(),
            contexts=df[context_column].to_numpy(),
            labels=df[label_column].to_numpy(),
            max_seq_len=self.max_seq_len,
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
        )

        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def prepare_data(
        self, df, text_column, context_column, label_column, train_size, val_size
    ):
        train, test = train_test_split(
            df,
            test_size=1 - train_size,
            random_state=self.random_seed,
            stratify=df[[label_column]],
        )
        test, val = train_test_split(
            test,
            test_size=val_size / (1 - train_size),
            random_state=self.random_seed,
            stratify=test[[label_column]],
        )

        train_data_loader = self._create_data_loader(
            train, text_column, context_column, label_column
        )
        val_data_loader = self._create_data_loader(
            val, text_column, context_column, label_column
        )
        test_data_loader = self._create_data_loader(
            test, text_column, context_column, label_column
        )
        return train_data_loader, train, val_data_loader, val, test_data_loader, test

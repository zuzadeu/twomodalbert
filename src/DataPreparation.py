from src.Dataset import TwoModalDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class TwoModalDataPreparation:
    def __init__(
        self,
        max_seq_len,
        batch_size,
        num_workers=2,
        random_seed=42,
        pretrained_model_name_or_path="bert-base-uncased",
    ):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def _create_data_loader(self, df, line_column, context_column, label_column):

        ds = TwoModalDataset(
            lines=df[line_column].to_numpy(),
            contexts=df[context_column].to_numpy(),
            speakers=df[label_column].to_numpy(),
            max_seq_len=self.max_seq_len,
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
        )

        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def prepare_data(
        self, df, line_column, context_column, label_column, train_size, val_size
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
            train, line_column, context_column, label_column
        )
        val_data_loader = self._create_data_loader(
            val, line_column, context_column, label_column
        )
        test_data_loader = self._create_data_loader(
            test, line_column, context_column, label_column
        )
        return train_data_loader, train, val_data_loader, val, test_data_loader, test

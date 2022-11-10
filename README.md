

![Alt text](https://github.com/zuzadeu/twomodalbert/blob/develop/images/2022-11-09-19-01-23-image.png)

*Create a neural network with two text modes of tunable weights*

`pip install twomodalbert`

# Introduction

Let's consider whether we want to classify the below sample as positive or negative.

| context                                          | text                            |
| ------------------------------------------------ | ------------------------------- |
| This product is horrible, which I didn't expect. | Its website is just incredible! |

When only the message is considered, you can see its sentiment is positive due to the word 'incredible'. However, when the context is considered, sentiment will be rather negative.

The TwoModalBERT package allows us to quickly run an experiment with the two-modal neural network architecture described in this section. It allows quickly constructing a model on top of `PyTorch` and `transformers` libraries and enables experimenting with weights of two input texts. So, how to use the package?

# Usage

1. First, create a `config.ini` file in your working directory (parameters are described here). 
   
   ```ini
   [GENERAL]
    EPOCHS = 3
    RANDOM_SEED = 42
    BATCH_SIZE = 16
    MAX_SEQ_LEN = 200
    NUM_WORKERS = 2
    PRETRAINED_MODEL_NAME_OR_PATH = bert-base-uncased
    MODEL_SAVE_PATH = best_model_state.bin
   ```

   2. Read the file.

```python
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")
```

3. Set the device.
   
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"
   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

## Train

1. initialize  DataPreparation and Trainer modules.
   
   ```python
   from twomodalbert.DataPreparation import TwoModalDapaPreparation
   from twomodalbert.Trainer import TwoModalBertTrainer
   
   DataPreparation = TwoModalDataPreparation(config=config)
   Trainer = TwoModalBertTrainer(device=DEVICE, config=config)
   ```

2. Split input `df`  with *text*, *context*, *label* columns and create data loaders.
   
   ```python
   (
       train_data_loader,
       train,
       val_data_loader,
       val,
       test_data_loader,
       test,
   ) = DataPreparation.prepare_data(
       df,
       text_column="text",
       context_column="context",
       label_column="label",
       train_size=0.8,
       val_size=0.1,
   )
   ```

3. Train the model (nn parameters described in this section)
   
   ```
   model, history = Trainer.train_model(
       train_data_loader,
       train,
       val_data_loader,
       val,
       text_size=100,
       context_size=50,
       binary=False,
       text_p=0.3,
       context_p=0.3,
       output_p=0.3,
   
   )
   ```

## Predict

# Neural network architecture

![Alt text](https://github.com/zuzadeu/twomodalbert/blob/develop/images/2022-11-01-17-55-38-image.png)

# Parameters in `config.ini`

# Requirenments

# Contribution

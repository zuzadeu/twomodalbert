![Alt text](https://github.com/zuzadeu/twomodalbert/blob/develop/images/2022-11-09-19-01-23-image.png)

*Fine-tune a neural network with two text modes of tunable weights*

`pip install twomodalbert`

# Introduction

Let's consider whether we want to classify the below sample as positive or negative.

| context                                          | text                            |
| ------------------------------------------------ | ------------------------------- |
| This product is horrible, which I didn't expect. | Its website is just incredible! |

When only the message is considered, you can see its sentiment is positive due to the word 'incredible'. However, when the context is considered, sentiment will be rather negative.

The TwoModalBERT package allows us to quickly run an experiment with the two-modal neural network architecture described in [this section](https://github.com/zuzadeu/twomodalbert#neural-network-architecture). It allows quickly constructing a model on top of `PyTorch` and `transformers` libraries and enables experimenting with weights of two input texts. So, how to use the package?

# Usage

1. First, create a `config.ini` file in your working directory (parameters are described [here](https://github.com/zuzadeu/twomodalbert#parameters-in-configini)). 
   
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

3. Train the model (nn parameters described in [this section](https://github.com/zuzadeu/twomodalbert#neural-network-architecture))
   
   ```python
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

1. Load the model.
   
   ```python
   model = TwoModalBERTModel(
       text_size=100,
       context_size=50,
       binary=False,
       text_p=0.3,
       context_p=0.3,
       output_p=0.3,
   )
   
   model.load_state_dict(torch.load(config["GENERAL"]["MODEL_SAVE_PATH"]))
   ```

2. Evaluate it on a test set (choose any metric from `sklearn`).
   
   ```python
   y_pred, y_test = test_model(model, test_data_loader, DEVICE)
   ```

3. Run the model on two text inputs
   
   ```python
   text = "Its website is just incredible!"
   
   context = "This product is horrible, which I didn't expect."
   
   predict_on_text(model, text, context, DEVICE)
   ```

# Neural network architecture

Below you can find how TwoModalBERT is constructed and what are the class parameters.

![Alt text](https://github.com/zuzadeu/twomodalbert/blob/develop/images/2022-11-01-17-55-38-image.png)

First of all, on top of the last BERT layer, a linear layer is added. To be precise, it is added on top of the CLS token. As the CLS token aggregates the entire sequence representation, it is often used in a classification task.

The linear layer transforms input features with hidden size relevant to the BERT model, which is usually 768 for the models available in the *transformers* package, into features with hidden size equal to predefined `context_size` and `text_size`.

In the next step, the dropout layers with probabilities `context_p` and `text_p` are added on the top. Why? Because it makes the neural network less sensitive to the specific weights of neurons and not prone to overfitting.

Finally, both branches created similarly are combined and followed by another dropout layer of `output_p` and an activation function (Sigmoid if `binary`, else Softmax). 

# Parameters in `config.ini`

Settings to be defined in the  `config.ini` file:

| Variable                      | Description                                                                                        | Default Value     |
| ----------------------------- | -------------------------------------------------------------------------------------------------- | ----------------- |
| EPOCHS                        | the number of complete passes through the training dataset                                         | 3                 |
| RANDOM_SEED                   | a number used to initialize a pseudorandom number generator                                        | 42                |
| BARCH_SIZE                    | the number of training samples to work through before the model’s internal parameters are updated  | 16                |
| MAX_SEQ_LEN                   | the maximum length in number of tokens for the inputs to the transformer model                     | 200               |
| NUM_WORKERS                   | the number of processes that generate batches in parallel                                          | 2                 |
| PRETRAINED_MODEL_NAME_OR_PATH | of a pre-trained model configuration to load from cache or download (equivalent to `transformers`) | bert-base-uncased |
| MODEL_SAVE_PATH               | the model save path                                                                                | best_model.bin    |

# Requirenments

- `configparser-5.3.0`

- `scikit-learn-1.0.2`

- `torch-1.12.1+cu113`

- `transformers-4.24.0`

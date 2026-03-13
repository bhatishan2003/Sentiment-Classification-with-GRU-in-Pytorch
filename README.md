# Sentiment Classification with RNN in PyTorch

A PyTorch implementation of sentiment classification using Recurrent Neural Networks (RNN) and Bidirectional RNN (BiRNN) models. This project trains on the NLTK movie reviews dataset to classify text as positive or negative sentiment.

## Features

- **Two Model Architectures**: Simple RNN and Bidirectional RNN for sentiment analysis
- **Automatic Vocabulary Building**: Creates vocabulary from training data with configurable max size
- **Text Preprocessing**: Tokenization, stopword removal, and punctuation filtering
- **Training and Evaluation**: Complete training pipeline with validation
- **Prediction Mode**: Classify sentiment of custom text inputs
- **Checkpoint Saving**: Automatically saves best performing models
- **GPU Support**: Utilizes CUDA if available

## Installation

1. Clone or download this repository
    ```bash
    git clone https://github.com/bhatishan2003/Sentiment Classification with RNN in PyTorch.git
    cd Sentiment Classification with RNN in PyTorch
    ```
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
1. Run the installation script to download additional NLTK data:
    ```bash
    python install.py
    ```

## Usage

### Training a Model

Train an RNN model for 10 epochs:

```bash
python sentiment_classifier.py --model rnn --epochs 10
```

Train a BiRNN model with custom hyperparameters:

```bash
python sentiment_classifier.py --model birnn --epochs 20 --batch_size 64 --lr 0.001 --embed_dim 128 --hidden_dim 256
```

### Making Predictions

Predict sentiment on custom text using a trained model:

```bash
python sentiment_classifier.py --model rnn --predict --text "This movie was absolutely fantastic!"
```

Example output:

```
Text      : This movie was absolutely fantastic!
Tokens    : ['movie', 'absolutely', 'fantastic']
Sentiment : POSITIVE 👍
Confidence: 94.2%
```

## Development Notes

- Pre-commit

    We use pre-commit to automate linting of our codebase.
    - Install hooks:
        ```bash
        pre-commit install
        ```
    - Run Hooks manually (optional):
        ```bash
        pre-commit run --all-files
        ```

- Ruff:
    - Lint and format:
        ```bash
        ruff check --fix
        ruff format
        ```

# Sentiment Analysis with BERT and Optuna Optimization

This repository provides a complete workflow for training a BERT-based model on the Yelp review dataset for sentiment classification. It includes dataset preprocessing, model training, hyperparameter optimization using Optuna, and result visualization.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Evaluation and Visualization](#evaluation-and-visualization)

## Installation

First, install the required libraries:

```bash
pip install datasets evaluate optuna transformers torch scikit-learn matplotlib
```

## Data Preparation
The dataset used for this project is the Yelp Review Full dataset. This section includes code to load and preprocess the data:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
data_sentiment = load_dataset('yelp_review_full')

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = data_sentiment.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```
### Checking for Null Values
Ensure that there are no null or empty values in the dataset:

```python
def check_null_values(split):
    null_text_count = sum(1 for x in data_sentiment[split]['text'] if x is None or x.strip() == "")
    null_label_count = sum(1 for x in data_sentiment[split]['label'] if x is None)
    return null_text_count, null_label_count

# Check for null or empty values
train_null_texts, train_null_labels = check_null_values('train')
test_null_texts, test_null_labels = check_null_values('test')

print(f"Training split - Null or empty texts: {train_null_texts}, Null labels: {train_null_labels}")
print(f"Test split - Null or empty texts: {test_null_texts}, Null labels: {test_null_labels}")
```
## Model Training
The following code demonstrates how to train a BERT model on the preprocessed data:

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    loss = np.mean(eval_pred[0])
    return {
        'accuracy': accuracy,
        'loss': loss
    }

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

training_args = TrainingArguments(
    output_dir="best_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
```
## Hyperparameter Optimization
We use Optuna to optimize hyperparameters. The following code sets up the Optuna study and trains the model with different hyperparameters:

```python
import optuna

def objective(trial):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 3)
    
    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results['eval_accuracy']

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3)

print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)
```
## Evaluation and Visualization
Visualize the training and evaluation metrics using matplotlib:

```python
import matplotlib.pyplot as plt

def extract_logs(trainer):
    logs = trainer.state.log_history
    train_loss = [entry['loss'] for entry in logs if 'loss' in entry]
    eval_loss = [entry['eval_loss'] for entry in logs if 'eval_loss' in entry]
    eval_accuracy = [entry['eval_accuracy'] for entry in logs if 'eval_accuracy' in entry]
    
    train_steps = range(1, len(train_loss) + 1)
    epochs = range(1, len(eval_loss) + 1)
    
    return train_steps, train_loss, epochs, eval_loss, eval_accuracy

train_steps, train_loss, epochs, eval_loss, eval_accuracy = extract_logs(trainer)

def plot_training_loss(train_steps, train_loss):
    plt.figure(figsize=(6, 6))
    plt.plot(train_steps, train_loss, label='Training Loss', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss During Training')
    plt.legend()
    plt.show()

def plot_evaluation_metrics(epochs, eval_loss, eval_accuracy):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, eval_loss, label='Evaluation Loss', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss During Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_accuracy, label='Evaluation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy During Training')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_loss(train_steps, train_loss)
plot_evaluation_metrics(epochs, eval_loss, eval_accuracy)
```
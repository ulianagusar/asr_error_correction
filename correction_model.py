# !pip install evaluate
# !pip install jiwer
# !pip install --upgrade sympy

import evaluate
import re
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import pipeline
import torch
import numpy as np
from datasets import DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
import warnings
import logging
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


# dataset = load_dataset("Ultralordb0d/medical_asr_prediction_usm" )
# print(f"Dataset splits available: {dataset.keys()}")
dataset = load_dataset("Ultralordb0d/medical_asr_prediction_usm_augmented")
print(f"Dataset splits available: {dataset.keys()}")

def clean_text(text):

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # видаляє пунктуацію
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def preprocess(example):

    return {
       # "input_text": clean_text(example["prediction"])  ,
        "input_text": example["prediction_embedding_model_not_nltk_160_train_new_n_gram"]  ,
        "target_text": example["transcription_clean"]
        # "target_text": example["transcription_clean"]
    }





train_dataset = dataset["train"].map(preprocess)
train_test_split = train_dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

train_val_split = test_dataset.train_test_split(test_size=0.5, seed=42)
test_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]



print(f'Training examples: {train_dataset["target_text"][:5]}')
print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")
print(f"Test examples: {len(test_dataset)}")

model_name = "t5-small"

print("train_dataset")
print(len(train_dataset))
print("test_dataset")
print(len(test_dataset))
print("val_dataset")
print(len(val_dataset))

tokenizer = T5Tokenizer.from_pretrained(model_name)

model = T5ForConditionalGeneration.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

max_input_length = 128
max_target_length = 128

def tokenize_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]

    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

batch_size = 32
train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(tokenized_val, batch_size=batch_size)
test_dataloader = DataLoader(tokenized_test, batch_size=batch_size)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = torch.compile(model)
model.to(device)
optimizer = AdamW(model.parameters(), lr=0.001)

num_epochs = 6
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)



def train():
  #  progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []


    model.train()
    scaler = GradScaler()  

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_wer = 0.0
        num_batches = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with autocast():  
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            running_loss += loss.item()


        avg_train_loss = running_loss / len(train_dataloader)



        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")


        val_loss  = validate()
        val_losses.append(val_loss)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            model.save_pretrained("./medical_asr_correction_model_best")
            tokenizer.save_pretrained("./medical_asr_correction_model_best")


    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 10))

    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)

    plt.title("Training and Validation Loss", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)

    plt.legend(loc="upper right", fontsize=14)

    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Saving final model...")
    model.save_pretrained("./medical_asr_correction_model_final")
    tokenizer.save_pretrained("./medical_asr_correction_model_final")





def validate():
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss



def evaluate_model():

    model_path = "./medical_asr_correction_model_best"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    predictions = []
    references = []
    inputs = []


    with torch.no_grad():

        for batch in test_dataloader :
            batch = {k: v.to(device) for k, v in batch.items()}


            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True
            )


            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            input_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

            predictions.extend(pred_texts)
            references.extend(ref_texts)
            inputs.extend(input_texts)




    
    wer = wer_metric.compute(predictions=predictions, references=references)

    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = correct / len(predictions)

    print(f"Test Results :")
    print(f"  Exact Match Accuracy: {accuracy:.4f}")
    print(f"  Word Error Rate (WER): {wer:.4f}")

    print("\nSample predictions :")
    for i in range(min(5, len(predictions))):
        print(f"Input:      {inputs[i]}")
        print(f"Prediction: {predictions[i]}")
        print(f"Reference:  {references[i]}")
        print("-" * 50)

    return inputs ,  predictions ,references



wer_metric = evaluate.load("wer")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./medical_asr_correction_model_best"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)
model.eval()

def predict_single(text, max_length=128):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)


    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )


    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction
text_with_error = "judund( judant juvent cinodn) forte is a potent medicine for relieving pain and inflammation"


# text_with_error = clean_text('Consider incorporating by FITABACTERium by FITABACTERium by FITABACTERium by FITABACTERium by FITABACTERium by FITABACTERium by FITABACTERium by')
# text_with_error =text_with_error+" (fibrasium fibr disj)"
corrected = predict_single(text_with_error)
print(f"Оригінал: {text_with_error}")
print(f"Виправлено: {corrected}")
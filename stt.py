
import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import torch
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)



csv_path = '/content/United-Syn-Med/data/train.csv'
audio_base_dir = "/content/United-Syn-Med/train"
model_name = "openai/whisper-tiny.en"


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model=model_name, chunk_length_s=30, device=0 if device=="cuda" else -1)


df = pd.read_csv(csv_path)
df = df.drop_duplicates(subset=["transcription"])
df = df[175000:200000]
results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = os.path.join(audio_base_dir, row["file_name"])
    if not os.path.exists(audio_path):
        print(f"Пропущено: {audio_path}")
        continue

    prediction = pipe(audio_path)["text"]
    results.append({
        "filename": row["file_name"],
        "transcription": row["transcription"],
        "prediction": prediction
    })


dataset_repo = "Ultralordb0d/medical_asr_prediction_usm"
existing = load_dataset("Ultralordb0d/medical_asr_prediction_usm" )
new_test = Dataset.from_list(results)


combined_train = concatenate_datasets([existing["train"], new_test])

final_dataset = DatasetDict({
    "train": combined_train
})

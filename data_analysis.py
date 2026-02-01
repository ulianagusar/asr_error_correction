# !pip install huggingface_hub
# !huggingface-cli login
# !pip install -U datasets




from huggingface_hub import snapshot_download
import pandas as pd
from IPython.display import Audio
import tarfile
from transformers import pipeline
from datasets import load_dataset
from collections import Counter
from datasets import concatenate_datasets, DatasetDict
import librosa
import matplotlib.pyplot as plt
import tqdm


dataset = load_dataset("Ultralordb0d/medical_asr_prediction_usm_augmented")
print(f"Dataset splits available: {dataset.keys()}")

repo_id = "united-we-care/United-Syn-Med"


snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="./United-Syn-Med"
    )

with tarfile.open("/content/United-Syn-Med/data/audio/test.tar.gz", "r:gz") as tar:
        tar.extractall("/content/United-Syn-Med")
with tarfile.open("/content/United-Syn-Med/data/audio/train.tar.gz", "r:gz") as tar:
        tar.extractall("/content/United-Syn-Med")

train = pd.read_csv('/content/United-Syn-Med/data/train.csv')
test = pd.read_csv('/content/United-Syn-Med/data/test.csv')

len(train["file_name"])
len(test["file_name"])

len(train["file_name"].unique())
len(test["file_name"].unique())



def print_male_famale_plot(df):
    female_count = df['file_name'].str.contains('female', case=False, na=False).sum()
    male_count = df.shape[0] - female_count
    counts = pd.Series({'female': female_count, 'male': male_count})


    plt.figure()
    counts.plot(kind='bar')
    plt.xlabel('Стать')
    plt.ylabel('Кількість')
    plt.title('Кількість female та male')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

print_male_famale_plot(train)
print_male_famale_plot(test)



def audio_duration_dist(df ,ds_t):
        durations = []
        for filepath in tqdm.tqdm(df['file_name']):
            y, sr = librosa.load("/content/United-Syn-Med/"+ ds_t +"/"+filepath, sr=None)
            durations.append(len(y) / sr)


        df['duration_sec'] = durations

        print("total duration: "+ str(sum(df['duration_sec'])/60/60) + "hours")
        if ds_t == "train":
            print("total 200s duration: "+ sum(df['duration_sec'][:200_000])/60/60 + "hours")

        plt.figure()
        plt.hist(df['duration_sec'], bins=30)
        plt.xlabel('Тривалість аудіо (секунди)')
        plt.ylabel('Кількість файлів')
        plt.title('Розподіл тривалості аудіофайлів' + ds_t )
        plt.tight_layout()
        plt.show()

audio_duration_dist(train ,"train")
audio_duration_dist(test ,"test")

# Audio("/content/United-Syn-Med/test/drug-male-8b85e001-c0db-41b2-b932-2e3dd390352f.mp3")



def word_symbols_dist(df , var_tar):
        char_counts = df[var_tar].str.len()

        word_counts = df[var_tar].str.split().str.len()

        df['char_count'] = char_counts
        df['word_count'] = word_counts

        plt.figure()
        plt.hist(df['char_count'], bins=30)
        plt.xlabel('Кількість символів')
        plt.ylabel('Кількість рядків')
        plt.title('Розподіл кількості символів у '+ var_tar)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.hist(df['word_count'], bins=30)
        plt.xlabel('Кількість слів')
        plt.ylabel('Кількість рядків')
        plt.title('Розподіл кількості слів у транскрипціях'+ var_tar)
        plt.tight_layout()
        plt.show()

word_symbols_dist(train , "transcription")
word_symbols_dist(test, "transcription")





#після транскрибації
dataset_repo = "Ultralordb0d/medical_asr_prediction_usm"
dataset = load_dataset("Ultralordb0d/medical_asr_prediction_usm" )

train = pd.DataFrame(dataset["train"])
word_symbols_dist(train, "prediction")



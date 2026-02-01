# !pip install https://github.com/kpu/kenlm/archive/master.zip
# !python -m pip install cmake
# !git clone https://github.com/kpu/kenlm.git
# !python -m pip install cmake
# !git clone https://github.com/kpu/kenlm.git
# %cd kenlm
# !mkdir -p build && cd build && cmake .. && make -j4


import re
import load_dataset
dataset = load_dataset("Ultralordb0d/medical_asr_prediction_usm" )
print(f"Dataset splits available: {dataset.keys()}")


def clean_text(text):

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # видаляє пунктуацію
        text = re.sub(r'\s+', ' ', text)
        return text.strip()



def preprocess(example):

    return {
        "input_text": clean_text(example["prediction"]) ,
        "target_text": clean_text(example["transcription"])
    }



train_dataset = dataset["train"].map(preprocess)
train_test_split = train_dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

train_val_split = test_dataset.train_test_split(test_size=0.5, seed=42)
test_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]


sentences = train_dataset["target_text"]
with open("/content/corpus.txt", "w") as f:
    for sentence in sentences:
        f.write(sentence.strip() + "\n")

# !/content/kenlm/build/bin/lmplz -o 3 < /content/corpus.txt > /content/3gram.arpa
# !/content/kenlm/build/bin/build_binary /content/3gram.arpa /content/3gram.bin

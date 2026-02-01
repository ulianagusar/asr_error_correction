import kenlm
from data_base import find_nearest_words
import re
from datasets import load_dataset


def add_sim_panphon(sentence , model ):

        sentence_logprod = []
        for logprod , _ ,_ in model.full_scores(sentence, bos=True, eos=False) :
            sentence_logprod.append(logprod)
        words = sentence.split()


        final_sentence = ""
        combined_words = ""

        for i in range(len(sentence_logprod)):


            if sentence_logprod[i] <= -5 and combined_words != "":
                  combined_words += words[i]
                  final_sentence = final_sentence + " " + words[i]

            elif sentence_logprod[i] > -5 and combined_words != "":
                #  print(combined_words)
                if len(combined_words)>=3 and not combined_words.isdigit():
                    res = find_nearest_words(combined_words,3)
                    similar_words = '('
                    for word, _ in res:
                        similar_words = similar_words + " " + word
                    final_sentence = final_sentence + similar_words +")" + " " + words[i]
                else:
                    final_sentence = final_sentence + " " + words[i]

                combined_words = ""
            elif sentence_logprod[i] <= -5 and combined_words == "" :
                  combined_words += words[i]
                  final_sentence = final_sentence + " " + words[i]
            else:
                  final_sentence = final_sentence + " " + words[i]
            if  i == len(sentence_logprod) -1 and combined_words != "":
                  if len(combined_words)>=3 and not combined_words.isdigit():
                    # print(combined_words)
                      res = find_nearest_words(combined_words , 3)
                      similar_words = '('
                      for word, _ in res:
                          similar_words = similar_words + " " + word
                      final_sentence = final_sentence + similar_words +")"
                  # else:
                  #     final_sentence = final_sentence + " " + words[i]



        final_sentence = final_sentence[1:]

        return final_sentence


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def preprocess(example):

    example["prediction_embedding_model_not_nltk_160_train_new_n_gram"] = add_sim_panphon(clean_text(example["prediction"]), model )

    return example



model = kenlm.Model("3gram.bin")
sentence = "how are you helllooo "
print(add_sim_panphon(sentence , model ))


# dataset = load_dataset("Ultralordb0d/medical_asr_prediction_usm_augmented")
# print(f"Dataset splits available: {dataset.keys()}")

# dataset = dataset.map(preprocess)
# dataset.push_to_hub("Ultralordb0d/medical_asr_prediction_usm_augmented")
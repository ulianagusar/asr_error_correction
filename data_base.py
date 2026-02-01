# !pip install faiss-cpu

import faiss
import numpy as np
import torch
import pickle
import panphon
import epitran
import tqdm
from phonetic_embedding import PhoneticLSTMEncoder


ft = panphon.FeatureTable()
epi = epitran.Epitran('eng-Latn')


class PhoneticEmbeddingModel:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = PhoneticLSTMEncoder(input_dim=24, hidden_dim=128, output_dim=64, num_layers=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def encode_word(self, word, feature_extractor):
   
        features = feature_extractor(word)
        if not features:
            return None

        with torch.no_grad():
            features_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
            lengths = torch.tensor([len(features)])
            embedding = self.model(features_tensor, lengths)
            return embedding.cpu().numpy().flatten()


model_path = "best_phonetic_embedding_model-9.pth"
embedding_model = PhoneticEmbeddingModel(model_path)


def get_phonetic_features(word):
    try:
        word_ipa = epi.transliterate(word)
        vec = ft.word_to_vector_list(word_ipa, numeric=True)
        return vec
    except Exception:
        return []


word_vectors = []
word_labels = []

def add_word_to_db(word):
    embedding = embedding_model.encode_word(word, get_phonetic_features)
    if embedding is not None:
        word_vectors.append(embedding)
        word_labels.append(word)

# with open("words_not_nltk.txt", 'r', encoding='utf-8') as f:
#             words = [line.strip() for line in f]
# for word in tqdm(words):
#     add_word_to_db(word)


# word_vectors = np.array(word_vectors, dtype='float32')
# dimension = word_vectors.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(word_vectors)

# print(f"Додано {len(word_labels)} слів у векторну базу.")


def find_nearest_words(query_word, k=5):
    query_vec = embedding_model.encode_word(query_word, get_phonetic_features)
    if query_vec is None:
        print(f"Не вдалося отримати вбудування для '{query_word}'")
        return []

    D, I = index.search(np.array([query_vec], dtype='float32'), k)
    results = [(word_labels[i], float(D[0][j])) for j, i in enumerate(I[0])]
    return results

# збереження індексу та міток
# faiss.write_index(index, "phonetic_index.faiss")
# with open("word_labels.pkl", "wb") as f:
#     pickle.dump(word_labels, f)
    
#  вивантаження індексу та міток
index = faiss.read_index("phonetic_index_no_num-4.faiss")
with open("word_labels_no_num.pkl", "rb") as f:
    word_labels = pickle.load(f)
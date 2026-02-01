# !pip install git+https://github.com/dmort27/epitran.git
# !git clone http://github.com/festvox/flite
# %cd flite
# ! ./configure && make
# ! sudo make install
# %cd testsuite
# ! make lex_lookup
# ! sudo cp lex_lookup /usr/local/bin
# %cd ..
# %cd ..
import epitran
import panphon
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from torch.utils.data import Dataset, DataLoader
import panphon
import epitran
from tqdm import tqdm

import warnings
import logging


warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
ft = panphon.FeatureTable()
epi = epitran.Epitran('eng-Latn')

def get_phonetic_features(word):

    try:
        word_ipa = epi.transliterate(word)
        vec = ft.word_to_vector_list(word_ipa, numeric=True)
        return vec
    except Exception as e:
        print(f"Помилка отримання ознак для '{word}': {e}")
        return []

def substitution_cost(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    feature_count = len(vec1)
    return np.sum(np.abs(vec1 - vec2)) / feature_count if feature_count > 0 else 1.0
import torch


def articulatory_distance(source_features, target_features):

    m, n = len(source_features), len(target_features)

    if m == 0 and n == 0:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    A = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        A[i, 0] = i  # вартість видалення
    for j in range(n + 1):
        A[0, j] = j  # вартість вставки

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = substitution_cost(source_features[i-1], target_features[j-1])
            A[i, j] = min(
                A[i-1, j] + 1,      
                A[i, j-1] + 1,      
                A[i-1, j-1] + sub_cost  
            )

    return A[m, n]


class PhoneticLSTMEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super(PhoneticLSTMEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )


        self.projection = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x, lengths):


        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed_x)


        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        output = self.projection(hidden)

        return output


class ArticulatoryMetricLoss(nn.Module):

    def __init__(self):
        super(ArticulatoryMetricLoss, self).__init__()

    def forward(self, embeddings, words, articulatory_distance_fn):

        batch_size = embeddings.size(0)
        loss = 0.0
        pairs_count = 0

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:

                    emb_distance = torch.norm(embeddings[i] - embeddings[j], p=2) ** 2


                    art_distance = articulatory_distance_fn(words[i], words[j])


                    pair_loss = (emb_distance - art_distance) ** 2
                    loss += pair_loss
                    pairs_count += 1


        if pairs_count > 0:
            loss = loss / pairs_count

        return loss


class WordDataset(Dataset):
    def __init__(self, words, feature_extractor):
        self.words = words
        self.feature_extractor = feature_extractor
        self.features_cache = {}  

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]


        if word in self.features_cache:
            features = self.features_cache[word]
        else:
            features = self.feature_extractor(word)
            self.features_cache[word] = features

        return {
            'word': word,
            'features': features,
            'length': len(features)
        }


def collate_fn(batch):

    words = [item['word'] for item in batch]
    features = [item['features'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])

    max_len = max(lengths).item()

    padded_features = []
    for feat in features:
        feat_array = np.array(feat)

        if len(feat_array) == 0:
            feat_dim = 24  
            padded = np.zeros((max_len, feat_dim))
        else:
            feat_dim = feat_array.shape[1] if len(feat_array.shape) > 1 else feat_array.shape[0]
            padded = np.zeros((max_len, feat_dim))
            padded[:len(feat)] = feat_array

        padded_features.append(padded)

    padded_features = torch.FloatTensor(np.array(padded_features))

    return {
        'words': words,
        'features': padded_features,
        'lengths': lengths
    }


class PhoneticEmbeddingModel:
    def __init__(self, input_dim=24, hidden_dim=256, output_dim=100,
                 num_layers=2, dropout=0.2, learning_rate=0.001):
        self.ft = panphon.FeatureTable()
        self.epi = epitran.Epitran('eng-Latn')  

        self.model = PhoneticLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.criterion = ArticulatoryMetricLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


        self.best_loss = float('inf')
        self.best_correlation = float('-inf')
        self.best_model_path = "best_phonetic_embedding_model.pth"

    def get_phonetic_features(self, word):

        try:
            word_ipa = self.epi.transliterate(word)
            vec = self.ft.word_to_vector_list(word_ipa, numeric=True)
            return vec
        except Exception as e:
            print(f"Error getting features for '{word}': {e}")
            return []

    def compute_articulatory_distance(self, word1, word2):

        features1 = self.get_phonetic_features(word1)
        features2 = self.get_phonetic_features(word2)
        return articulatory_distance(features1, features2)

    def train(self, word_list, batch_size=32, num_epochs=10, validation_pairs=None, validate_every=1):


        dataset = WordDataset(word_list, self.get_phonetic_features)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        print(f"Training on {len(word_list)} words for {num_epochs} epochs")
        print(f"Using device: {self.device}")


        training_stats = {
            'epoch_losses': [],
            'validation_losses': [],
            'learning_rates': [],
            'validation_metrics': []
        }

        total_batches = len(dataloader)
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_loss = 0


            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}",
                                unit="batch", ncols=100, disable=True)

            for batch_idx, batch in enumerate(progress_bar):

                words = batch['words']
                features = batch['features'].to(self.device)
                lengths = batch['lengths']


                self.optimizer.zero_grad()
                embeddings = self.model(features, lengths)


                loss = self.criterion(embeddings, words, self.compute_articulatory_distance)


                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss


            avg_loss = total_loss / total_batches
            epoch_time = time.time() - epoch_start_time
            training_stats['epoch_losses'].append(avg_loss)


            current_lr = self.optimizer.param_groups[0]['lr']
            training_stats['learning_rates'].append(current_lr)

            print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - Avg loss: {avg_loss:.4f}")


            
            val_metrics = self.validate(validation_pairs, verbose=False)
            val_loss = val_metrics.get('loss', float('inf'))
            print("val_loss" + str(val_loss))
            training_stats['validation_losses'].append(val_loss)
            training_stats['validation_metrics'].append(val_metrics)


            correlation = val_metrics.get('correlation', float('-inf'))


            if correlation > self.best_correlation:
                    self.best_loss = min(val_loss, self.best_loss)
                    self.best_correlation = max(correlation, self.best_correlation)
                    self.save_model(self.best_model_path)
                    print(f"Best model saved - Loss: {self.best_loss:.4f}, Correlation: {self.best_correlation:.4f}")


        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Final loss: {training_stats['epoch_losses'][-1]:.4f}")
        print(f"Best model saved with Loss: {self.best_loss:.4f}, Correlation: {self.best_correlation:.4f}")

        return training_stats


    def validate(self, pairs, verbose=True):

            validate_epoch_start_time = time.time()

            self.model.eval()

            if verbose:
                print("\nValidation:")


            emb_distances = []
            art_distances = []
            

            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():

                batch_size = 32 
                
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i + batch_size]
                    

                    words_batch = []
                    features_batch = []
                    lengths_batch = []
                    
                    for w1, w2 in batch_pairs:
                        words_batch.extend([w1, w2])
                        

                        f1 = self.get_phonetic_features(w1)
                        f2 = self.get_phonetic_features(w2)
                        
                        features_batch.extend([f1, f2])
                        lengths_batch.extend([len(f1), len(f2)])
                    

                    lengths = torch.tensor(lengths_batch)
                    max_len = max(lengths).item() if len(lengths) > 0 else 1
                    
                    padded_features = []
                    for feat in features_batch:
                        feat_array = np.array(feat)
                        
                        if len(feat_array) == 0:
                            feat_dim = 24
                            padded = np.zeros((max_len, feat_dim))
                        else:
                            feat_dim = feat_array.shape[1] if len(feat_array.shape) > 1 else feat_array.shape[0]
                            padded = np.zeros((max_len, feat_dim))
                            padded[:len(feat)] = feat_array
                        
                        padded_features.append(padded)
                    
                    if len(padded_features) == 0:
                        continue
                        
                    features_tensor = torch.FloatTensor(np.array(padded_features)).to(self.device)
                    

                    embeddings = self.model(features_tensor, lengths.to(self.device))
                    

                    batch_loss = self.criterion(embeddings, words_batch, self.compute_articulatory_distance)
                    total_loss += batch_loss.item()
                    num_batches += 1

                
                for w1, w2 in pairs:
                    e1 = self.encode_word(w1)
                    e2 = self.encode_word(w2)
                    
                    emb_dist = torch.norm(e1 - e2, p=2).item()
                    art_dist = self.compute_articulatory_distance(w1, w2)
                    
                    emb_distances.append(emb_dist)
                    art_distances.append(art_dist)

 
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            

            correlation = float('nan')
            if len(pairs) > 1:
                try:
                    correlation = np.corrcoef(emb_distances, art_distances)[0, 1]
                    if verbose:
                        print(f"Correlation: {correlation:.4f}")
                except:
                    if verbose:
                        print("Unable to calculate correlation")

            if verbose:
                print(f"Validation loss (ArticulatoryMetricLoss): {avg_loss:.4f}")


            validate_epoch_time = time.time() - validate_epoch_start_time
            print("validate_epoch_time " + str(validate_epoch_time))
            return {
                'loss': avg_loss,  
                'correlation': correlation,
                'emb_distances': emb_distances,
                'art_distances': art_distances
            }
    def encode_word(self, word):
 
        self.model.eval()
        with torch.no_grad():
            features = self.get_phonetic_features(word)
            if not features:
                return torch.zeros(self.model.projection.out_features).to(self.device)

            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            length = torch.tensor([len(features)])

            embedding = self.model(features_tensor, length)
            return embedding.squeeze(0)

    def save_model(self, path):

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    with open("words_not_nltk.txt", 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
            #words = words[:10000]

    model = PhoneticEmbeddingModel(
        input_dim=24,  
        hidden_dim=128,
        output_dim=64,
        num_layers=2
    )

    train_ratio = 0.8
    train_size = int(len(words) * train_ratio)
    random.shuffle(words)
    train_words = words[:train_size]
    val_words = words[train_size:]

    print(f"Training on {len(train_words)} words, validating on {len(val_words)} words")

    training_stats = model.train(
        train_words,
        batch_size=8,
        num_epochs=6,
        validation_words=val_words 
    )


    model.save_model("phonetic_embedding_model.pth")


    try:

        

        plt.figure(figsize=(12, 5))
        

        plt.subplot(1, 2, 1)
        plt.plot(training_stats['epoch_losses'], label='Training Loss', color='blue')
        
        if 'validation_losses' in training_stats and training_stats['validation_losses']:
            plt.plot(training_stats['validation_losses'], label='Validation Loss', color='orange')
        
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        

        plt.subplot(1, 2, 2)
        if 'validation_metrics' in training_stats and training_stats['validation_metrics']:
            correlations = [m['correlation'] for m in training_stats['validation_metrics'] 
                          if not np.isnan(m['correlation'])]
            if correlations:
                plt.plot(correlations, label='Validation Correlation', color='green', marker='o')
                plt.title('Validation Correlation')
                plt.xlabel('Epoch')
                plt.ylabel('Correlation')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid correlation data', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Validation Correlation (No Data)')
        else:
            plt.text(0.5, 0.5, 'No correlation data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Validation Correlation (No Data)')
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        print("Training progress plot saved to training_progress.png")
        
    except Exception as e:
        print(f"Unable to generate training progress plot: {e}")

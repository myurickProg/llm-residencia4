from data_clean import data_cleaned
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Passo 3: Dividir os dados em conjuntos de treinamento e validação
train_data = data_cleaned.sample(frac=0.1, random_state=42)
val_data = data_cleaned.drop(train_data.index)

train_texts = train_data['summary'].tolist()
train_labels = train_data['cvss'].tolist()  # Rótulo como CVSS
val_texts = val_data['summary'].tolist()
val_labels = val_data['cvss'].tolist()

# Passo 4: Tokenização
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)


# Passo 5: Criar Dataset para PyTorch
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # CVSS como rótulo
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CodeDataset(train_encodings, train_labels)
val_dataset = CodeDataset(val_encodings, val_labels)

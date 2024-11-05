from train_validation import train_dataset, val_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Passo 6: Criar e compilar o modelo
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=1)

# Passo 7: Configurar parâmetros de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Passo 8: Criar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Passo 9: Treinar o modelo
trainer.train()

# Passo 10: Avaliar o modelo
trainer.evaluate()
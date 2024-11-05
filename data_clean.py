from dataset import path
import os
import pandas as pd

# Caminho do dataset baixado
dataset_path = path

file_path = os.path.join(dataset_path, "cve.csv")
data = pd.read_csv(file_path)

# Exibir as primeiras linhas do dataset
print(data.head())
# Listar arquivos no diretório
print(os.listdir(dataset_path))
# Ver informações sobre o DataFrame
print(data.info())

# Verificar as colunas disponíveis
print(data.columns)

# Remover linhas com valores ausentes na coluna 'summary' e 'cvss'
data_cleaned = data.dropna(subset=['summary', 'cvss'])
import os
import pandas as pd
import re
import torch
from transformers import BertModel, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np

# Charger le modèle BioBERT pré-entraîné
model_bio = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
tokenizer_bio = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Charger le modèle T5 pré-entraîné
model_t5 = T5ForConditionalGeneration.from_pretrained("racha009/t5-small-checkpoint-finetuned-pav1", token="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")
tokenizer_t5 = T5Tokenizer.from_pretrained("racha009/t5-small-checkpoint-finetuned-pav1", token="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")

def format_function(data):
    if isinstance(data, str):
        # Remplacer les citations par la balise <cit>
        data = re.sub(r'\[\d+[,0-9/-]*\]', ' <cit> ', data)
        data = re.sub(r'\[\d+[" ,"/0-9/-]*\]', ' <cit> ', data)

        # Supprimer le texte entre parenthèses et crochets
        data = re.sub(r'\([^)]+\)', '', data)
        data = re.sub(r'\[.*?\]', '', data)

        # Remplacer les chiffres par la balise <dig> uniquement pour les chaînes de caractères
        data = re.sub(r'\b\d+\b', ' <dig> ', data)

        # Supprimer les tables et les figures
        data = re.sub(r'\ntable \d+.*?\n', '', data, flags=re.IGNORECASE)
        data = re.sub(r'.*\t.*?\n', '', data)
        data = re.sub(r'\nfigure \d+.*?\n', '', data, flags=re.IGNORECASE)
        data = re.sub(r'[(]figure \d+.*?[)]', '', data, flags=re.IGNORECASE)
        data = re.sub(r'[(]fig. \d+.*?[)]', '', data, flags=re.IGNORECASE)
        data = re.sub(r'[(]fig \d+.*?[)]', '', data, flags=re.IGNORECASE) 
    return data

# Lire le DataFrame à partir d'un fichier CSV
file_path = '/kaggle/input/pfe-1-1/output1.csv'
data = pd.read_csv(file_path)

# Appliquer la fonction de formatage à chaque ligne de texte
data['texte_formate'] = data['cleaned_text1'].apply(format_function)

# Fonction pour générer des représentations vectorielles avec BioBERT
def generate_embeddings_bio(text):
    inputs = tokenizer_bio(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_bio(**inputs)
    hidden_states = outputs.last_hidden_state
    embeddings = hidden_states.mean(dim=1)
    return embeddings

# Appliquer la fonction de génération de représentations vectorielles à chaque texte prétraité
data['embeddings'] = data['texte_formate'].apply(generate_embeddings_bio)

# Utiliser les embeddings pour calculer les poids de chaque phrase
data['weights'] = data['embeddings'].apply(lambda x: x.sum().item())

# Définir la capacité du sac à dos (nombre maximal de phrases dans le résumé)
capacity = 5

# Sélectionner les phrases pour le résumé en utilisant l'algorithme du sac à dos
summary_indices = np.argsort(data['weights'].values)[-capacity:]

# Utiliser T5 pour générer le résumé abstrait
all_summaries2 = []
for idx in summary_indices:
    source_text = data.iloc[idx]['texte_formate']
    preprocessed_text = "summarize: " + source_text
    inputs = tokenizer_t5(preprocessed_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model_t5.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
    summary_text = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
    all_summaries2.append(summary_text)

# Enregistrer les résumés dans un fichier CSV
data_summary = data.iloc[summary_indices]
data_summary['abstractive_summary'] = all_summaries2

# Créer le répertoire summarized_data s'il n'existe pas
output_dir = '/kaggle/working/summarized_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file_path = '/kaggle/working/summarized_data/all_summaries2.csv'
data_summary.to_csv(output_file_path, index=False)

# Charger les résumés de référence
reference_summaries = data['shorter_abstract'].tolist()

# Calculer les scores BLEU et ROUGE
rouge = Rouge()
bleu_scores = []

# Calculer les scores BLEU et ROUGE pour tous les résumés
for reference_summary, generated_summary in zip(reference_summaries, all_summaries2):
    # Convertir les résumés en listes de phrases
    reference_summary = [reference_summary.split()]
    generated_summary = [generated_summary.split()]
    
    # Calculer le score BLEU
    bleu_score = corpus_bleu([reference_summary], generated_summary, smoothing_function=SmoothingFunction().method1)
    bleu_scores.append(bleu_score)
    
# Calculer le score ROUGE
rouge_scores = rouge.get_scores(all_summaries2, reference_summaries, avg=True)

# Afficher les scores BLEU et ROUGE
print("\nScore BLEU moyen:", sum(bleu_scores) / len(bleu_scores))
print("Score ROUGE moyen:", rouge_scores)

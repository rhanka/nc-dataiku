# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import io
import pandas as pd
import re
import os
from math import ceil
from concurrent.futures import ThreadPoolExecutor
import time

# Définir les dossiers d'entrée et de sortie
pdf_folder = dataiku.Folder("W8lS5GmB")  # Dossier contenant les PDF originaux
md_folder = dataiku.Folder("d7DdDueY")   # Dossier contenant les annotations générées

# Lister les fichiers PDF
pdf_files = [f for f in pdf_folder.list_paths_in_partition() if f.lower().endswith(".pdf")]
pdf_files.sort()

# Taille des lots
BATCH_SIZE = 100
# Nombre maximum d'images
MAX_IMAGES = 9

# Mettre en cache la liste des fichiers du dossier md pour éviter de la récupérer à chaque itération
all_md_files = set(md_folder.list_paths_in_partition())

# Fonction pour lire le contenu d'un fichier s'il existe
def read_file_content(folder, file_path):
    if file_path in all_md_files:
        with folder.get_download_stream(file_path) as stream:
            return io.BytesIO(stream.read()).read().decode("utf-8")
    return None

# Fonction pour traiter un PDF et créer une ligne de données
def process_pdf(pdf_file):
    base_name = os.path.splitext(pdf_file)[0]
    doc_root = base_name.split('_page_')[0] if '_page_' in base_name else base_name
    
    # Chemins des fichiers d'annotation
    json_file = base_name + ".json"
    md_file = base_name + ".md"
    md_img_file = base_name + "__with_img_desc.md"
    json_img_file = base_name + "__with_img_desc.json"
    
    # Lire le contenu MD et MD_IMG
    md_content = read_file_content(md_folder, md_file)
    md_img_content = read_file_content(md_folder, md_img_file)
    
    # Si md_img est vide, utiliser md à la place
    if not md_img_content:
        md_img_content = md_content
    
    # Initialiser les données de la ligne
    row_data = {
        "doc": pdf_file,
        "doc_root": doc_root + ".pdf",
        "json": json_file,
        "md": md_content,
        "md_img": md_img_content,
        "json_img": json_img_file
    }
    
    # Initialiser toutes les colonnes d'images à None
    for i in range(MAX_IMAGES):
        row_data[f"img-{i}"] = None
        row_data[f"img-{i}-desc"] = None
    
    # Optimisation: Utiliser un pattern précompilé et filtrer d'abord les fichiers pertinents
    img_pattern = re.compile(rf"^{re.escape(base_name)}-img-(\d+)\.jpeg$")
    
    # Chercher seulement les fichiers qui pourraient être des images ou des descriptions pour ce PDF
    base_prefix = f"{base_name}-img-"
    relevant_files = [f for f in all_md_files if f.startswith(base_prefix)]
    
    # Traiter les fichiers pertinents
    for file_path in relevant_files:
        img_match = img_pattern.match(file_path)
        if img_match:
            img_num = int(img_match.group(1))
            if img_num >= MAX_IMAGES:
                continue
            
            img_key = f"img-{img_num}"
            desc_file = f"{base_name}-img-{img_num}.md"
            
            row_data[img_key] = file_path
            if desc_file in all_md_files:
                row_data[f"{img_key}-desc"] = read_file_content(md_folder, desc_file)
    
    return row_data

# Accéder au dataset de sortie
output_dataset = dataiku.Dataset("a220_tech_docs_content")

# Diviser les fichiers en lots
total_batches = ceil(len(pdf_files) / BATCH_SIZE)
print(f"Traitement de {len(pdf_files)} documents en {total_batches} lots de {BATCH_SIZE} documents.")

# Définir toutes les colonnes possibles à l'avance
base_columns = ["doc", "doc_root", "json", "md", "md_img", "json_img"]
image_columns = [f"img-{i}" for i in range(MAX_IMAGES)] + [f"img-{i}-desc" for i in range(MAX_IMAGES)]
all_columns = base_columns + image_columns

# Traiter le premier lot pour définir le schéma
start_idx = 0
end_idx = min(BATCH_SIZE, len(pdf_files))
first_batch = pdf_files[start_idx:end_idx]
print(f"Traitement du premier lot : documents {start_idx + 1} à {end_idx}...")

start_time = time.time()

# Utiliser le multithreading pour le premier lot
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_pdf, first_batch))

# Créer le DataFrame à partir des résultats
df_first_batch = pd.DataFrame(results)

# Réorganiser les colonnes pour correspondre à all_columns
df_first_batch = df_first_batch[all_columns]

# Définir le schéma à partir du premier lot
output_dataset.write_schema_from_dataframe(df_first_batch)

# Ouvrir le writer une seule fois pour tous les lots
with output_dataset.get_writer() as writer:
    # Écrire le premier lot
    writer.write_dataframe(df_first_batch)
    first_batch_time = time.time() - start_time
    print(f"Lot 1 traité : {len(df_first_batch)} documents ajoutés au dataset en {first_batch_time:.2f} secondes.")
    
    # Traiter les lots restants
    for batch_num in range(1, total_batches):
        batch_start_time = time.time()
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(pdf_files))
        current_batch = pdf_files[start_idx:end_idx]
        print(f"Traitement du lot {batch_num + 1}/{total_batches}, documents {start_idx + 1} à {end_idx}...")
        
        # Utiliser le multithreading pour traiter le lot courant
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_pdf, current_batch))
        
        # Créer le DataFrame à partir des résultats
        df_batch = pd.DataFrame(results)
        
        # Réorganiser les colonnes pour correspondre à all_columns
        df_batch = df_batch[all_columns]
        
        # Écrire le lot courant
        writer.write_dataframe(df_batch)
        batch_time = time.time() - batch_start_time
        print(f"Lot {batch_num + 1} traité : {len(df_batch)} documents ajoutés au dataset en {batch_time:.2f} secondes.")
        
        # Estimer le temps restant
        avg_time_per_batch = (time.time() - start_time) / (batch_num + 1)
        remaining_batches = total_batches - (batch_num + 1)
        estimated_time_remaining = avg_time_per_batch * remaining_batches
        hours, remainder = divmod(estimated_time_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Temps restant estimé : {int(hours)}h {int(minutes)}m {int(seconds)}s")

total_time = time.time() - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Extraction terminée avec succès. Total : {len(pdf_files)} documents traités en {total_batches} lots.")
print(f"Temps total d'exécution : {int(hours)}h {int(minutes)}m {int(seconds)}s")
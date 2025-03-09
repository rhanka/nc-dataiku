# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import io
import pandas as pd
import re
import os
import json
from math import ceil

# Définir les dossiers d'entrée et de sortie
pdf_folder = dataiku.Folder("W8lS5GmB")  # Dossier contenant les PDF originaux
md_folder = dataiku.Folder("d7DdDueY")   # Dossier contenant les annotations générées

# Lister les fichiers PDF
pdf_files = [f for f in pdf_folder.list_paths_in_partition() if f.lower().endswith(".pdf")]
pdf_files.sort()

# Taille des lots
BATCH_SIZE = 10
# Nombre maximum d'images
MAX_IMAGES = 9

# Fonction pour lire le contenu d'un fichier s'il existe
def read_file_content(folder, file_path):
    if file_path in folder.list_paths_in_partition():
        with folder.get_download_stream(file_path) as stream:
            return io.BytesIO(stream.read()).read().decode("utf-8")
    return None

# Accéder au dataset de sortie
output_dataset = dataiku.Dataset("a220_tech_docs_content")

# Diviser les fichiers en lots
total_batches = ceil(len(pdf_files) / BATCH_SIZE)
print(f"Traitement de {len(pdf_files)} documents en {total_batches} lots de {BATCH_SIZE} documents.")

# Définir toutes les colonnes possibles à l'avance
base_columns = ["doc", "doc_root", "json", "md", "md_img", "json_img"]
image_columns = [f"img-{i}" for i in range(MAX_IMAGES)] + [f"img-{i}-desc" for i in range(MAX_IMAGES)]
all_columns = base_columns + image_columns

for batch_num in range(total_batches):
    start_idx = batch_num * BATCH_SIZE
    end_idx = min((batch_num + 1) * BATCH_SIZE, len(pdf_files))
    current_batch = pdf_files[start_idx:end_idx]

    print(f"Traitement du lot {batch_num + 1}/{total_batches}, documents {start_idx + 1} à {end_idx}...")

    # Initialiser le dictionnaire pour stocker les données du lot
    data = {col: [] for col in all_columns}

    # Pour chaque PDF dans le lot, extraire toutes les annotations associées
    for pdf_file in current_batch:
        base_name = os.path.splitext(pdf_file)[0]
        doc_root = base_name.split('_page_')[0] if '_page_' in base_name else base_name

        # Chemins des fichiers d'annotation
        json_file = base_name + ".json"
        md_file = base_name + ".md"
        md_img_file = base_name + "__with_img_desc.md"
        json_img_file = base_name + "__with_img_desc.json"

        # Lire le contenu des fichiers d'annotation
        row_data = {
            "doc": pdf_file,
            "doc_root": doc_root + ".pdf",
            "json": read_file_content(md_folder, json_file),
            "md": read_file_content(md_folder, md_file),
            "md_img": read_file_content(md_folder, md_img_file),
            "json_img": read_file_content(md_folder, json_img_file)
        }

        # Initialiser toutes les colonnes d'images à None
        for col in image_columns:
            row_data[col] = None

        # Identifier les images associées au document
        img_pattern = re.compile(rf"^{re.escape(base_name)}-img-(\d+)\.jpeg$")
        all_files = md_folder.list_paths_in_partition()

        # Chercher toutes les images et leurs descriptions
        for file_path in all_files:
            img_match = img_pattern.match(file_path)
            if img_match:
                img_num = int(img_match.group(1))
                if img_num >= MAX_IMAGES:
                    continue

                img_key = f"img-{img_num}"
                desc_file = f"{base_name}-img-{img_num}.md"
                row_data[img_key] = file_path

                if desc_file in all_files:
                    row_data[f"{img_key}-desc"] = read_file_content(md_folder, desc_file)

        # Ajouter les données de cette ligne
        for col in all_columns:
            data[col].append(row_data.get(col, None))

    # Créer le DataFrame
    df_batch = pd.DataFrame(data)

    # For the first batch, set the schema and write data
    if batch_num == 0:
        output_dataset.write_schema_from_dataframe(df_batch)
        with output_dataset.get_writer() as writer:
            writer.write_dataframe(df_batch)
    else:
        # For subsequent batches, append data without modifying the schema
        with output_dataset.get_writer() as writer:
            writer.write_dataframe(df_batch)

    print(f"Lot {batch_num + 1} traité : {len(df_batch)} documents ajoutés au dataset.")

print(f"Extraction terminée avec succès. Total : {len(pdf_files)} documents traités en {total_batches} lots vers le dataset 'a220_tech_docs_content'.")
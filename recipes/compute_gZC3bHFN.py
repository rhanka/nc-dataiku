# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NC_types_random_500_md_concat = dataiku.Dataset("NC_types_random_500_md_concat")
NC_types_random_500_md_concat_df = NC_types_random_500_md_concat.get_dataframe()


# Write recipe outputs
NC_types_500_md_files = dataiku.Folder("gZC3bHFN")
NC_types_500_md_files_info = NC_types_500_md_files.get_info()


# Parcourir chaque ligne du DataFrame et écrire les fichiers Markdown
for _, row in NC_types_random_500_md_concat_df.iterrows():
    doc_id = row["doc"]  # Identifiant pour le fichier
    markdown_content = row["chunk_concat"]  # Contenu Markdown
    
    # Définir le nom du fichier
    file_name = f"{doc_id}.md"
    
    # Écrire dans le dossier de sortie
    with NC_types_500_md_files.get_writer(file_name) as writer:
        writer.write(markdown_content.encode("utf-8"))
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import os
import markitdown

# Folders
A220_tech_docs = dataiku.Folder("SoQWOnhR")          # Input folder
A220_tech_docs_prep = dataiku.Folder("AXB1Cyno")    # Output folder

# Lister les fichiers PDF
pdf_files = [f for f in A220_tech_docs.list_paths_in_partition() if f.lower().endswith(".pdf")]

for pdf_file in pdf_files:
    # Lire le contenu PDF
    with A220_tech_docs.get_download_stream(pdf_file) as f:
        pdf_data = f.read()

    # Convertir en Markdown
    md_content = markitdown.convert(pdf_data)

    # Écrire le fichier .md
    md_file_name = os.path.splitext(pdf_file)[0] + ".md"
    with A220_tech_docs_prep.get_writer(md_file_name) as writer:
        writer.write(md_content)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import importlib.metadata

for dist in importlib.metadata.distributions():
    if dist.metadata['Name'] == "markitdown":
        print(f"{dist.metadata['Name']}=={dist.version}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(dir(markitdown))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import markitdown
print(dir(markitdown))
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import os
from markitdown import MarkItDown
import tempfile

md = MarkItDown()

# Folders
A220_tech_docs = dataiku.Folder("W8lS5GmB")          # Input folder
A220_tech_docs_prep = dataiku.Folder("AXB1Cyno")    # Output folder

# Lister les fichiers PDF
pdf_files = [f for f in A220_tech_docs.list_paths_in_partition() if f.lower().endswith(".pdf")]

for pdf_file in pdf_files:
    # Lire le contenu PDF
    with A220_tech_docs.get_download_stream(pdf_file) as f:
        pdf_data = f.read()
        
    # Utiliser un fichier temporaire pour la conversion
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_pdf.flush()  # Assurez-vous que le contenu est écrit sur le disque

        # Convertir en Markdown
        md_content = md.convert(temp_pdf.name)

        # Afficher le nombre de lignes dans le contenu Markdown
        num_lines = len(md_content.text_content.splitlines())
        print(f"Nombre de lignes : {num_lines}")
        
        # Écrire le fichier .md
        md_file_name = os.path.splitext(pdf_file)[0] + ".md"
        with A220_tech_docs_prep.get_writer(md_file_name) as writer:
            writer.write(md_content.text_content.encode('utf-8'))
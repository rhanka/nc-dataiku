# -*- coding: utf-8 -*-
import dataiku
import os
import tempfile
from pypdf import PdfReader, PdfWriter

# Folders
A220_tech_docs = dataiku.Folder("SoQWOnhR")          # Input folder
A220_tech_docs_pages = dataiku.Folder("W8lS5GmB")    # Output folder

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

        # Lire le PDF
        reader = PdfReader(temp_pdf.name)

        # Extraire chaque page et sauvegarder en tant que fichier PDF séparé
        for page_number, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)

            # Supprimer les métadonnées manuellement
            writer.metadata = {}  # Efface toutes les métadonnées pour réduire la taille

            # Formatage du numéro de page avec padding (0001, 0002, ...)
            padded_page_number = str(page_number + 1).zfill(4)  # Ajout de padding à 4 chiffres

            # Définir le nom de fichier pour chaque page
            page_pdf_file_name = f"{os.path.splitext(os.path.basename(pdf_file))[0]}_page_{padded_page_number}.pdf"

            # Créer un fichier temporaire pour chaque page PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as page_pdf:
                writer.write(page_pdf)
                page_pdf.flush()

                # Lire le contenu du fichier temporaire
                with open(page_pdf.name, 'rb') as page_pdf_file:
                    page_pdf_data = page_pdf_file.read()  # Lire les données en bytes
                    A220_tech_docs_pages.upload_data(page_pdf_file_name, page_pdf_data)

            # Supprimer le fichier temporaire
            os.remove(page_pdf.name)

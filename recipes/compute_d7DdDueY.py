# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import os
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import tempfile
import json

client = dataiku.api_client()
project = client.get_default_project()
auth_info = client.get_auth_info(with_secrets=True)
MISTRAL_API_KEY = None
for secret in auth_info["secrets"]:
    if secret["key"] == "MISTRAL_API_KEY":
        MISTRAL_API_KEY = secret["value"]

client = Mistral(api_key=MISTRAL_API_KEY)

# Folders
A220_tech_docs = dataiku.Folder("SoQWOnhR")          # Input folder
A220_tech_docs_prep = dataiku.Folder("rhnW9xGx")    # Output folder

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

        # Create upload file object for Mistral API
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_file.stem,
                "content": pdf_file.read_bytes(),
            },
            purpose="ocr",
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        pdf_response = client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest", include_image_base64=True)
        response_dict = json.loads(pdf_response.json())
        json_string = json.dumps(response_dict, indent=4)
        
        # Écrire le fichier .json
        json_file_name = os.path.splitext(pdf_file)[0] + ".json"
        with A220_tech_docs_prep.get_writer(json_file_name) as writer:
            writer.write(json_string.encode('utf-8'))
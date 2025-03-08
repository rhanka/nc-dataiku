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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Folders
A220_tech_docs = dataiku.Folder("W8lS5GmB")          # Input folder
A220_tech_docs_prep = dataiku.Folder("d7DdDueY")    # Output folder

# Lister les fichiers PDF
pdf_files = [f for f in A220_tech_docs.list_paths_in_partition() if f.lower().endswith(".pdf")]
pdf_files.sort()

# Lister les fichiers JSON existants pour éviter les doublons
existing_json_files = set(A220_tech_docs_prep.list_paths_in_partition())

index = 0
while index < len(pdf_files):
    pdf_file = pdf_files[index]
    json_file_name = os.path.splitext(pdf_file)[0] + ".json"

    # Vérifier si le fichier JSON existe déjà
    if json_file_name in existing_json_files:
        print(f"{json_file_name} existe déjà, passe au suivant.")
        index += 1
        continue

    # Lire le contenu PDF
    with A220_tech_docs.get_download_stream(pdf_file) as f:
        print(f"Traitement du fichier : {pdf_file}")
        try:
            uploaded_file = client.files.upload(
                file={
                    "file_name": pdf_file,
                    "content": f.read(),
                },
                purpose="ocr",
            )
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
        except Exception as e:
            print(f"Erreur lors du traitement de {pdf_file}: {e}. Réessai dans 2 secondes.")
            time.sleep(2)
            try:
                pdf_response = client.ocr.process(
                    document=DocumentURLChunk(document_url=signed_url.url),
                    model="mistral-ocr-latest",
                    include_image_base64=True
                )
            except Exception as e:
                print(f"Échec répété pour {pdf_file}, ajout en fin de liste: {e}")
                pdf_files.append(pdf_file)
                index += 1
                continue

        response_dict = json.loads(pdf_response.json())
        json_string = json.dumps(response_dict, indent=4)

        # Écrire le fichier .json
        with A220_tech_docs_prep.get_writer(json_file_name) as writer:
            writer.write(json_string.encode('utf-8'))

    index += 1
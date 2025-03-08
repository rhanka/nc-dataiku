# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import tempfile
import base64
from io import BytesIO

client = dataiku.api_client()
project = client.get_default_project()
auth_info = client.get_auth_info(with_secrets=True)
MISTRAL_API_KEY = None
for secret in auth_info["secrets"]:
    if secret["key"] == "MISTRAL_API_KEY":
        MISTRAL_API_KEY = secret["value"]

client = Mistral(api_key=MISTRAL_API_KEY)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import time

# Paramétrer le nombre de requêtes parallèles
MAX_WORKERS = 10

# Folders
A220_tech_docs = dataiku.Folder("W8lS5GmB")          # Input folder
A220_tech_docs_prep = dataiku.Folder("d7DdDueY")    # Output folder

# Lister les fichiers PDF
pdf_files = [f for f in A220_tech_docs.list_paths_in_partition() if f.lower().endswith(".pdf")]
pdf_files.sort()

# Lister les fichiers existants
existing_files = set(A220_tech_docs_prep.list_paths_in_partition())


def extract_md_and_images(json_file_name, response_dict):
    base_name = os.path.splitext(json_file_name)[0]

    # Extraire et écrire le Markdown
    md_file_name = base_name + ".md"
    if md_file_name not in existing_files:
        print(f"Extraction md et images : {md_file_name}")
        markdown_content = "\n\n".join(page["markdown"] for page in response_dict.get("pages", []))
        with A220_tech_docs_prep.get_writer(md_file_name) as writer:
            writer.write(markdown_content.encode('utf-8'))
    else:
        print(f"{md_file_name} existe déjà.")

    # Extraire et écrire les images
    for page in response_dict.get("pages", []):
        for image in page.get("images", []):
            image_file_name = base_name + "-" + image["id"]
            #if image_file_name not in existing_files:
            if True:
                image_data = image["image_base64"].split(",")[1]
                image_bytes = base64.b64decode(image_data)
                with A220_tech_docs_prep.get_writer(image_file_name) as writer:
                    writer.write(BytesIO(image_bytes).getvalue())


def process_pdf(pdf_file):
    json_file_name = os.path.splitext(pdf_file)[0] + ".json"

    # Vérifier si le JSON existe déjà
    if json_file_name in existing_files:
        print(f"{json_file_name} existe déjà.")

        # Charger le JSON existant pour extraction MD et images
        with A220_tech_docs_prep.get_download_stream(json_file_name) as reader:
            response_dict = json.load(reader)
            extract_md_and_images(json_file_name, response_dict)
        return

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
                print(f"Échec répété pour {pdf_file}, réessai plus tard: {e}")
                return pdf_file  # à réessayer plus tard

        response_dict = json.loads(pdf_response.json())
        json_string = json.dumps(response_dict, indent=4)

        # Écrire le fichier .json
        with A220_tech_docs_prep.get_writer(json_file_name) as writer:
            writer.write(json_string.encode('utf-8'))

        # Extraire Markdown et images
        extract_md_and_images(json_file_name, response_dict)


pending_files = pdf_files
while pending_files:
    retry_files = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pending_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                retry_files.append(result)

    pending_files = retry_files
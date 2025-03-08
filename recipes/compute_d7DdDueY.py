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
import base64
from io import BytesIO

# Paramétrer le nombre de requêtes parallèles
MAX_WORKERS = 3

# Folders
A220_tech_docs = dataiku.Folder("W8lS5GmB")          # Input folder
A220_tech_docs_prep = dataiku.Folder("d7DdDueY")    # Output folder

# Lister les fichiers PDF
pdf_files = [f for f in A220_tech_docs.list_paths_in_partition() if f.lower().endswith(".pdf")]
pdf_files.sort()

# Lister les fichiers existants
existing_files = set(A220_tech_docs_prep.list_paths_in_partition())

def generate_image_description(image_data):
    """
    Utilise Mistral Vision (Pixtral) pour générer une description de l'image
    """
    try:
        # Décoder l'image base64
        image_bytes = base64.b64decode(image_data.split(",")[1])
        
        # Appeler l'API Mistral Vision avec le modèle pixtral-large-2411
        response = client.chat.completions.create(
            model="pixtral-large-2411",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Provide a description in markdown of this image coming from an A220 technical document, using exact terms if any included in the picture."},
                        {"type": "image", "image_url": {"url": f"data:image/png;base64,{image_data.split(',')[1]}"}}
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extraire la description générée
        description = response.choices[0].message.content
        return description
    
    except Exception as e:
        print(f"Erreur lors de la génération de description d'image: {e}")
        return "Description non disponible: erreur de traitement"

def extract_md_and_images(json_file_name, response_dict):
    base_name = os.path.splitext(json_file_name)[0]
    
    # Extraire et écrire le Markdown
    md_file_name = base_name + ".md"
    if md_file_name not in existing_files:
        markdown_content = "\n\n".join(page["markdown"] for page in response_dict.get("pages", []))
        with A220_tech_docs_prep.get_writer(md_file_name) as writer:
            writer.write(markdown_content.encode('utf-8'))
    
    # Extraire, écrire les images et générer leurs descriptions
    for page in response_dict.get("pages", []):
        for image in page.get("images", []):
            # Nom de fichier pour l'image
            image_file_name = base_name + "-" + image["id"]
            
            # Écrire l'image si elle n'existe pas déjà
            if image_file_name not in existing_files:
                image_data = image["image_base64"]
                image_bytes = base64.b64decode(image_data.split(",")[1])
                with A220_tech_docs_prep.get_writer(image_file_name) as writer:
                    writer.write(BytesIO(image_bytes).getvalue())
            
            # Générer et écrire la description de l'image
            description_file_name = image_file_name + ".md"
            if description_file_name not in existing_files:
                # Générer la description avec Mistral Vision
                description = generate_image_description(image["image_base64"])
                
                # Écrire la description dans un fichier .md
                with A220_tech_docs_prep.get_writer(description_file_name) as writer:
                    writer.write(description.encode('utf-8'))
                
                print(f"Description générée pour {image_file_name}")

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
        
        # Extraire Markdown et images avec descriptions
        extract_md_and_images(json_file_name, response_dict)

# Exécution principale
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
    if retry_files:
        print(f"{len(retry_files)} fichiers à réessayer...")
        time.sleep(5)  # Attendre avant de réessayer
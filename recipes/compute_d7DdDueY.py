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
import re
from io import BytesIO
from mistralai import Mistral

# Paramétrer le nombre de requêtes parallèles
MAX_WORKERS = 2

# Folders
A220_tech_docs = dataiku.Folder("W8lS5GmB")          # Input folder
A220_tech_docs_prep = dataiku.Folder("d7DdDueY")    # Output folder

# Vision prompt
vision_prompt = """
    Provide a description in markdown of this image coming from an A220 technical document,
    using exact terms if any included in the picture. 
    Don't wrap the markdown with ``` and only provide the description without additionnal comment.
"""
pixtral_model="pixtral-12b-2409"

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
        # Préparer l'image au format base64 pour l'API Mistral
        base64_image = image_data.split(",")[1] if "," in image_data else image_data
        
        # Préparer les messages pour l'API Mistral
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": vision_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Appeler l'API Mistral Vision avec le modèle pixtral-large-2411
        response = client.chat.complete(
            model=pixtral_model,
            messages=messages
        )
        
        # Extraire la description générée
        description = response.choices[0].message.content
        return description
    
    except Exception as e:
        print(f"Erreur lors de la génération de description d'image: {e}")
        # Imprimer plus de détails pour le débogage
        import traceback
        traceback.print_exc()
        # Retourner None pour indiquer l'échec
        return None

def clean_description_for_alt_text(description):
    """
    Nettoie le texte de description markdown pour l'utiliser comme texte alternatif.
    - Enlève les sauts de ligne
    - Supprime les formatages markdown comme # et *
    - Supprime les éléments de table markdown
    - Supprime les espaces multiples
    """
    # Remplacer les sauts de ligne par des espaces
    cleaned = description.replace('\n', ' ')
    
    # Supprimer les titres markdown (# Titre)
    cleaned = re.sub(r'#+\s+', '', cleaned)
    
    # Supprimer les formatages comme * ou _, ~, `, etc.
    cleaned = re.sub(r'[*_~`]', '', cleaned)
    
    # Supprimer les liens markdown
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
    
    # Supprimer les éléments de table markdown (|, ----, etc.)
    cleaned = re.sub(r'\|', ' ', cleaned)  # Remplacer les | par des espaces
    cleaned = re.sub(r'[-:]+', '', cleaned)  # Supprimer les lignes de séparation de table (----, :---:, etc.)
    
    # Supprimer les blocs de code ```
    cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
    
    # Supprimer les listes à puces/numérotées
    cleaned = re.sub(r'^\s*[\-\*\+]\s+', '', cleaned)  # Listes à puces
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned)     # Listes numérotées
    
    # Supprimer les espaces multiples
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Supprimer les espaces en début et fin de chaîne
    cleaned = cleaned.strip()
        
    return cleaned

def extract_md_and_images(json_file_name, response_dict):
    base_name = os.path.splitext(json_file_name)[0]
    md_file_name = base_name + ".md"
    
    # Dictionnaire pour stocker les descriptions d'images
    image_descriptions = {}
    
    # Créer une copie du dictionnaire response_dict pour le JSON enrichi
    enriched_json = json.loads(json.dumps(response_dict))
    
    # Extraire, écrire les images et générer leurs descriptions
    has_images = False
    for page_index, page in enumerate(response_dict.get("pages", [])):
        if page.get("images", []):
            has_images = True
        
        # Préparer la version alternative du markdown de la page
        page_markdown = page.get("markdown", "")
        page_markdown_alt = page_markdown
        
        for image_index, image in enumerate(page.get("images", [])):
            # Nom de fichier pour l'image
            image_file_name = base_name + "-" + image["id"]
            image_id = image["id"]  # ID utilisé dans le markdown pour référencer l'image
            
            # Écrire l'image si elle n'existe pas déjà
            if image_file_name not in existing_files:
                image_data = image["image_base64"]
                image_bytes = base64.b64decode(image_data.split(",")[1])
                with A220_tech_docs_prep.get_writer(image_file_name) as writer:
                    writer.write(BytesIO(image_bytes).getvalue())
            
            # Générer et écrire la description de l'image
            # Utiliser le format base_name-img-id.md pour la description
            description_file_name = base_name + "-" + image_id.split('.')[0] + ".md"
            if description_file_name not in existing_files:
                # Générer la description avec Mistral Vision
                description = generate_image_description(image["image_base64"])
                
                # Vérifier si la génération a échoué
                if description is None:
                    print(f"Échec lors de la génération de description pour l'image {image_id}")
                    return
                
                # Écrire la description dans un fichier .md
                with A220_tech_docs_prep.get_writer(description_file_name) as writer:
                    writer.write(description.encode('utf-8'))
                
                print(f"Description générée pour {image_id} et sauvegardée dans {description_file_name}")
            else:
                # Lire la description existante
                with A220_tech_docs_prep.get_download_stream(description_file_name) as reader:
                    description = reader.read().decode('utf-8')
            
            # Stocker la description pour mise à jour du markdown
            image_descriptions[image_id] = clean_description_for_alt_text(description)
            
            # Ajouter la description à l'image dans le JSON enrichi
            enriched_json["pages"][page_index]["images"][image_index]["description"] = description
            
            # Mettre à jour le markdown alternatif de la page avec la description
            # Format standard
            pattern_standard = rf'!\[{image_id}\]\({image_id}\)'
            replacement_standard = f'![{image_descriptions[image_id]}]({image_id})'
            page_markdown_alt = re.sub(pattern_standard, replacement_standard, page_markdown_alt)
            
            # Format sans texte alternatif
            pattern_no_desc = rf'!\[\]\({image_id}\)'
            page_markdown_alt = re.sub(pattern_no_desc, replacement_standard, page_markdown_alt)
            
            # Anciens patterns
            pattern_hash = rf'!\[\]?\(#image-{image_id}\)'
            replacement_hash = f'![{image_descriptions[image_id]}](#image-{image_id})'
            page_markdown_alt = re.sub(pattern_hash, replacement_hash, page_markdown_alt)
            
            # Pattern avec texte alternatif existant
            pattern_existing = rf'!\[([^\]]*)\]\({image_id}\)'
            page_markdown_alt = re.sub(pattern_existing, replacement_standard, page_markdown_alt)
        
        # Ajouter le markdown alternatif à la page dans le JSON enrichi
        enriched_json["pages"][page_index]["markdown_alt"] = page_markdown_alt
    
    # Vérifier s'il y a des images dans le document
    if not has_images:
        print(f"Aucune image trouvée dans {json_file_name}, pas de création de fichier avec descriptions.")
        
        # Extraire et écrire le Markdown original sans modifications
        if md_file_name not in existing_files:
            markdown_content = "\n\n".join(page["markdown"] for page in response_dict.get("pages", []))
            with A220_tech_docs_prep.get_writer(md_file_name) as writer:
                writer.write(markdown_content.encode('utf-8'))
        return
    
    # Debug: Voir quelles images ont été traitées
    print(f"Images trouvées et décrites: {list(image_descriptions.keys())}")
    
    # Vérifier s'il y a des images et des descriptions dans le document
    if not image_descriptions:
        print(f"Aucune image trouvée ou décrite dans {json_file_name}, pas de création de fichier avec descriptions.")
        return
    
    # Sauvegarder le JSON enrichi
    json_with_desc_file_name = base_name + "__with_img_desc.json"
    if json_with_desc_file_name not in existing_files:
        json_string = json.dumps(enriched_json, indent=4)
        with A220_tech_docs_prep.get_writer(json_with_desc_file_name) as writer:
            writer.write(json_string.encode('utf-8'))
        print(f"Fichier JSON enrichi {json_with_desc_file_name} créé avec descriptions d'images.")
    else:
        print(f"Le fichier JSON enrichi {json_with_desc_file_name} existe déjà, pas de régénération.")
        
    # Créer un nouveau fichier markdown avec les descriptions d'images
    new_md_file_name = base_name + "__with_img_desc.md"
    
    # Vérifier si le fichier markdown avec descriptions existe déjà
    if new_md_file_name in existing_files:
        print(f"Le fichier {new_md_file_name} existe déjà, pas de régénération.")
        return
    
    # Vérifier si le fichier markdown original existe
    if md_file_name in existing_files:
        # Lire le contenu du markdown original
        with A220_tech_docs_prep.get_download_stream(md_file_name) as reader:
            markdown_content = reader.read().decode('utf-8')
    else:
        # Générer le markdown à partir des données de réponse
        markdown_content = "\n\n".join(page["markdown"] for page in response_dict.get("pages", []))
        # Écrire le markdown original s'il n'existe pas encore
        with A220_tech_docs_prep.get_writer(md_file_name) as writer:
            writer.write(markdown_content.encode('utf-8'))
    
    # Écrire la version du markdown avec les descriptions d'images
    # On peut utiliser le markdown_alt généré dans le JSON enrichi
    markdown_with_desc = "\n\n".join(page.get("markdown_alt", page.get("markdown", "")) 
                                   for page in enriched_json.get("pages", []))
    with A220_tech_docs_prep.get_writer(new_md_file_name) as writer:
        writer.write(markdown_with_desc.encode('utf-8'))
        print(f"Fichier {new_md_file_name} créé avec les descriptions d'images.")

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
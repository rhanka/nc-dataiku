# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import tempfile
import base64
from io import BytesIO
import traceback

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
import traceback
from io import BytesIO
from mistralai import Mistral
import httpx
import threading

# Paramétrer le nombre de requêtes parallèles
MAX_WORKERS = 4

# Folders
A220_tech_docs = dataiku.Folder("W8lS5GmB")          # Input folder
A220_tech_docs_prep = dataiku.Folder("d7DdDueY")    # Output folder

# Vision prompt
vision_prompt = """
    Provide a description in markdown of this image coming from an A220 technical document,
    using exact terms if any included in the picture.
    Don't wrap the markdown with any format backquotes ``` and only provide the description without additionnal comment.
"""
pixtral_model = "pixtral-12b-2409"

# Nombre maximal de tentatives pour les appels Pixtral
MAX_PIXTRAL_RETRIES = 2
# Délai entre les tentatives (secondes)
PIXTRAL_RETRY_DELAY = 1

# Lister les fichiers PDF
pdf_files = [f for f in A220_tech_docs.list_paths_in_partition() if f.lower().endswith(".pdf")]
pdf_files.sort()

# Lister les fichiers existants
existing_files = set(A220_tech_docs_prep.list_paths_in_partition())

# Compteurs globaux pour le suivi
total_images_processed = 0
total_descriptions_generated = 0
total_pdfs_ocrized = 0

# Compteurs d'erreurs et de retries
total_ocr_errors = 0
total_pixtral_errors = 0
total_retries = 0

# Lock pour l'écriture de fichiers dans le dossier de sortie
folder_lock = threading.Lock()

def generate_image_description(image_data, retries=MAX_PIXTRAL_RETRIES):
    """
    Utilise Mistral Vision (Pixtral) pour générer une description de l'image
    avec mécanisme de réessai en cas d'erreur SSL ou réseau
    """
    global total_pixtral_errors
    
    if retries <= 0:
        # Message simplifié pour réduire le verbosité
        return None
    
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
        
        # Appeler l'API Mistral Vision avec le modèle pixtral
        response = client.chat.complete(
            model=pixtral_model,
            messages=messages
        )
        
        # Extraire la description générée
        description = response.choices[0].message.content
        return description
    except (httpx.ReadError, httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        # Gérer spécifiquement les erreurs SSL et de connexion
        total_pixtral_errors += 1
        print(f"Erreur SSL/connexion lors de l'appel à Pixtral: {e}")
        time.sleep(PIXTRAL_RETRY_DELAY)
        return generate_image_description(image_data, retries-1)
    except Exception as e:
        # Autres types d'erreurs
        total_pixtral_errors += 1
        print(f"Erreur lors de la génération de description d'image: {e}")
        
        # Pour certaines erreurs, on peut vouloir réessayer
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower() or "connection" in str(e).lower() or "ssl" in str(e).lower():
            time.sleep(PIXTRAL_RETRY_DELAY)
            return generate_image_description(image_data, retries-1)
        
        # Pour les autres types d'erreurs, on abandonne directement
        return None

def clean_description_for_alt_text(description):
    """
    Nettoie le texte de description markdown pour l'utiliser comme texte alternatif.
    - Enlève les sauts de ligne
    - Supprime les formatages markdown comme # et *
    - Supprime les éléments de table markdown
    - Supprime les espaces multiples
    """
    if not description:
        return "Image sans description"
        
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

def safe_write_to_folder(folder, file_path, content, is_binary=False):
    """
    Fonction pour écrire en toute sécurité dans un dossier avec un verrou
    """
    with folder_lock:
        try:
            with folder.get_writer(file_path) as writer:
                if is_binary:
                    writer.write(content)
                else:
                    writer.write(content.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Erreur lors de l'écriture du fichier {file_path}: {e}")
            return False

def safe_read_from_folder(folder, file_path, binary=False):
    """
    Fonction pour lire en toute sécurité depuis un dossier avec un verrou
    """
    with folder_lock:
        try:
            with folder.get_download_stream(file_path) as reader:
                if binary:
                    return reader.read()
                else:
                    return reader.read().decode('utf-8')
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            return None

def extract_md_and_images(json_file_name, response_dict, retry_files=None):
    global total_images_processed, total_descriptions_generated, total_retries
    
    base_name = os.path.splitext(json_file_name)[0]
    md_file_name = base_name + ".md"
    original_pdf = base_name + ".pdf"  # Pour réessayer en cas d'erreur Pixtral
    
    # Dictionnaire pour stocker les descriptions d'images
    image_descriptions = {}
    
    # Créer une copie du dictionnaire response_dict pour le JSON enrichi
    enriched_json = json.loads(json.dumps(response_dict))
    
    # Extraire, écrire les images et générer leurs descriptions
    has_images = False
    pixtral_error = False
    
    # Compteurs locaux pour ce fichier
    images_in_file = 0
    descriptions_generated = 0
    
    for page_index, page in enumerate(response_dict.get("pages", [])):
        if page.get("images", []):
            has_images = True
        
        # Préparer la version alternative du markdown de la page
        page_markdown = page.get("markdown", "")
        page_markdown_alt = page_markdown
        
        for image_index, image in enumerate(page.get("images", [])):
            # Incrémenter le compteur d'images
            images_in_file += 1
            total_images_processed += 1
            
            # Nom de fichier pour l'image
            image_file_name = base_name + "-" + image["id"]
            image_id = image["id"]  # ID utilisé dans le markdown pour référencer l'image
            
            # Écrire l'image si elle n'existe pas déjà
            if image_file_name not in existing_files:
                try:
                    image_data = image["image_base64"]
                    image_bytes = base64.b64decode(image_data.split(",")[1])
                    if not safe_write_to_folder(A220_tech_docs_prep, image_file_name, BytesIO(image_bytes).getvalue(), is_binary=True):
                        continue
                except Exception as e:
                    print(f"Erreur lors de l'écriture de l'image {image_id}: {e}")
                    continue
            
            # Générer et écrire la description de l'image
            # Utiliser le format base_name-img-id.md pour la description
            description_file_name = base_name + "-" + image_id.split('.')[0] + ".md"
            
            if description_file_name not in existing_files:
                # Générer la description avec Mistral Vision
                description = generate_image_description(image["image_base64"])
                
                # Vérifier si la génération a échoué
                if description is None:
                    print(f"Échec de description pour {image_id} dans {original_pdf}")
                    pixtral_error = True
                    # On continue pour traiter les autres images, mais on marquera le fichier pour réessai
                    continue
                
                # Écrire la description dans un fichier .md
                if not safe_write_to_folder(A220_tech_docs_prep, description_file_name, description):
                    continue
                
                # Incrémenter le compteur de descriptions générées
                descriptions_generated += 1
                total_descriptions_generated += 1
            else:
                # Lire la description existante
                description = safe_read_from_folder(A220_tech_docs_prep, description_file_name)
                if description is None:
                    description = "Erreur lors de la lecture de la description"
            
            # Stocker la description pour mise à jour du markdown
            image_descriptions[image_id] = clean_description_for_alt_text(description)
            
            # Ajouter la description à l'image dans le JSON enrichi
            try:
                enriched_json["pages"][page_index]["images"][image_index]["description"] = description
            except IndexError:
                print(f"Erreur d'index lors de l'ajout de la description à l'image {image_id}")
                continue
            
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
    
    # Si une erreur Pixtral s'est produite, ajouter le fichier à la liste des fichiers à réessayer
    if pixtral_error and retry_files is not None:
        print(f"Ajout de {original_pdf} à la file des réessais en raison d'erreurs Pixtral")
        retry_files.append(original_pdf)
        total_retries += 1
        return
    
    # Extraire et écrire le Markdown original sans modifications
    if md_file_name not in existing_files:
        markdown_content = "\n\n".join(page["markdown"] for page in response_dict.get("pages", []))
        safe_write_to_folder(A220_tech_docs_prep, md_file_name, markdown_content)
    
    # Ne générer les fichiers enrichis que s'il y a des images
    if has_images and image_descriptions:
        # Sauvegarder le JSON enrichi
        json_with_desc_file_name = base_name + "__with_img_desc.json"
        if json_with_desc_file_name not in existing_files:
            json_string = json.dumps(enriched_json, indent=4)
            safe_write_to_folder(A220_tech_docs_prep, json_with_desc_file_name, json_string)
        
        # Créer le markdown avec descriptions d'images
        md_with_desc_file_name = base_name + "__with_img_desc.md"
        if md_with_desc_file_name not in existing_files:
            # Générer le markdown à partir des données enrichies
            markdown_with_desc = "\n\n".join(page.get("markdown_alt", page.get("markdown", "")) 
                                            for page in enriched_json.get("pages", []))
            safe_write_to_folder(A220_tech_docs_prep, md_with_desc_file_name, markdown_with_desc)

def process_pdf(pdf_file, retry_files=None, current_count=None, total_count=None):
    global total_pdfs_ocrized, total_ocr_errors, total_retries
    
    json_file_name = os.path.splitext(pdf_file)[0] + ".json"
    
    # Vérifier si le JSON existe déjà
    if json_file_name in existing_files:
        # Charger le JSON existant pour extraction MD et images
        try:
            json_content = safe_read_from_folder(A220_tech_docs_prep, json_file_name)
            if json_content:
                response_dict = json.loads(json_content)
                extract_md_and_images(json_file_name, response_dict, retry_files)
            else:
                if retry_files is not None:
                    retry_files.append(pdf_file)
        except Exception as e:
            print(f"Erreur lors du chargement du JSON existant {json_file_name}: {e}")
            if retry_files is not None:
                retry_files.append(pdf_file)
        return
    
    # Lire le contenu PDF
    try:
        pdf_content = safe_read_from_folder(A220_tech_docs, pdf_file, binary=True)
        if pdf_content is None:
            if retry_files is not None:
                retry_files.append(pdf_file)
            return
            
        try:
            uploaded_file = client.files.upload(
                file={
                    "file_name": pdf_file,
                    "content": pdf_content,
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
            print(f"Erreur lors du traitement OCR de {pdf_file}: {e}. Réessai dans 2 secondes.")
            total_ocr_errors += 1
            time.sleep(2)
            try:
                pdf_response = client.ocr.process(
                    document=DocumentURLChunk(document_url=signed_url.url),
                    model="mistral-ocr-latest",
                    include_image_base64=True
                )
            except Exception as e:
                print(f"Échec répété pour l'OCR de {pdf_file}, ajout à la file de réessai: {e}")
                total_ocr_errors += 1
                if retry_files is not None:
                    retry_files.append(pdf_file)
                    total_retries += 1
                return
        
        response_dict = json.loads(pdf_response.json())
        json_string = json.dumps(response_dict, indent=4)
        
        # Écrire le fichier .json
        if not safe_write_to_folder(A220_tech_docs_prep, json_file_name, json_string):
            if retry_files is not None:
                retry_files.append(pdf_file)
            return
        
        # Incrémenter le compteur de PDF OCRisés
        total_pdfs_ocrized += 1
        
        # Extraire Markdown et images avec descriptions
        extract_md_and_images(json_file_name, response_dict, retry_files)
    except Exception as e:
        print(f"Erreur générale lors du traitement de {pdf_file}: {e}")
        traceback.print_exc()
        total_ocr_errors += 1
        if retry_files is not None:
            retry_files.append(pdf_file)
            total_retries += 1

# Exécution principale
pending_files = pdf_files
max_iterations = 10  # Limite pour éviter les boucles infinies
iteration = 0
total_files = len(pending_files)
progress_interval = 20  # Afficher l'avancement tous les 20 fichiers

while pending_files and iteration < max_iterations:
    iteration += 1
    print(f"Itération {iteration}/{max_iterations}, {len(pending_files)} fichiers à traiter")
    retry_files = []
    
    # Réinitialiser les compteurs pour cette itération
    initial_images_count = total_images_processed
    initial_descriptions_count = total_descriptions_generated
    initial_pdfs_ocrized = total_pdfs_ocrized
    initial_ocr_errors = total_ocr_errors
    initial_pixtral_errors = total_pixtral_errors
    initial_retries = total_retries
    
    # Compteur pour suivre l'avancement
    processed_count = 0
    total_in_batch = len(pending_files)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pdf, pdf, retry_files, processed_count + i + 1, total_in_batch): pdf 
                  for i, pdf in enumerate(pending_files)}
        
        for future in as_completed(futures):
            try:
                future.result()
                processed_count += 1
                
                # Afficher la progression tous les X fichiers
                if processed_count % progress_interval == 0 or processed_count == total_in_batch:
                    new_images = total_images_processed - initial_images_count
                    new_descriptions = total_descriptions_generated - initial_descriptions_count
                    new_pdfs_ocrized = total_pdfs_ocrized - initial_pdfs_ocrized
                    new_ocr_errors = total_ocr_errors - initial_ocr_errors
                    new_pixtral_errors = total_pixtral_errors - initial_pixtral_errors
                    new_retries = total_retries - initial_retries
                    
                    print(f"Avancement: {processed_count}/{total_in_batch} fichiers ({(processed_count/total_in_batch)*100:.1f}%) - "
                          f"OCRisés: {new_pdfs_ocrized} - Images: {new_images} - Descriptions: {new_descriptions} - "
                          f"Erreurs OCR: {new_ocr_errors}, Pixtral: {new_pixtral_errors} - Retries: {new_retries}")
            
            except Exception as e:
                pdf = futures[future]
                print(f"Exception non gérée lors du traitement de {pdf}: {e}")
                traceback.print_exc()
                retry_files.append(pdf)
                processed_count += 1
    
    # Afficher un résumé de l'itération
    print(f"Fin de l'itération {iteration}: {processed_count} fichiers traités")
    print(f"  - PDF OCRisés: {total_pdfs_ocrized} au total")
    print(f"  - Images traitées: {total_images_processed} au total")
    print(f"  - Descriptions générées: {total_descriptions_generated} au total")
    print(f"  - Erreurs OCR: {total_ocr_errors}, Pixtral: {total_pixtral_errors}")
    print(f"  - Fichiers à réessayer: {len(retry_files)}, Total retries: {total_retries}")
    
    # Éviter les boucles infinies en vérifiant si les mêmes fichiers échouent constamment
    if set(retry_files) == set(pending_files) and retry_files:
        print(f"Mêmes fichiers en échec après {iteration} itérations, arrêt du traitement.")
        print(f"Fichiers en échec permanent: {retry_files}")
        break
    
    pending_files = retry_files
    if retry_files:
        print(f"{len(retry_files)} fichiers à réessayer...")
        # Augmenter le délai à chaque itération
        delay = 5 * iteration  # 5s, 10s, 15s, etc.
        print(f"Attente de {delay} secondes avant la prochaine tentative...")
        time.sleep(delay)

if pending_files and iteration >= max_iterations:
    print(f"Nombre maximal d'itérations atteint. Fichiers restants non traités: {pending_files}")

# Afficher les statistiques finales
print(f"\nRésumé final:")
print(f"Total des fichiers traités: {total_files - len(pending_files)}/{total_files}")
print(f"Total des PDF OCRisés: {total_pdfs_ocrized}")
print(f"Total des images traitées: {total_images_processed}")
print(f"Total des descriptions générées: {total_descriptions_generated}")
print(f"Total des erreurs OCR: {total_ocr_errors}")
print(f"Total des erreurs Pixtral: {total_pixtral_errors}")
print(f"Total des retries: {total_retries}")
print(f"Fichiers non traités: {len(pending_files)}")
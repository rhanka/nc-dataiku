import dataiku
import pandas as pd
from flask import request
from flask_cors import CORS
import json
from langchain.chains.question_answering import load_qa_chain
from dataiku.langchain.dku_llm import DKULLM, DKUChatLLM

CORS(app, resources={r"/*": {"origins": "https://svelte.dev"}})

@app.route('/nc')
def non_conformities():
    # Récupération des arguments de requête
    max_rows = int(request.args.get('max_rows', 500))  # Par défaut, limite à 500 lignes
    record_id = request.args.get('id')  # Récupère l'id s'il est présent dans la requête

    # Charger le dataset
    mydataset = dataiku.Dataset("NC_types_random_500_final_structured")
    mydataset_df = mydataset.get_dataframe()

    # Convertir la colonne de date et analyser le champ JSON
    mydataset_df['nc_event_date'] = mydataset_df['nc_event_date'].astype(str)
    mydataset_df['analysis_history'] = mydataset_df['analysis_history'].apply(json.loads)

    # Si un id est fourni, filtrez les données
    if record_id:
        filtered_df = mydataset_df[mydataset_df['nc_event_id'] == record_id]
        data = filtered_df.to_dict(orient='records')
    else:
        # Sinon, limitez à max_rows
        data = mydataset_df.head(max_rows).to_dict(orient='records')
    
    return json.dumps(data)

#LLM_ID = "retrievalaugmented:8jpMQW44:gpt-4o-mini-a220-all-sources"
#LLM_ID = "retrievalaugmented:WnKb6p17:gpt-4o-mini-a220-nc"
#LLM_ID = "retrievalaugmented:zQ92IhQ9:gpt-4o-mini-a220-tech-docs"
LLM_ID = "openai:OpenAI-FA:gpt-4o-mini"
KB_IDs = {
    "tech_docs": "zQ92IhQ9",
    "non_conformities": "WnKb6p17"
}

# Create a handle for the LLM of your choice
client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm(LLM_ID)



# Preparing the Knowledge Bank, Vector store and LLM
KBs = {
    key: dataiku.KnowledgeBank(id=value, project_key=project.project_key)
    for key, value in KB_IDs.items()
}
vector_stores = {
    key: value.as_langchain_vectorstore()
    for key, value in KBs.items()
}
# Create and run a completion query

#langchain_llm = DKUChatLLM(llm_id=LLM_ID, temperature=0)
#chain = load_qa_chain(langchain_llm, chain_type="stuff")

@app.route('/ai', methods=['POST'])
def ai():
    app.logger.info("Handling /ai endpoint.")
    # Récupérer le JSON envoyé dans la requête POST
    data = request.json

    # Vérifier que le champ "messages" est présent
    if not data or "messages" not in data:
        return json.dumps({"error": "Invalid input: 'messages' field is required."}), 400

    # Récupérer le dernier message utilisateur (assumé en dernier dans l'historique)
    messages = data["messages"]
    if not messages or len(messages) == 0:
        return json.dumps({"error": "Invalid input: 'messages' cannot be empty."}), 400

    roles = ["000", "100", "200", "300", "400", "500"]
    
    role = messages[-1]["text"] if messages[-1]["role"] in roles else "000"
    
    user_message = messages[-1]["text"]
    
    # 1s step: expand query
    prompt = f"""
        Une non conformité de l'A220 doit être traitée selon le processus suivant :
            
            000 - rapport de non-conformité par le Quality Controler
            100 - analyse et recommandation / plan d'action par le Design Office
            200 - validation de l'analyse / plan d'action par le Design Manager
            300 - calcul de structure lié au plan d'action et recommandation / selon le Stress Office
            400 - du calcul / plan d'action amendé par le Stress Manager
            500 - plan d'action final validé par le Quality Manager
        
        You're supporting the role for {role} and rely on the knowledge from the A220 technical 
        doc and non conformity knowledge base (vector databases). You must provide an optimized expanded 
        prompt towards those langchain vector databases to enable the best retrieval given the user input. 
        The expansion should only concern specificity of the domain (eg variants for wings or fuel) vocabulary and should NOT include any word
        about the processus itself (like non-conformity, words not relative to the process itself and especialy not
        further steps beyon {role} itself).
        Avoid too generic words like system integrity operations, efficiency, standards, component, procedure, failure,
        repair, troubleshooting.
        
        Format of the output: Please just provide a liste of words for vectord db query in engish without any comment to be reused as is. 
        Optimal request should be between 10 and 20 words
        
        The user is the following:
        {user_message}
        
        
        Remember to only provide the requested query for the knowledge database without any comment.
    """
    llm = project.get_llm(LLM_ID)
    completion = llm.new_completion()
    completion.with_message(prompt)
    completion.settings["temperature"] = 0
    resp = completion.execute()
    
    if resp.success:
        query = f"task {role} for {resp.text}"
        
        try:
            # 2nd step : gather documents relative to query
            search_results = [
                result
                for key, value in vector_stores.items() 
                for result in value.similarity_search(query)
            ]
            
            search_results = [
                    {
                        "doc": s.metadata['doc'],
                        "chunk_id": s.metadata['chunk_id'],
                        "chunk": s.page_content
                    }
                    for s in search_results
                ]

        except Exception as e:
            deep_chat_response = {
                    "text": f"Error while generating response",
                    "error": f"{e}",
                    "role": "ai"
                }
            return json.dumps(deep_chat_response)
        

    
        # 3rd step : give the best advice given the documents
        
        description_prompts = {
            "000": """La description doit contenir les section suivantes (rappel: en anglais, toujours et en markdown):
                - Designation: numéro de série de l'avion (MSN5020...), zone sur l'avion, code ATA, numério de pièce, date
                - Observation : Description factuelle de la non-conformité (sans jugement ou interprétation), avec des références aux documents de fabrication et/ou d’assemblage pertinents.
                - Root Cause: Cause identifiée de la non-conformité, ou mention « inconnue » si non déterminée.
                - Dimensions: Mesures (système métrique) caractérisant la non-conformité.
                - References: Lien vers les documents de référence (fabrication et/ou assemblage liés).
                A cette étape, la description ne contient ni l'analyse, ni la classification, ni la résolution ou plan d'action correctif.
            """,
            "100": """La description doit contenir les section suivantes (rappel: en anglais, toujours et en markdown):
                - Synthesis: Synthèse de l’analyse pour l’ATA concerné.
                - Subtasks demands: Demandes d’analyses supplémentaires (Tâches 101, 102, etc.) pour les ATA tiers impactés si nécessaire.
                - Classification: Classification de la non-conformité (T, C, R, etc.) selon son importance.
                - Resolution: Description de la solution retenue pour mettre en conformité (réparation, remplacement, etc.).
                - References: Lien vers les documents de référence (fabrication et/ou assemblage liés).
            """
        }
        
        description_prompt = description_prompt[role]
        
        prompt = f"""
            #Processus
            Une non conformité de l'A220 doit être traitée selon le processus suivant :

            000 - rapport de non-conformité par le Quality Controler
            100 - analyse et recommandation / plan d'action par le Design Office
            200 - validation de l'analyse / plan d'action par le Design Manager
            300 - calcul de structure lié au plan d'action et recommandation / selon le Stress Office
            400 - du calcul / plan d'action amendé par le Stress Manager
            500 - plan d'action final validé par le Quality Manager

            Vous supportez le role de l'étape {000} et devez rédiger de la facon la plus explicite en prenant
            les exemples fournis et la documentation technique.

            #Exemples et documentation technique:
            {json.dumps(search_results)}

            #La requête utilisateur est la suivante:
            {user_message}

            #Réponse
            ## Instruction globales du processus
            **Instructions du processus** :
            - **Analyse des causes** : Les causes doivent être réalistes et adaptées au contexte spécifique de l'A220, en tenant compte des impacts possibles.
            - **Causes internes et externes** : Différencier les causes internes (ex. : erreurs d'assemblage, calibrations incorrectes) et externes (ex. : défauts fournisseurs, intempéries).
            - **Orientation industrielle** : Prioriser les scénarios ayant un impact direct sur la navigabilité, la résistance (statique et fatigue), ou les coûts de production.

            **Orientation sur les gains industriels** :
            - **Réduction des coûts** : Prioriser les scénarios avec un impact financier élevé ou nécessitant des corrections coûteuses si elles ne sont pas détectées à temps.
            - **Efficacité temporelle** : Mettre en place des étapes d’analyse optimisées et des moyens de détection rapide pour réduire les délais de production.
            - **Pertinence industrielle** : Adopter une approche réaliste et contextuellement adaptée à l’industrie aéronautique, afin de garantir la navigabilité, la fiabilité et la conformité des produits.

            Les principes relatifs aux **non-conformités significatives** stipulent qu'une non-conformité qui peut affecter la navigabilité, la résistance (statique et fatigue), l’installation, le fonctionnement, ou tout autre domaine impactant la qualité et la sécurité doit être soigneusement évaluée et traitée. 

            Chaque **non-conformité significative** doit faire l'objet d'une demande de dérogation soumise à l'ingénierie pour une évaluation approfondie. Le processus de dérogation ne doit pas être utilisé pour des erreurs de conception ou des problèmes de configuration non anticipée. En outre, les **suffixes de dérogation** doivent être attribués pour définir les limitations permanentes ou temporaires sur les articles concernés.


            ## Instructions de réponse    
            Veuillez répondre pour l'étape {role}, en fournissant le meilleur 'label' et la meilleure 'description' possible selon les exemples, n'hésitant pas à illustrer selon les
            documentation technique le cas échéant. La description fournie doit être complètement rédigée.
            Si l'utilisateur a fourni un json avec un 'label' et une 'description' vous modifierez la description ou
            le titre selon les instruction de l'utilisateur, en maintenant un rôle de conseil vis à vis des exemples et
            de la documentation technique.
            Ne pas empiéter sur les rôles autres que {role}. Ainsi, a l'étape 000 on se contente de formuler le rapport de
            description de la non-conformité observée, on ne prend pas les rôles d'analyse des primary causes ni
            de préconisation de plan d'action ni d'analyse d'impact ou de calcul des structures. 
            Idem pour 100: on ne fait pas le calcul des structure, on se concentre sur l'analyse.



            ## Format de réponse
            Répondez en anglais sauf si l'utilisateur utilise une autre langue ou précise des instructions de langue.
            Format de réponse attendu en json sans autre mise en forme (pas de ```json). Vos commentaires sont fournis
            dans l'item 'comment':
            \\{{ label: ..., description: ..., comment: ...\\}}
            - 'description' est en markdown. Dans tous les cas, le style reste technique et concis, 
            avec une approche plus télégraphique que rédigée de manière complexe (pas de phrases longues 
            ou compliquées). Faire comme dans les exemples, sans ajouter de termes de type "ce rapport précise",
            le rapport sera fourni dans un outil de ticketting, il faut rester concis et précis.
            - label : ne pas mentionner 'A220 Non-Conformity Report', juste le label de la non conformité, 
            comme dans les exemples
            - comment: fourni le cas échéant en markdwon pour l'interaction en mode canevas avec l'utilisateur {role}
            
        """
        
        completion = llm.new_completion()
        completion.with_message(prompt)
        completion.settings["temperature"] = 0
        resp = completion.execute()
        
        # Vérifier le succès de la réponse
        if resp.success:
            try:
                response_content = json.loads(resp.text)
                deep_chat_response = {
                    "text": response_content['comment'],
                    "label": response_content['label'],
                    "description": response_content['description'],
                    "sources": search_results,
                    "user_query": user_message,
                    "knowledge_query": query,
                    "role": "ai"
                }
                return json.dumps(deep_chat_response)
            except:
                deep_chat_response = {
                    "text": f"I couldn't update the form: \n{resp.text}",
                    "sources": search_results,
                    "user_query": user_message,
                    "knowledge_query": query,
                    "role": "ai"
                }
                return json.dumps(deep_chat_response)

        else:
            # En cas d'échec du modèle, retourner une réponse d'erreur
            deep_chat_response = { "text": "I'm sorry, I couldn't process your request.", error: "500" }
            return json.dumps(deep_chat_response), 500
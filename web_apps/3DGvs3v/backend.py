import dataiku
import pandas as pd
from flask import request
from flask_cors import CORS

CORS(app, resources={r"/*": {"origins": "https://svelte.dev"}})

from dataiku.langchain.dku_llm import DKULLM, DKUChatLLM

LLM_ID = "retrievalaugmented:8jpMQW44:gpt-4o-mini-a220-all-sources"

# Create a handle for the LLM of your choice
client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm(LLM_ID)

# Create and run a completion query
completion = llm.new_completion()

langchain_llm = DKUChatLLM(llm_id=LLM_ID)


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

    user_message = messages[-1]["text"] if messages[-1]["role"] == "user" else ""

    # Construire le prompt pour le modèle
    prompt = (
        f"You're supporting Quality Controller for A220 and rely on the knowledge from the A220 technical "
        f"doc and non conformity knowledge base. When answering questions, be sure to provide "
        f"answers that reflect the content of the knowledge base, but avoid saying things like "
        "'according to the knowledge base'. Instead, subtly mention that the information is based "
        "on the A220 knowledge base."
        f"Now try to answer the following question: {user_message}\n"
    )
    # Préparer et exécuter la requête au modèle LLM
    completion = llm.new_completion()
    completion.with_message(prompt)
    resp = completion.execute()

    # Vérifier le succès de la réponse
    if resp.success:
        try:
            # Tenter de convertir le texte en JSON si applicable
            response_content = json.loads(resp.text)
            deep_chat_response = {
                "text": response_content["result"],
                "sources": response_content["sources"],
                "role": "ai"
            }
        except json.JSONDecodeError:
            # Utiliser le texte brut si ce n'est pas un JSON valide
            deep_chat_response = { "text": resp.text, "role": "ai" }

        # Structure compatible DeepChat
        return json.dumps(deep_chat_response)

    else:
        # En cas d'échec du modèle, retourner une réponse d'erreur
        deep_chat_response = { "text": "I'm sorry, I couldn't process your request.", error: "500" }
        return json.dumps(deep_chat_response), 500
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
completion = llm.new_completion()

langchain_llm = DKUChatLLM(llm_id=LLM_ID, temperature=0)
chain = load_qa_chain(langchain_llm, chain_type="stuff")

@app.route('/ai', methods=['POST'])
def ai():
    app.logger.info("Handling /ai endpoint.")
    # Récupérer le JSON envoyé dans la requête POST
    data = request.json

    # Vérifier que le champ "messages" est présent
    if not data or "messages" not in data:
        return json.dumps({"error": "Invalid input: 'messages' field is required."}), 400

    
    # 1s step: expand query
    prompt = (
        f"You're supporting Quality Controller for A220 and rely on the knowledge from the A220 technical "
        f"doc and non conformity knowledge base (vector databases). You must provide an optimized expanded prompt towards "
        f"those vector databases to enable the best retrieval given the user input. "
        f"The expansion should only concern specificity around the user query and avoid retrieval of non specific vocabulary, "
        f"as knowledge databses will contain any past non conformity. Avoid generic vocabulary like 'non-conformity', 'issue', "
        f"'specification', 'standard', 'operations', 'maintainance'. But expand domain vocabulary.\n "
        f"Format of the output: Please just provide the query without any comment to be reused as is. "
        f"Optimal request should be between 20 and 50 words \n\n"
        f"The user is the following:\n {user_message}\n\n\n"
        f"Remember to only provide the requested query for the knowledge database without any comment."
    )
    llm = project.get_llm(LLM_ID)
    completion = llm.new_completion()
    completion.with_message(prompt)
    resp = completion.execute()
        
    if resp.success:
        # 2nd step : gather documents
        query = resp.text
        search_results = [
            result
            for key, value in vector_stores.items() 
            for result in value.similarity_search(query)
        ]
        search_results = [ {
                "doc": s.metadata['doc'],
                "chunk_id": s.metadata['chunk_id'],
                #"chunk": s.page_content
            }
            for s in search_results
        ]
print(search_results)

    # Récupérer le dernier message utilisateur (assumé en dernier dans l'historique)
    messages = data["messages"]
    if not messages or len(messages) == 0:
        return json.dumps({"error": "Invalid input: 'messages' cannot be empty."}), 400

    user_message = messages[-1]["text"] if messages[-1]["role"] == "user" else ""

    # Construire le prompt pour le modèle
    prompt = (
        f"You're supporting Quality Controller for A220 and rely on the knowledge from the A220 technical "
        f"doc and non conformity knowledge base. When answering questions, be sure to provide "
        f"answers that reflect the content of the knowledge base, including both pdf (tech docs) and "
        "md (non conformitities tasks) but avoid saying things like "
        "'according to the knowledge base'. Instead, subtly mention that the information is based "
        "on the A220 knowledge base."
        f"Now try to answer the following question: {user_message}\n"
    )
    # Préparer et exécuter la requête au modèle LLM
    completion = llm.new_completion()
    completion.with_message(prompt)
    resp = completion.execute()

    if resp.success:
        try:
            
    
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
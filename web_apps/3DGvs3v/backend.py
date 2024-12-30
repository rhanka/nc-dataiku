import dataiku
import pandas as pd
from flask import request
from flask_cors import CORS


CORS(app, resources={r"/*": {"origins": "https://svelte.dev"}})

from dataiku.langchain.dku_llm import DKULLM, DKUChatLLM

LLM_ID = "retrievalaugmented:zQ92IhQ9:gpt-4o-mini-a220-rag"

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

@app.route('/ai')
def ai():
    input = request.args.get('input')
    return json.dumps({msg: input.encode('utf-8')})

def bob():
    prompt = (
        f"Non conformity label: {input}\n"
        f"You're Quality Controler for A220 and rely on the knowledge from the A220 technical "
        f"doc and non conformity knowledge base. When answering questions, be sure to provide "
        f"answers that reflect the content of the knowledge base, but avoid saying things like "
        "'according to the knowledge base'. Instead, subtly mention that the information is based "
        "on the A220 knowledge base."
    )
    completion.with_message(prompt)
    resp = completion.execute()

    # Display the LLM output
    if resp.success:
        print(resp.text)
        return json.dumps({msg: text.encode('utf-8')})
    else:
        return json.dumps({msg:"failed"})
    
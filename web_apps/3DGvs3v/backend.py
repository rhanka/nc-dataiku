import dataiku
import pandas as pd
from flask import request, Response, jsonify, stream_with_context

from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException

import json
from langchain.chains.question_answering import load_qa_chain
from dataiku.langchain.dku_llm import DKULLM, DKUChatLLM
from dataikuapi.dss.llm import DSSLLMStreamedCompletionChunk, DSSLLMStreamedCompletionFooter


client = dataiku.api_client()
project = client.get_default_project()
auth_info = client.get_auth_info(with_secrets=True)
JWT_SECRET_KEY = None
MY_APP_USERNAME = None
MY_APP_PASSWORD = None
for secret in auth_info["secrets"]:
    if secret["key"] == "JWT_SECRET_KEY":
        JWT_SECRET_KEY = secret["value"]
    elif secret["key"] == "MY_APP_USERNAME":
        MY_APP_USERNAME = secret["value"]
    elif secret["key"] == "MY_APP_PASSWORD":
        MY_APP_PASSWORD = secret["value"]
        

if not JWT_SECRET_KEY or not MY_APP_USERNAME or not MY_APP_PASSWORD:
        raise Exception("secret not found")
        
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
jwt = JWTManager(app)

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Transforme les erreurs HTTP en réponse JSON."""
    response = e.get_response()
    response.data = jsonify({
        "error": e.name,
        "description": e.description,
        "status_code": e.code
    }).get_data()
    response.content_type = "application/json"
    return response


##########################################################################
#####                       data endpoints                      ##########
#####                 docs and non conformities                 ##########
##########################################################################

@app.route('/doc/<filename>', methods=['GET'])
def get_doc(filename):
    """
    Serve a PDF file from a folder dataset based on the filename.
    """
    try:
        # Replace "pdf_folder_dataset" with the actual folder dataset name
        folder = dataiku.Folder("SoQWOnhR")
        folder_path = folder.get_path()

        # Build the file path
        file_path = f"{folder_path}/{filename}"

        # Serve the file if it exists
        with open(file_path, 'rb') as pdf_file:
            return Response(
                pdf_file.read(),
                mimetype='application/pdf',
                headers={"Content-Disposition": f"inline; filename={filename}"}
            )
    except FileNotFoundError:
        return json.dumps({"error": f"File {filename} not found."}), 404
    except Exception as e:
        return json.dumps({"error": f"An error occurred: {str(e)}"}), 500


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


##########################################################################
#####                       AI config                           ##########
#####                     AI endpoints                          ##########
##########################################################################


agents = {
    "query": "compute_nc_scenarios_query",
    "nc_search": "compute_nc_scenarios_search_nc",
    "doc_search": "compute_nc_scenarios_search_techdocs",
    "000": "compute_nc_scenarios_propose_000",
    "propose_000": "compute_nc_scenarios_propose_000",
    "propose_100": "compute_nc_scenarios_propose_100",
    "100": "compute_nc_scenarios_propose_100"
}

agentsMsg = {
    "query": "Build appropriate request",
    "nc_search": "Search for similar non-conformities",
    "doc_search": "Search for relevant technical documents",
    "000": "Propose structured non-conformity report",
    "propose_000": "Propose structured non-conformity report",
    "propose_100": "Analysing non-conformity",
    "100": "compute_nc_scenarios_propose_100"
}

def completion_from_prompt_recipe(recipe_name, inputs):
    #partial method
    recipe = project.get_recipe(recipe_name)
    config = recipe.get_settings().get_json_payload()
    promptStudioId = { "id": config["associatedPromptStudioId"], "prompt_id": config["associatedPromptStudioPromptId"] }
    llm_id = config["llmId"]
    prompt_inputs = config["prompt"]["textPromptTemplateInputs"]
    system_prompt = config["prompt"]["textPromptSystemTemplate"]
    user_prompt = config["prompt"]["textPromptTemplate"]
    temperature = config["completionSettings"]["temperature"]
    for input_def in prompt_inputs:
        placeholder = f"{{{{{input_def['name']}}}}}"  # Exemple : {{description}}
        replacement = str(inputs[input_def["name"]])
        system_prompt = system_prompt.replace(placeholder, replacement)
        user_prompt = user_prompt.replace(placeholder, replacement)
    llm = project.get_llm(llm_id)
    completion = llm.new_completion()
    completion.settings["temperature"] = temperature
    completion.with_message(system_prompt, role='system')
    completion.with_message(user_prompt, role='user')
    return completion
    
def exec_prompt_recipe(recipe_name, inputs):
    resp = completion_from_prompt_recipe(recipe_name, inputs).execute()
    try:
        return json.loads(resp.text)
    except:
        return resp.text

def format_event_stream(input, metadata):
    data = {'v': input.replace('\n', '\\n')}
    if (metadata):
        data["metadata"] = metadata
    return f"event: delta\ndata: {json.dumps(data)}\n\n"

def format_data_stream(type, input,metadata):
    data = {'type': type, 'text': input}
    if (not isinstance(input, dict)):
        data['text'] = input.replace('\n', '\\n')
    if (metadata):
        data['metadata'] = metadata
    return f"data: {json.dumps(data)}\n\n"

def stream_prompt_recipe(recipe_name, inputs):
    agent_name = next((agent for agent, recipe in agents.items() if recipe == recipe_name), recipe_name)
    yield format_data_stream("action",agentsMsg[agent_name],agent_name)
    result = None
    for chunk in completion_from_prompt_recipe(recipe_name, inputs).execute_streamed():
        if isinstance(chunk, DSSLLMStreamedCompletionChunk):
            yield format_event_stream(chunk.data['text'],agent_name)
        elif isinstance(chunk, DSSLLMStreamedCompletionFooter):
            result = chunk.data['trace']['children'][1]['outputs']['text']
            yield format_data_stream("result",result,agent_name)
    return result

def consume(gen):
    """Iterate through a streaming generator but also capture its final return value."""
    final = None
    try:
        for _ in gen:
            pass
    except StopIteration as e:
        final = e.value
    return final

@app.route('/ai', methods=['POST'])
@stream_with_context
def ai():
    # Mode stream ou non
    stream = (request.headers.get('accept') == 'text/event-stream')
    app.logger.info(f"stream {stream}")
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
    
    role = messages[-1]["role"] if messages[-1] and (messages[-1]["role"] in roles) else "000"
    
    user_message = messages[-1]["text"]
    history = {}
    sources = None
    try:
        history = json.loads(messages[-1]["history"]) if messages[-1] and messages[-1]["history"] else {}
    except:
        history = {}
        
    try:
        sources = json.loads(messages[-1]["sources"]) if messages[-1] and messages[-1]["history"] else None
    except:
        sources = None
    
    if (not stream):
        if (not sources):
            # 1s step: expand query
            query = exec_prompt_recipe(agents["query"], {
                "role": role,
                "description" : user_message
            })

            # 2nd step : gather documents relative to query
            sources = {
                "non_conformities": exec_prompt_recipe(agents["nc_search"], {"input": query}),
                "tech_docs": exec_prompt_recipe(agents["doc_search"], {"input": query})
            }

        # 3rd step : give the best advice given the documents
        response_content = exec_prompt_recipe(agents[role], {
            "role": role,
            "description": user_message,
            "search_docs": json.dumps(sources["tech_docs"]),
            "search_nc": json.dumps(sources["non_conformities"]),
            "history": json.dumps(history)
        })

        return json.dumps({
            "text": response_content['comment'],
            "label": response_content['label'],
            "description": response_content['description'],
            "sources": sources,
            "user_query": user_message,
            "knowledge_query": query,
            "role": "ai",
            "user_role": role
        })
    else:
        def events(role,user_message,sources,history):
            yield "event: delta_encoding\ndata: \"v1\"\n\n"
            if (not sources):
            # 1s step: expand query
                app.logger.info("query")
                query = stream_prompt_recipe(agents["query"], {
                    "role": role,
                    "description" : user_message
                })
                try:
                    while True:
                        yield next(query)
                except StopIteration as e:
                    query = e.value  # capture final return
                
                # 2nd step : gather documents relative to query
                app.logger.info("nc_search")
                yield format_data_stream("action",agentsMsg["nc_search"],"nc_search")
                non_conformities = exec_prompt_recipe(agents["nc_search"], {"input": query})
                yield format_data_stream("result",non_conformities,"nc_search")
                
                app.logger.info("doc_search")
                yield format_data_stream("action",agentsMsg["doc_search"],"doc_search")
                tech_docs = exec_prompt_recipe(agents["doc_search"], {"input": query})
                yield format_data_stream("result",tech_docs,"doc_search")
                
                sources = {
                    "non_conformities": non_conformities,
                    "tech_docs": tech_docs
                }
                
            app.logger.info(agents[role])
            response_content = stream_prompt_recipe(agents[role], {
                "role": role,
                "description": user_message,
                "search_docs": json.dumps(sources["tech_docs"]),
                "search_nc": json.dumps(sources["non_conformities"]),
                "history": json.dumps(history)
            })
            try:
                while True:
                    yield next(response_content)
            except StopIteration as e:
                response_content = json.loads(e.value)  # capture final return
            
            app.logger.info("end")
            formatted_result = json.dumps({
                "text": response_content['comment'],
                "label": response_content['label'],
                "description": response_content['description'],
                "sources": sources,
                "user_query": user_message,
                "knowledge_query": query,
                "role": "ai",
                "user_role": role
            })
            yield format_data_stream("result",formatted_result, "final")

        return Response(events(role,user_message,sources,history), content_type='text/event-stream', headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}) 

        
##########################################################################
#####                       auth endpoints                      ##########
#####                       register, login                     ##########
##########################################################################

# Base de données simulée (dictionnaire)
users = { MY_APP_USERNAME: generate_password_hash(MY_APP_PASSWORD)}

# Route d'inscription
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username in users:
        return jsonify({"message": "User already exists"}), 400

    # Hachage du mot de passe
    hashed_password = generate_password_hash(password)
    users[username] = hashed_password
    return jsonify({"message": "User registered successfully"}), 201

# Route de connexion
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username not in users or not check_password_hash(users[username], password):
        return jsonify({"message": "Invalid credentials"}), 401

    # Génération du token JWT
    access_token = create_access_token(identity=str(username))
    refresh_token = create_refresh_token(identity=str(username))
    return jsonify(access_token=access_token, refresh_token=refresh_token), 200

# Route pour accéder aux ressources protégées
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({"message": f"Welcome {current_user}!"}), 200

# Route pour rafraîchir le token
@app.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify(access_token=new_access_token), 200
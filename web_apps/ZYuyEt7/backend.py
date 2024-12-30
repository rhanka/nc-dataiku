from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State
from dash import no_update
from dash import set_props

import dataiku
from dataiku.langchain.dku_llm import DKUChatLLM
from dataiku import SQLExecutor2
from duckduckgo_search import DDGS

from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Type

from textwrap import dedent

LOG_ALL_ANSWERS = False # Whether all answers or only answers with user feedback should be logged
WEBAPP_NAME = "qanda_webapp" # Name of the app (logged when a conversation is flagged)
VERSION = "1.0" # Version of the app (logged when a conversation is flagged)

# Folder to log answers and positive/negative reactions
answers_folder = dataiku.Folder("bsdWEIKi")

LLM_ID = "retrievalaugmented:zQ92IhQ9:gpt-4o-mini-a220-rag"
llm = DKUChatLLM(llm_id=LLM_ID, temperature=0)

KB_ID = "zQ92IhQ9"
documents = dataiku.Folder("SoQWOnhR")
documents_md = dataiku.Folder("AXB1Cyno")
documents_md_table = dataiku.Dataset("NC_types_random_500_md_concat")
non_conformities = dataiku.Folder("gZC3bHFN")
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""

## Tools

class NonConformityInfo(BaseModel):
    """Parameter for GetNonConformityInfo"""
    id: str = Field(description="Non Conformity ID")


class GetNonConformityInfo(BaseTool):
    """Gathering customer information"""

    name = "GetNonConformityInfo"
    description = "Provide the history of the non conformity; given the Non Conformity ID"
    args_schema: Type[BaseModel] = NonConformityInfo(BaseModel):


    def _run(self, id: str):
        dataset = documents_md_table
        table_name = dataset.get_location_info().get('info', {}).get('table')
        executor = SQLExecutor2(dataset=dataset)
        eid = id.replace("'", "\\'")
        query_reader = executor.query_to_iter(
            f"""SELECT name, job, company FROM "{table_name}" WHERE id = '{eid}'""")
        for (name, job, company) in query_reader.iter_tuples():
            return f"The customer's name is \"{name}\", holding the position \"{job}\" at the company named {company}"
        return f"No information can be found about the non conformity {id}"

    def _arun(self, id: str):
        raise NotImplementedError("This tool does not support async") 

tools = [GetNC(), GetCompanyInfo()]
tool_names = [tool.name for tool in tools]

# Initializes the agent
prompt = ChatPromptTemplate.from_template(
    """Answer the following questions as best you can. You have only access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}""")



CONNECTION_AVAILABLE = len(LLM_ID) > 0

if CONNECTION_AVAILABLE:
    llm = DKUChatLLM(llm_id=LLM_ID, temperature=0)
    project = dataiku.api_client().get_default_project()
    kb = project.get_knowledge_bank(KB_ID).as_core_knowledge_bank()
    with folder.get_download_stream('/bm25result.pkl') as stream:
        sparse_retriever = pickle.load(io.BytesIO(stream.read()))
    dense_retriever = kb.as_langchain_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=ensemble_retriever,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(template)},
        return_source_documents=True
    )

ERROR_MESSAGE_MISSING_KEY = """
LLM Connection missing. You need to add it as a project variable. Cf. this project's [wiki](https://gallery.dataiku.com/projects/EX_ADVANCED_RAG/wiki/1/Project%20description).

**Please note that the question answering web app is not live on Dataiku’s public project gallery but you can test it by downloading the project and providing an LLM Connection**.

You can find examples of answers in the `answers` dataset.
"""

# Question answering prompt

def escape_markdown(text):
    return text.replace('\\`', '`').replace('\\_', '_')\
        .replace('\\~', '~').replace('\\>', '>')\
        .replace('\\[', '[').replace('\\]', ']')\
        .replace('\\(', '(').replace('\\)', ')')\
        .replace('`', '\\`').replace('_', '\\_')\
        .replace('~', '\\~').replace('>', '\\>').replace('[', '\\[')\
        .replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')

def format_sources(sources):
    result = "\n\n**Sources**:"
    for d in sources:
        chunk = escape_markdown(d.page_content)
        if "url" in d.metadata:
            splitted = chunk.split("\n")
            chunk = f"[{splitted[0]}]({d.metadata['url']})" + "\n" + "\n".join(splitted[1:])
        result = result + "\n\n" + chunk
    return result
        
@functools.lru_cache()
def get_answer(query):
    """
    Provide the LLM with the query and chunks extracted from the source documents and get an answer.
    """
    result = qa_chain({"query": query})
    return escape_markdown(result['result']) + format_sources(result['source_documents'])

# Layout

STYLE_ANSWER = {
    "margin-top": "20px",
    "align-items": "flex-start",
    "display": "flex",
    "height": "auto"
}

STYLE_BUTTON = {
    "width": "20px",  
    "text-align": "center",
    "margin": "0px 5px",
}

STYLE_FEEDBACK = {
    "margin-left": "10px",
    "display": "none"
}

ok_icon_fill = html.Span(html.I(className="bi bi-emoji-smile-fill"))
nok_icon_fill = html.Span(html.I(className="bi bi-emoji-frown-fill"))
ok_icon = html.Span(html.I(className="bi bi-emoji-smile"))
nok_icon = html.Span(html.I(className="bi bi-emoji-frown"))

send_icon = html.Span(html.I(className="bi bi-send"))
question_bar = dbc.InputGroup(
    [
        dbc.Input(id='query', value='', type='text', minLength=0),
        dbc.Button(send_icon, id='send-btn', title='Get an answer')
    ],
    style = {"margin-top": "20px"}
)

app.title = "Question answering"
app.config.external_stylesheets = [
    dbc.themes.ZEPHYR,
    dbc.icons.BOOTSTRAP
]
app.layout = html.Div(
    [
        html.H4(
            "Question answering over documents",
            style={"margin-top": "20px", "text-align": "center"}
        ),
        question_bar,   
        html.Div(
            [
                dbc.Spinner(
                    dcc.Markdown(
                        id='answer',
                        link_target="_blank",
                        style={"min-width": "100px"}
                    ),
                    color="primary"
                ),
                html.Div(
                    [
                        html.A(ok_icon, id="link_ok", href="#", style=STYLE_BUTTON),
                        html.A(nok_icon, id="link_nok", href="#", style=STYLE_BUTTON)   
                    ],
                    id="feedback_buttons",
                    style=STYLE_FEEDBACK)
            ],
            style=STYLE_ANSWER
        ),
        html.Div(id='debug'),
        dcc.Store(id='feedback', data=2, storage_type='memory'),
        dcc.Store(id='question', storage_type='memory'),
        dcc.Store(id='question_id', storage_type='memory'),
    ],
    style={
        "margin": "auto",
        "text-align": "left",
        "max-width": "800px"
    }
)

# Callbacks

@app.callback(
    Output('answer', 'children'),
    Output('feedback_buttons', 'style'),
    Output('question', 'data'),
    Output('question_id', 'data'),
    Input('send-btn', 'n_clicks'),
    Input('query', 'n_submit'),
    State('query', 'value'),
)
def answer_question(n_clicks, n_submit, query):
    """
    Display the answer
    """
    if len(query) == 0:
        return "", STYLE_FEEDBACK, query, 0
    start = time.time()
    if len(LLM_ID) == 0 or not CONNECTION_AVAILABLE:
        return ERROR_MESSAGE_MISSING_KEY, STYLE_ANSWER, query, 0
    style = dict(STYLE_FEEDBACK)
    style["display"] = "flex"
    answer = get_answer(query)
    answer = f"{answer}\n\n{(time.time()-start):.1f} seconds"
    return answer, style, query, hash(str(start) + query)

@app.callback(
    Output('debug', 'children'),
    Input('answer', 'children'),
    Input('link_ok', 'children'),
    Input('link_nok', 'children'),
    State('query', 'value'),
    State('question_id', 'data'),
    State('feedback', 'data'),
)
def log_answer(answer, ok, nok, query, question_id, feedback):
    """
    Log the question and the answer
    """
    if len(LLM_ID) > 0 and len(answer) > 0:
        path = f"/{str(question_id)}.json"
        if LOG_ALL_ANSWERS or feedback in [1, -1]:
            with answers_folder.get_writer(path) as w:
                w.write(bytes(json.dumps({
                    "question": query,
                    "answer": answer,
                    "feedback": 0 if feedback == 2 else feedback,
                    "timestamp": str(datetime.now()),
                    "webapp": WEBAPP_NAME,
                    "version": VERSION
                }), "utf-8"))
        else:
            if path in answers_folder.list_paths_in_partition():
                answers_folder.delete_path(path)
    return ""

@app.callback(
    Output('link_ok', 'children'),
    Output('link_nok', 'children'),
    Input('feedback', 'data')
)
def update_icons(value):
    """
    Update the feedback icons when the user likes or dislikes an answer
    """
    ok = ok_icon_fill if value is not None and value == 1 else ok_icon
    nok = nok_icon_fill if value is not None and value == -1 else nok_icon
    return ok, nok

@app.callback(
    Output('feedback', 'data'),
    Input('link_ok', 'n_clicks'),
    Input('link_nok', 'n_clicks'),
    Input('send-btn', 'n_clicks'),
    Input('query', 'n_submit'),
    State('feedback', 'data'),
    State('query', 'value'),
    State('question', 'data'),
)
def provide_feedback(ts_ok, ts_nok, click, submit, value, question, previous_question):
    """
    Record the feedback of the user
    """
    triggered = dash.ctx.triggered_id
    if triggered == "link_ok":
        return 1 if value != 1 else 0
    elif triggered == "link_nok":
        return -1 if value != -1 else 0
    return value if question == previous_question else 2
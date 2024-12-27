# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import io
import pandas as pd
from bs4 import BeautifulSoup
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

folder = dataiku.Folder("AXB1Cyno")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
headers = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5")
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
d = {"chunk": [], "doc": []}

for path in folder.list_paths_in_partition():
    doc = path[1:].replace("=", "/")
    with folder.get_download_stream(path) as stream:
        s = io.BytesIO(stream.read()).read().decode("utf-8")
        result = text_splitter.split_documents(md_splitter.split_text(s))

    for i in range(len(result)):
        header = " > ".join(
            [
                result[i].metadata[k[1]].replace("#", "").strip()
                for k in headers
                if k[1] in result[i].metadata
                if result[i].metadata[k[1]] is not None
            ]
        )

        d["chunk"].append((header + "\n\n" + result[i].page_content).strip())
        d["doc"].append(doc)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = pd.DataFrame.from_dict(d)
df["chunk_id"] = range(len(df))
dataiku.Dataset("a220_tech_docs_content").write_with_schema(df)
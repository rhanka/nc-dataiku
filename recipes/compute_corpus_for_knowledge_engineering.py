# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Import libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Get texts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()
# Assuming the folder contains text files, we can read them into a DataFrame

file_paths = A220_tech_docs_text.list_paths_in_partition()
texts = []

for file_path in file_paths:
    with A220_tech_docs_text.get_download_stream(file_path) as f:
        texts.append(f.read().decode('utf-8'))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Clean texts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clean_texts = texts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.
clean_texts = [text.replace("\n\n", "\\pp") for text in clean_texts]
clean_texts = [text.replace("\n", " ") for text in clean_texts]
clean_texts = [text.replace("\\pp", "\n\n") for text in clean_texts]
clean_texts = [text.replace("\x0c", "\n") for text in clean_texts]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Save dataframe

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_for_knowledge_engineering_df = pd.DataFrame({"text": clean_texts})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_for_knowledge_engineering_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_for_knowledge_engineering_df["text"][2]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Write recipe outputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_for_knowledge_engineering = dataiku.Dataset("corpus_for_knowledge_engineering")
corpus_for_knowledge_engineering.write_with_schema(corpus_for_knowledge_engineering_df)
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from dataiku import pandasutils as pdu
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()

# Assuming the text data is stored in files within the folder
file_paths = A220_tech_docs_text.list_paths_in_partition()

# Read the content of each file into a list
documents = []
for file_path in file_paths:
    with A220_tech_docs_text.get_download_stream(file_path) as f:
        documents.append(f.read().decode('utf-8'))

# Compute recipe outputs
# Create a CountVectorizer to convert the text data to a term-document matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Convert the sparse matrix to a DataFrame
TDM_term_document_matrix_df = pd.DataFrame(X.toarray().T, index=vectorizer.get_feature_names_out(), columns=file_paths)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
TDM_term_document_matrix = dataiku.Dataset("TDM_term_document_matrix")
TDM_term_document_matrix.write_with_schema(TDM_term_document_matrix_df.reset_index().rename(columns={'index': 'term'}))

# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()

# Assuming the folder contains text files, we can read them into a DataFrame
import os

file_paths = A220_tech_docs_text.list_paths_in_partition()
texts = []

for file_path in file_paths:
    with A220_tech_docs_text.get_download_stream(file_path) as f:
        texts.append(f.read().decode('utf-8'))

# Create a DataFrame from the texts
sentences_df = pd.DataFrame({'text': texts})

# Write recipe outputs
sentences = dataiku.Dataset("sentences")
sentences.write_with_schema(sentences_df)

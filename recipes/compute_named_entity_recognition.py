# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()

# Assuming the folder contains text files, read them into a DataFrame
file_paths = A220_tech_docs_text.list_paths_in_partition()
texts = []
for file_path in file_paths:
    with A220_tech_docs_text.get_download_stream(file_path) as f:
        texts.append(f.read().decode('utf-8'))

# Create a DataFrame
df = pd.DataFrame({'text': texts})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Perform named entity recognition
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df['entities'] = df['text'].apply(extract_entities)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Convert the DataFrame to the required format
named_entity_recognition_df = df.explode('entities').dropna().reset_index(drop=True)
named_entity_recognition_df[['entity', 'label']] = pd.DataFrame(named_entity_recognition_df['entities'].tolist(), index=named_entity_recognition_df.index)
named_entity_recognition_df = named_entity_recognition_df.drop(columns=['entities'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
named_entity_recognition = dataiku.Dataset("named_entity_recognition")
named_entity_recognition.write_with_schema(named_entity_recognition_df)
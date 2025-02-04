# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

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

# Perform named entity recognition
def extract_entities(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags)
    entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entity = " ".join([token for token, pos in chunk.leaves()])
            entity_label = chunk.label()
            entities.append((entity, entity_label))
    return entities

df['entities'] = df['text'].apply(extract_entities)

# Convert the DataFrame to the required format
named_entity_recognition_df = df.explode('entities').dropna().reset_index(drop=True)
named_entity_recognition_df[['entity', 'label']] = pd.DataFrame(named_entity_recognition_df['entities'].tolist(), index=named_entity_recognition_df.index)
named_entity_recognition_df = named_entity_recognition_df.drop(columns=['entities'])

# Write recipe outputs
named_entity_recognition = dataiku.Dataset("named_entity_recognition_with_nltk")
named_entity_recognition.write_with_schema(named_entity_recognition_df)

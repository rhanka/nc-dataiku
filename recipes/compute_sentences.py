# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from dataiku import pandasutils as pdu
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

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

# Extract sentences from the texts
sentences = []
for text in texts:
    text = text.replace("\x0c", "\n")
    text = text.strip()
    sentences.extend(sent_tokenize(text))

# Create a DataFrame from the sentences
sentences_df = pd.DataFrame({'sentence': sentences})

# Splitting sentences by newline character
sentences_df['sentence'] = sentences_df['sentence'].apply(lambda x: x.split('\n'))
# Stripping whitespace from each sentence
exploded_df = sentences_df.explode('sentence')
exploded_df = exploded_df.drop_duplicates()
exploded_df['sentence'] = exploded_df['sentence'].apply(lambda x: x.strip() if isinstance(x, str) else x)
exploded_df['sentence'].replace('', np.nan, inplace=True)
exploded_df = exploded_df.dropna().reset_index(drop=True)
clean_sentences_according_to_nltk_df = exploded_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clean_sentences_according_to_nltk_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import spacy
import pandas as pd

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to refine sentences and add POS tags
def refine_sentences_with_pos(df, column_name):
    refined_sentences = []
    pos_sentences = []
    for sentence in df[column_name]:
        doc = nlp(sentence)
        refined_sentences.extend([sent.text for sent in doc.sents])
        pos_sentences.extend([' '.join([token.pos_ for token in sent]) for sent in doc.sents])
    return pd.DataFrame({column_name: refined_sentences, 'POS_sentence': pos_sentences})

# Refine the sentences and add POS tags to the DataFrame
refined_df = refine_sentences_with_pos(clean_sentences_according_to_nltk_df, 'sentence')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
refined_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
valid_sentences_df = refined_df[refined_df["POS_sentence"].contains("VERB")]
valid_sentences_df = refined_df[refined_df["POS_sentence"].endswith("PUNCT")]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
sentences = dataiku.Dataset("sentences")
sentences.write_with_schema(refined_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write in chunks to avoid memory overload
with sentences.get_writer() as writer:
    chunk_size = 10000  # Adjust based on your memory capacity
    for chunk in np.array_split(refined_df, max(1, len(refined_df) // chunk_size)):
        writer.write_dataframe(chunk)
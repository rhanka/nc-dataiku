# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Import libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from dataiku import pandasutils as pdu
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import spacy
# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
import os

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Read recipe inputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Get corpus

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
# ## Prepare texts for sentence extraction

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clean_texts = texts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clean_texts = [text.replace("\n\n", "\\pp") for text in clean_texts]
clean_texts = [text.replace("\n", " ") for text in clean_texts]
clean_texts = [text.replace("\\pp", "\n\n") for text in clean_texts]
clean_texts = [text.replace("\x0c", "\n") for text in clean_texts]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Extract sentences from the texts with nltk

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sentences = []
for text in clean_texts:
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Extract sentences from the texts with spacy

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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
valid_sentences_df = refined_df[refined_df["POS_sentence"].str.contains("VERB")]
valid_sentences_df = valid_sentences_df[valid_sentences_df["POS_sentence"].str.endswith("PUNCT")]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
valid_sentences_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
sentences = dataiku.Dataset("sentences")
sentences.write_with_schema(valid_sentences_df)
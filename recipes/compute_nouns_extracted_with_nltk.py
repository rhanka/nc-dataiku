# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Import libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import nltk
from nltk import word_tokenize, pos_tag

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Read recipe inputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_for_knowledge_engineering = dataiku.Dataset("corpus_for_knowledge_engineering")
corpus_for_knowledge_engineering_df = corpus_for_knowledge_engineering.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_for_knowledge_engineering_df["lower_text"] = corpus_for_knowledge_engineering_df["text"].str.lower()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to extract nouns from text
def extract_nouns(text):
    words = word_tokenize(text)
    words_pos = pos_tag(words)
    #nouns = [word for word, pos in words_pos if pos.startswith('NN')]
    # ' '.join(nouns)
    return words_pos

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
data = extract_nouns(text)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = pd.DataFrame(data, columns=['word', 'tag'])
tags = ["NN", "NNS", "NNP"]
df = df[df["tag"].isin(tags)]
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pos_df = corpus_for_knowledge_engineering_df['text'].apply(extract_nouns)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pos_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lower_pos_df = corpus_for_knowledge_engineering_df['lower_text'].apply(extract_nouns)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply the function to the dataframe
corpus_for_knowledge_engineering_df['nouns'] = corpus_for_knowledge_engineering_df['text'].apply(extract_nouns)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
nouns_extracted_with_nltk_df = corpus_for_knowledge_engineering_df[['nouns']]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_extracted_with_nltk_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
nouns_extracted_with_nltk = dataiku.Dataset("nouns_extracted_with_nltk")
nouns_extracted_with_nltk.write_with_schema(nouns_extracted_with_nltk_df)
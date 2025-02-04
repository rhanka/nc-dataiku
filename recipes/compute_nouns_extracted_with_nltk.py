# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import nltk
from nltk import word_tokenize, pos_tag

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Read recipe inputs
corpus_for_knowledge_engineering = dataiku.Dataset("corpus_for_knowledge_engineering")
corpus_for_knowledge_engineering_df = corpus_for_knowledge_engineering.get_dataframe()

# Function to extract nouns from text
def extract_nouns(text):
    words = word_tokenize(text)
    words_pos = pos_tag(words)
    nouns = [word for word, pos in words_pos if pos.startswith('NN')]
    return ' '.join(nouns)

# Apply the function to the dataframe
corpus_for_knowledge_engineering_df['nouns'] = corpus_for_knowledge_engineering_df['text_column'].apply(extract_nouns)

# Compute recipe outputs from inputs
nouns_extracted_with_nltk_df = corpus_for_knowledge_engineering_df[['nouns']]

# Write recipe outputs
nouns_extracted_with_nltk = dataiku.Dataset("nouns_extracted_with_nltk")
nouns_extracted_with_nltk.write_with_schema(nouns_extracted_with_nltk_df)

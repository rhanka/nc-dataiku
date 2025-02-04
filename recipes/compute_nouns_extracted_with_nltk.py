# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Import libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import nltk
from nltk import word_tokenize, pos_tag

from nltk.tokenize import word_tokenize

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
corpus_for_knowledge_engineering_df["upper_text"] = corpus_for_knowledge_engineering_df["text"].str.upper()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_for_knowledge_engineering_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Extract nouns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define the function to extract nouns and return a DataFrame
def extract_nouns(doc_id, text):
    words = word_tokenize(text)
    words_pos = pos_tag(words)
    nouns = [(doc_id, word, pos) for word, pos in words_pos if pos.startswith('NN')]
    return pd.DataFrame(nouns, columns=['doc_id', 'word', 'tag'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply the function to each row and concatenate the results
pos_df = pd.concat([extract_nouns(row['doc_id'], row['text']) for _, row in corpus_for_knowledge_engineering_df.iterrows()], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pos_df["string"] = pos_df["word"].str.lower()
pos_df = pos_df.drop_duplicates()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pos_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Lower case

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply the function to each row and concatenate the results
lower_pos_df = pd.concat([extract_nouns(row['doc_id'], row['lower_text']) for _, row in corpus_for_knowledge_engineering_df.iterrows()], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lower_pos_df["string"] = lower_pos_df["word"].str.lower()
lower_pos_df.rename(columns={'tag':'lower_tag'}, inplace=True)
lower_pos_df.drop(columns=['word'], inplace=True)
lower_pos_df = lower_pos_df.drop_duplicates()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lower_pos_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
merged_pos_df = pd.merge(pos_df, lower_pos_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
merged_pos_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Upper case

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply the function to each row and concatenate the results
upper_pos_df = pd.concat([extract_nouns(row['doc_id'], row['upper_text']) for _, row in corpus_for_knowledge_engineering_df.iterrows()], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
upper_pos_df["string"] = upper_pos_df["word"].str.lower()
upper_pos_df.rename(columns={'tag':'upper_tag'}, inplace=True)
upper_pos_df.drop(columns=['word'], inplace=True)
upper_pos_df = upper_pos_df.drop_duplicates()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
merged_pos_df = pd.merge(merged_pos_df, upper_pos_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_extracted_with_nltk_df = merged_pos_df.drop_duplicates()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_extracted_with_nltk_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_frequency_df = nouns_extracted_with_nltk_df["string"].value_counts().rename_axis('string').reset_index(name='frequency')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_frequency_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_frequency_df = nouns_frequency_df[nouns_frequency_df["frequency"] > 1]
nouns_frequency_df.columns = ['string', 'frequency']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate relative frequency
total_count = nouns_frequency_df['frequency'].sum()
nouns_frequency_df['relative_frequency'] = nouns_frequency_df['frequency'] / total_count

# Rank the strings by frequency
nouns_frequency_df['rank'] = nouns_frequency_df['frequency'].rank(method='min', ascending=False)

# Sort by string in ascending order before calculating cumulative sum
nouns_frequency_df = nouns_frequency_df.sort_values(by='frequency', ascending=False)

# Calculate cumulative sum of frequencies
nouns_frequency_df['cumulative_sum'] = nouns_frequency_df['frequency'].cumsum()

# Calculate pareto as cumulative_sum percentage
nouns_frequency_df['pareto'] = (nouns_frequency_df['cumulative_sum'] / total_count) * 100

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_frequency_df[nouns_frequency_df["pareto"] < 80]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Write recipe outputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nouns_extracted_with_nltk = dataiku.Dataset("nouns_extracted_with_nltk")
nouns_extracted_with_nltk.write_with_schema(nouns_frequency_df)
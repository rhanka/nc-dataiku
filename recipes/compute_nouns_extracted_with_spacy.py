# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Import libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import spacy
from dframcy import DframCy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def extract_nouns_with_spacy(text):
    dframcy = DframCy(nlp)
    doc = dframcy.nlp(text)
    pos_df = dframcy.to_dataframe(doc)
    tags = ["NN", "NNP"]
    filtered_pos_df = pos_df[pos_df["token_tag_"].isin(tags)]
    return filtered_pos_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Assuming you have already defined your extract_nouns_with_spacy function
def extract_nouns_with_spacy(text):
    nlp.max_length = 3000000  # Set this to a value higher than your text length
    dframcy = DframCy(nlp)
    doc = dframcy.nlp(text)
    pos_df = dframcy.to_dataframe(doc)
    tags = ["NN", "NNP"]
    filtered_pos_df = pos_df[pos_df["token_tag_"].isin(tags)]
    return filtered_pos_df

# Apply the function to the 'text' column and consolidate the results
nouns_extracted_with_spacy_df = corpus_for_knowledge_engineering_df['text'].apply(extract_nouns_with_spacy)

# If you want to concatenate all the results into a single dataframe
consolidated_results_df = pd.concat(nouns_extracted_with_spacy_df.tolist(), ignore_index=True)

print(consolidated_results_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Extract nouns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define the function to extract nouns and return a DataFrame
def extract_nouns(doc_id, text):
    # Increase the max_length limit
    nlp.max_length = 3000000  # Set this to a value higher than your text length
    doc = nlp(text)
    nouns = [(doc_id, token.text, token.pos_) for token in doc if token.pos_ == "NOUN"]
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
# Write recipe outputs
nouns_extracted_with_spacy = dataiku.Dataset("nouns_extracted_with_spacy")
nouns_extracted_with_spacy.write_with_schema(nouns_extracted_with_spacy_df)
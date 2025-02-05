# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
corpus_for_knowledge_engineering = dataiku.Dataset("corpus_for_knowledge_engineering")
corpus_for_knowledge_engineering_df = corpus_for_knowledge_engineering.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

nouns_extracted_with_spacy_df = corpus_for_knowledge_engineering_df # For this sample code, simply copy input to output


# Write recipe outputs
nouns_extracted_with_spacy = dataiku.Dataset("nouns_extracted_with_spacy")
nouns_extracted_with_spacy.write_with_schema(nouns_extracted_with_spacy_df)



# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Import libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import spacy

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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Extract nouns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define the function to extract nouns and return a DataFrame
def extract_nouns(doc_id, text):
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

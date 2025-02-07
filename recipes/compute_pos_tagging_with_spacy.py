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


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

pos_tagging_with_spacy_df = corpus_for_knowledge_engineering_df # For this sample code, simply copy input to output


# Write recipe outputs
pos_tagging_with_spacy = dataiku.Dataset("pos_tagging_with_spacy")
pos_tagging_with_spacy.write_with_schema(pos_tagging_with_spacy_df)

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

noun_phrases_extracted_with_spacy_df = corpus_for_knowledge_engineering_df # For this sample code, simply copy input to output


# Write recipe outputs
noun_phrases_extracted_with_spacy = dataiku.Dataset("noun_phrases_extracted_with_spacy")
noun_phrases_extracted_with_spacy.write_with_schema(noun_phrases_extracted_with_spacy_df)

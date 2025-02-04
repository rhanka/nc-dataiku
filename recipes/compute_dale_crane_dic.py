# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Import libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import re

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Get data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
dictionaries_of_aviation = dataiku.Dataset("dictionaries_of_aviation")
dictionaries_of_aviation_df = dictionaries_of_aviation.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dictionaries_of_aviation_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dictionary = dictionaries_of_aviation_df["content"][2]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dictionary

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Clean data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clean_dictionary = dictionary
clean_dictionary = clean_dictionary.replace("-\n","")
clean_dictionary = clean_dictionary.replace(" \n"," ")

#clean_dictionary = re.sub(r'\s*\n\s*', ' ', clean_dictionary)
clean_dictionary = re.sub(r'([a-zA-Z]|[0-9])\n([a-zA-Z]|[0-9])', r'\1 \2', clean_dictionary)
clean_dictionary = re.sub(" \x02 ", ' ', clean_dictionary)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clean_dictionary

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Entries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
entries = re.findall(r'\b(\w+)\b\s+\b\1\b', clean_dictionary)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
entries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pattern = r'\b(\w+)\b\s+\b\1\b /'
replacement = r'[[[\1]]] /'
dictionary_with_entries = re.sub(pattern, replacement, clean_dictionary)
dictionary_with_entries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Split the text using triple brackets
entries = re.split(r'\[\[\[|\]\]\]', dictionary_with_entries)

# Remove empty entries and strip whitespace
entries = [entry.strip() for entry in entries if entry.strip()]

# Ensure both columns have the same length by adding empty strings if necessary
if len(entries) % 2 != 0:
    entries.append('')

# Create a DataFrame with two columns: 'Entry' and 'Definition'
data = {'Entry': entries[0::2], 'Definition': entries[1::2]}
df = pd.DataFrame(data)
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dale_crane_dic_df = pd.DataFrame({"entry":list(df["Definition"][0:1870]),"definition":list(df["Entry"][1:1871])})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
dale_crane_dic = dataiku.Dataset("dale_crane_dic")
dale_crane_dic.write_with_schema(dale_crane_dic_df)
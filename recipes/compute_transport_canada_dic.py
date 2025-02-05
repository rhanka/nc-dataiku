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






# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

transport_canada_dic_df = dictionaries_of_aviation_df # For this sample code, simply copy input to output


# Write recipe outputs
transport_canada_dic = dataiku.Dataset("transport_canada_dic")
transport_canada_dic.write_with_schema(transport_canada_dic_df)

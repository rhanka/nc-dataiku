# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

english_wiktionary_titles_df = ... # Compute a Pandas dataframe to write into english_wiktionary_titles


# Write recipe outputs
english_wiktionary_titles = dataiku.Dataset("english_wiktionary_titles")
english_wiktionary_titles.write_with_schema(english_wiktionary_titles_df)

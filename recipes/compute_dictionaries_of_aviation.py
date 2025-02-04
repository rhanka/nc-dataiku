# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
Dictionaries = dataiku.Folder("PP5gsWW1")
Dictionaries_info = Dictionaries.get_info()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

dictionaries_of_aviation_df = ... # Compute a Pandas dataframe to write into dictionaries_of_aviation


# Write recipe outputs
dictionaries_of_aviation = dataiku.Dataset("dictionaries_of_aviation")
dictionaries_of_aviation.write_with_schema(dictionaries_of_aviation_df)

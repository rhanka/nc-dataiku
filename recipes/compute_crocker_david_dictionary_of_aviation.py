# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

crocker_david_dictionary_of_aviation_df = ... # Compute a Pandas dataframe to write into crocker_david_dictionary_of_aviation


# Write recipe outputs
crocker_david_dictionary_of_aviation = dataiku.Dataset("crocker_david_dictionary_of_aviation")
crocker_david_dictionary_of_aviation.write_with_schema(crocker_david_dictionary_of_aviation_df)

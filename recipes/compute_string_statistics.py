# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

string_statistics_df = ... # Compute a Pandas dataframe to write into string_statistics


# Write recipe outputs
string_statistics = dataiku.Dataset("string_statistics")
string_statistics.write_with_schema(string_statistics_df)

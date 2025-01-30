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

document_statistics_df = ... # Compute a Pandas dataframe to write into document_statistics


# Write recipe outputs
document_statistics = dataiku.Dataset("document_statistics")
document_statistics.write_with_schema(document_statistics_df)

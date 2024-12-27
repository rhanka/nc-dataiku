# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
A220_tech_docs_prep = dataiku.Folder("AXB1Cyno")
A220_tech_docs_prep_info = A220_tech_docs_prep.get_info()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

a220_tech_docs_content_df = ... # Compute a Pandas dataframe to write into a220_tech_docs_content


# Write recipe outputs
a220_tech_docs_content = dataiku.Dataset("a220_tech_docs_content")
a220_tech_docs_content.write_with_schema(a220_tech_docs_content_df)

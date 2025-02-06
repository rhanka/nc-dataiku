# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

test_library_df = ... # Compute a Pandas dataframe to write into test_library


# Write recipe outputs
test_library = dataiku.Dataset("test_library")
test_library.write_with_schema(test_library_df)

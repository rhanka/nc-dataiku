# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
dictionaries_of_aviation = dataiku.Dataset("dictionaries_of_aviation")
dictionaries_of_aviation_df = dictionaries_of_aviation.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

dale_crane_dic_df = dictionaries_of_aviation_df # For this sample code, simply copy input to output


# Write recipe outputs
dale_crane_dic = dataiku.Dataset("dale_crane_dic")
dale_crane_dic.write_with_schema(dale_crane_dic_df)

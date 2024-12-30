# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NC_types_random_500_md_concat = dataiku.Dataset("NC_types_random_500_md_concat")
NC_types_random_500_md_concat_df = NC_types_random_500_md_concat.get_dataframe()




# Write recipe outputs
NC_types_500_md_files = dataiku.Folder("gZC3bHFN")
NC_types_500_md_files_info = NC_types_500_md_files.get_info()

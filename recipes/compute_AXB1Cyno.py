# -*- coding: utf-8 -*-
import dataiku
#import pandas as pd, numpy as np
#from dataiku import pandasutils as pdu
from markitdown import markitdown



# Read recipe inputs
A220_tech_docs = dataiku.Folder("SoQWOnhR")
A220_tech_docs_info = A220_tech_docs.get_info()




# Write recipe outputs
A220_tech_docs_prep = dataiku.Folder("AXB1Cyno")
A220_tech_docs_prep_info = A220_tech_docs_prep.get_info()

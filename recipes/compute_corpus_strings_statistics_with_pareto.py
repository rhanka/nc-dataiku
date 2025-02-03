# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import numpy as np
from dataiku import pandasutils as pdu
import re
import string


# Read recipe inputs
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to read the corpus from the folder and extract strings
def read_corpus(folder):
    corpus = []
    for file_path in folder.list_paths_in_partition():
        with folder.get_download_stream(file_path) as f:
            text = f.read().decode('utf-8')
            # Remove punctuation and convert to lowercase before splitting
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            words = text.split()
            # Filter out any words containing numbers
            words = [word for word in words if not re.search(r'\d', word)]
            corpus.extend(words)
    return corpus

corpus = read_corpus(A220_tech_docs_text)

# Compute frequency of each string in the corpus
frequency = pd.Series(corpus).value_counts().reset_index()
frequency.columns = ['string', 'frequency']

# Calculate relative frequency
total_count = frequency['frequency'].sum()
frequency['relative_frequency'] = frequency['frequency'] / total_count

# Rank the strings by frequency
frequency['rank'] = frequency['frequency'].rank(method='min', ascending=False)

# Sort by string in ascending order before calculating cumulative sum
frequency = frequency.sort_values(by='frequency', ascending=False)

# Calculate cumulative sum of frequencies
frequency['cumulative_sum'] = frequency['frequency'].cumsum()

# Calculate pareto as cumulative_sum percentage
frequency['pareto'] = (frequency['cumulative_sum'] / total_count) * 100

# Compute recipe outputs
corpus_strings_statistics_with_pareto_df = frequency

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_strings_statistics_with_pareto_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
corpus_strings_statistics_with_pareto = dataiku.Dataset("corpus_strings_statistics_with_pareto")
corpus_strings_statistics_with_pareto.write_with_schema(corpus_strings_statistics_with_pareto_df)
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
from collections import Counter
import string

# Read recipe inputs
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to split text into strings, remove punctuation, convert to lowercase, and compute occurrences
def compute_string_frequencies(text, document_id):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    word_counts = Counter(words)
    data = [{"document_id": document_id, "string": word, "frequency": count} for word, count in word_counts.items()]
    return data

# Initialize an empty list to store the data
data = []

# Get the list of file paths in the folder
file_paths = A220_tech_docs_text.list_paths_in_partition()

# Iterate over each file path
for document_id, file_path in enumerate(file_paths, start=1):
    # Read the content of the file
    with A220_tech_docs_text.get_download_stream(file_path) as f:
        text = f.read().decode('utf-8')
        # Compute string frequencies
        file_data = compute_string_frequencies(text, document_id)
        # Append the data to the list
        data.extend(file_data)

# Convert the data to a DataFrame
df_string_stats = pd.DataFrame(data)

# Display the DataFrame
print(df_string_stats)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
string_statistics = dataiku.Dataset("strings_statistics")
string_statistics.write_with_schema(df_string_stats)
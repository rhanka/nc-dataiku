# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
import re
import textstat
from dataiku import pandasutils as pdu
from collections import Counter

# Read recipe inputs
A220_tech_docs_text = dataiku.Folder("rhnW9xGx")
A220_tech_docs_text_info = A220_tech_docs_text.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to compute text statistics
def compute_text_stats(text):
    num_chars = len(text)
    num_words = len(text.split())
    num_sentences = textstat.sentence_count(text)
    num_paragraphs = text.count('\n') + 1
    
    num_uppercase = sum(1 for c in text if c.isupper())
    num_lowercase = sum(1 for c in text if c.islower())
    num_digits = sum(1 for c in text if c.isdigit())
    num_punctuation = sum(1 for c in text if c in '.,;:!?')
    
    avg_word_length = np.mean([len(word) for word in text.split()]) if num_words > 0 else 0
    lexical_diversity = len(set(text.split())) / num_words if num_words > 0 else 0
    readability = textstat.flesch_kincaid_grade(text)
    
    words = text.split()
    unique_words = set(words)
    num_unique_words = len(unique_words)
    longest_word = max(words, key=len) if words else ""
    shortest_word = min(words, key=len) if words else ""
    most_common_word = Counter(words).most_common(1)[0][0] if words else ""

    return pd.Series({
        "num_chars": num_chars,
        "num_words": num_words,
        "num_sentences": num_sentences,
        "num_paragraphs": num_paragraphs,
        
        "num_uppercase": num_uppercase,
        "num_lowercase": num_lowercase,
        "num_digits": num_digits,
        "num_punctuation": num_punctuation,
        
        "avg_word_length": avg_word_length,
        "lexical_diversity": lexical_diversity,
        "readability": readability,
        
        "num_unique_words": num_unique_words,
        "longest_word": longest_word,
        "shortest_word": shortest_word,
        "most_common_word": most_common_word
    })

# Load text data from the folder
file_paths = A220_tech_docs_text.list_paths_in_partition()
data = []
for file_path in file_paths:
    with A220_tech_docs_text.get_download_stream(file_path) as f:
        text = f.read().decode('utf-8')
        stats = compute_text_stats(text)
        stats["file_path"] = file_path
        data.append(stats)

# Convert to DataFrame
df_stats = pd.DataFrame(data)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
document_statistics = dataiku.Dataset("document_statistics")
document_statistics.write_with_schema(df_stats)
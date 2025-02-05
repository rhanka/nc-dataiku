# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from collections import Counter

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
corpus_for_knowledge_engineering = dataiku.Dataset("corpus_for_knowledge_engineering")
corpus_for_knowledge_engineering_df = corpus_for_knowledge_engineering.get_dataframe()

dale_crane_dic = dataiku.Dataset("dale_crane_dic")
dale_crane_dic_df = dale_crane_dic.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus = corpus_for_knowledge_engineering_df["text"]
doc_ids = corpus_for_knowledge_engineering_df["doc_id"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
vocabulary = dale_crane_dic_df["entry"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def compute_statistics(corpus, vocabulary):
    # Tokenize the corpus into words
    words = corpus.split()

    # Count the occurrences of each word in the corpus
    word_counts = Counter(words)

    # Filter the counts to include only the words in the vocabulary
    vocab_counts = {word: word_counts[word] for word in vocabulary if word in word_counts}

    return vocab_counts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
statistics_list = []
for doc_id, text in zip(doc_ids, corpus):
    vocab_counts = compute_statistics(text, vocabulary)
    for entry, count in vocab_counts.items():
        statistics_list.append({'doc_id': doc_id, 'entry': entry, 'count': count})

corpus_statistics_from_voc_dale_crane_df = pd.DataFrame(statistics_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
corpus_statistics_from_voc_dale_crane_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
corpus_statistics_from_voc_dale_crane = dataiku.Dataset("corpus_statistics_from_voc_dale_crane")
corpus_statistics_from_voc_dale_crane.write_with_schema(corpus_statistics_from_voc_dale_crane_df)
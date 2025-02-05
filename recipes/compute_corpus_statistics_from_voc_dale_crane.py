# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from collections import Counter


# Read recipe inputs
corpus_for_knowledge_engineering = dataiku.Dataset("corpus_for_knowledge_engineering")
corpus_for_knowledge_engineering_df = corpus_for_knowledge_engineering.get_dataframe()
dale_crane_dic = dataiku.Dataset("dale_crane_dic")
dale_crane_dic_df = dale_crane_dic.get_dataframe()

vocabulary = dale_crane_dic_df["entry"]

def compute_statistics(corpus, vocabulary):
    # Tokenize the corpus into words
    words = corpus.split()
    
    # Count the occurrences of each word in the corpus
    word_counts = Counter(words)
    
    # Filter the counts to include only the words in the vocabulary
    vocab_counts = {word: word_counts[word] for word in vocabulary if word in word_counts}
    
    # Compute statistics
    total_words = len(words)
    vocab_word_count = sum(vocab_counts.values())
    vocab_word_percentage = (vocab_word_count / total_words) * 100 if total_words > 0 else 0
    
    return {
        'total_words': total_words,
        'vocab_word_count': vocab_word_count,
        'vocab_word_percentage': vocab_word_percentage,
        'vocab_counts': vocab_counts
    }



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

corpus_statistics_from_voc_dale_crane_df = compute_statistics(corpus, vocabulary)


# Write recipe outputs
corpus_statistics_from_voc_dale_crane = dataiku.Dataset("corpus_statistics_from_voc_dale_crane")
corpus_statistics_from_voc_dale_crane.write_with_schema(corpus_statistics_from_voc_dale_crane_df)

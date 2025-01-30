import dataiku
import pandas as pd
import requests
import gzip
import shutil
import os

# URL of the file
url = "https://dumps.wikimedia.org/frwiktionary/latest/frwiktionary-latest-all-titles.gz"
filename = "frwiktionary-latest-all-titles.gz"
unzipped_filename = "frwiktionary-latest-all-titles.txt"

def download_file(url, filename):
    """Download a file from a URL"""
    response = requests.get(url, stream=True)
    with open(filename, "wb") as file:
        shutil.copyfileobj(response.raw, file)
    print(f"Downloaded {filename}")

def extract_gzip(gz_filename, extracted_filename):
    """Extract a gzip file"""
    with gzip.open(gz_filename, "rb") as f_in:
        with open(extracted_filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted {gz_filename} to {extracted_filename}")

def load_to_dataframe(filename):
    """Load file content into a Pandas DataFrame"""
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
    df = pd.DataFrame(lines, columns=["Title"])
    return df

# Download, extract, and load data
download_file(url, filename)
extract_gzip(filename, unzipped_filename)
df = load_to_dataframe(unzipped_filename)

# Write DataFrame to Dataiku dataset
output_dataset = dataiku.Dataset("french_wiktionary_titles")
output_dataset.write_with_schema(df)
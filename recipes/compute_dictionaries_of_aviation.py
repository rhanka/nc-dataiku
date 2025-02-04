# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import fitz  # PyMuPDF
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
Dictionaries = dataiku.Folder("PP5gsWW1")
Dictionaries_info = Dictionaries.get_info()

# Initialize an empty list to store data
data = []

# Iterate through the files in the folder
for file_path in Dictionaries.list_paths_in_partition():
    with Dictionaries.get_download_stream(file_path) as f:
        # Open the PDF file
        pdf_document = fitz.open(stream=f.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        data.append({"file_path": file_path, "content": text})

# Convert the list to a dataframe
dictionaries_of_aviation_df = pd.DataFrame(data)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dictionaries_of_aviation_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
dictionaries_of_aviation = dataiku.Dataset("dictionaries_of_aviation")
dictionaries_of_aviation.write_with_schema(dictionaries_of_aviation_df)
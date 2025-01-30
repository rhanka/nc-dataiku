import requests
import dataiku
import pandas as pd

def get_category_members(category, cmcontinue=None):
    """
    Retrieve all members of a given Wiktionary category using the MediaWiki API.
    """
    url = "https://en.wiktionary.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "500",
        "format": "json"
    }
    
    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    response = requests.get(url, params=params)
    data = response.json()
    
    return data.get("query", {}).get("categorymembers", []), data.get("continue", {}).get("cmcontinue")

def get_all_category_members(category):
    """
    Retrieve all pages belonging to a category by handling pagination.
    """
    all_members = []
    cmcontinue = None
    
    while True:
        members, cmcontinue = get_category_members(category, cmcontinue)
        all_members.extend(members)
        
        if not cmcontinue:
            break
    
    return all_members

# Get all members of the "English_proper_nouns" category
category_name = "English_proper_nouns"
members = get_all_category_members(category_name)

# Convert to DataFrame
df = pd.DataFrame(members)

# Write DataFrame to a Dataiku dataset
output_dataset = dataiku.Dataset("english_proper_nouns_from_english_wiktionary")  # Change to your dataset name
output_dataset.write_with_schema(df)
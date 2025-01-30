from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time
import dataiku

# Define the Dataiku dataset where data will be saved
dataset_name = "wikidata_properties"

# Initialize SPARQL endpoint
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# Set batch size and offset for pagination
batch_size = 5000  # Fetch data in smaller chunks
offset = 0
all_data = []

while True:
    # Define the SPARQL query with pagination
    query = f"""
    SELECT ?property ?propertyType ?propertyLabel ?propertyDescription ?propertyAltLabel WHERE {{
      ?property wikibase:propertyType ?propertyType .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }}
    }}
    LIMIT {batch_size} OFFSET {offset}
    """

    # Configure the SPARQL request
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]

        if not bindings:
            break  # Stop if no more data

        # Process results
        for result in bindings:
            all_data.append({
                "property": result["property"]["value"],
                "propertyType": result["propertyType"]["value"],
                "propertyLabel": result.get("propertyLabel", {}).get("value", ""),
                "propertyDescription": result.get("propertyDescription", {}).get("value", ""),
                "propertyAltLabel": result.get("propertyAltLabel", {}).get("value", ""),
            })

        print(f"Retrieved {len(all_data)} records...")

        # Move to next batch
        offset += batch_size
        time.sleep(1)  # Prevent server overload

    except Exception as e:
        print(f"Error: {e}")
        break

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save DataFrame to a Dataiku dataset
dataiku.Dataset("all_wikidata_properties").write_with_schema(df)
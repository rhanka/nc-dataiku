from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import dataiku

# Define SPARQL endpoint
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# Define the SPARQL query
query = """
SELECT ?property ?propertyType ?propertyLabel ?propertyDescription ?propertyAltLabel WHERE {
  ?property wikibase:propertyType ?propertyType .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }
}
ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'P')))
"""

# Set up the query
sparql.setQuery(query)
sparql.setReturnFormat(JSON)

# Execute the query
results = sparql.query().convert()

# Extract results into a list of dictionaries
data = []
for result in results["results"]["bindings"]:
    data.append({
        "property": result["property"]["value"],
        "propertyType": result["propertyType"]["value"],
        "propertyLabel": result.get("propertyLabel", {}).get("value", ""),
        "propertyDescription": result.get("propertyDescription", {}).get("value", ""),
        "propertyAltLabel": result.get("propertyAltLabel", {}).get("value", ""),
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save DataFrame to Dataiku dataset
dataset_name = "all_wikidata_properties"
dataiku.Dataset(dataset_name).write_with_schema(df)
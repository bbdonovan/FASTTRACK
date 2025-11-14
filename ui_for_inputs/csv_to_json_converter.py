import csv
import json
import os

def load_csv_to_json(filename: str, root_name: str = "Global News Aggregator") -> dict:
    """
    Reads a CSV file containing news articles and converts it into a nested
    hierarchical JSON structure suitable for a D3 Collapsible Tree.

    Hierarchy: Root -> Country Code -> API Source -> Source ID -> Article Titles

    :param filename: The name of the input CSV file.
    :param root_name: The name for the top-level root node in the JSON tree (new parameter).
    :return: A dictionary representing the D3 hierarchical structure.
    """
    if not os.path.exists(filename):
        print(f"Error: Input file '{filename}' not found.")
        return {}

    # Intermediate structure: {country: {api_source: {source_id: [articles]}}}
    intermediate_data = {}
    
    # Required CSV fields
    REQUIRED_FIELDS = ['Title', 'Link', 'Publication Date', 'Source ID', 'Country Code', 'API Source']
    
    print(f"Reading data from {filename} and building hierarchy...")

    try:
        with open(filename, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if required fields exist
            if not all(field in reader.fieldnames for field in REQUIRED_FIELDS):
                print(f"Error: CSV is missing one or more required fields. Found: {reader.fieldnames}")
                return {}

            for row in reader:
                # --- Tier 1 & 2 Keys ---
                country = row.get('Country Code', 'UNKNOWN_COUNTRY').upper()
                api_source = row.get('API Source', 'UNKNOWN_API')
                source_id = row.get('Source ID', 'UNKNOWN_SOURCE')
                
                # --- Tier 4 Leaf Node Data ---
                article_node = {
                    # 'name' is the visible text in the tree
                    "name": row.get('Title', 'Untitled Article'),
                    # These fields are carried along for hyperlinking and display
                    "url": row.get('Link', 'N/A'),
                    "date": row.get('Publication Date', 'N/A'),
                    "snippet": row.get('Snippet', 'N/A')
                }

                # Build the nested structure dynamically
                # Nesting: Country -> API Source -> Source ID -> Articles
                if country not in intermediate_data:
                    intermediate_data[country] = {}
                
                if api_source not in intermediate_data[country]:
                    intermediate_data[country][api_source] = {}
                    
                if source_id not in intermediate_data[country][api_source]:
                    intermediate_data[country][api_source][source_id] = []
                    
                intermediate_data[country][api_source][source_id].append(article_node)

    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return {}

    # --- Convert Intermediate Dictionary to D3 Hierarchy (List of Lists) ---
    hierarchical_output = []
    
    # Tier 1: Country
    for country, api_sources_data in intermediate_data.items():
        country_node = {
            "name": country,
            "children": []
        }
        
        # Tier 2: API Source
        for api_source, sources_data in api_sources_data.items():
            api_source_node = {
                "name": api_source,
                "children": []
            }
            
            # Tier 3: Source ID
            for source_id, articles in sources_data.items():
                source_node = {
                    "name": source_id,
                    "children": []
                }
                
                # Tier 4: Article Titles (Leaf Nodes)
                for article in articles:
                    # Append the article node data
                    source_node['children'].append({
                        "name": article['name'],
                        "url": article['url'],
                        "date": article['date'],
                        "snippet": article['snippet'],
                        "value": 1 # D3 visualization requires a value on leaf nodes
                    })
                
                api_source_node['children'].append(source_node)
            
            country_node['children'].append(api_source_node)

        hierarchical_output.append(country_node)

    # Wrap the entire structure in a root node for D3 compatibility
    final_root_node = {
        # CHANGED: Use the new root_name parameter
        "name": root_name,
        "children": hierarchical_output
    }

    print("Hierarchy successfully built.")
    return final_root_node

def save_json(data: dict, output_filename: str):
    """
    Writes the Python dictionary data to a formatted JSON file.
    """
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Success: Hierarchy written to '{output_filename}'")
    except Exception as e:
        print(f"❌ Error writing JSON file: {e}")


# --- Example Usage (If run directly) ---
if __name__ == '__main__':
    
    # NOTE: You must run news_fetcher.py and newsapi_org_fetcher.py 
    # to create this CSV file before running this script!
    INPUT_CSV = 'political_news_combined.csv'
    OUTPUT_JSON = 'country_news_tree.json'

    # 1. Read CSV and build the hierarchical structure (Now uses the default root name)
    json_data = load_csv_to_json(INPUT_CSV)
    
    # 2. Save the structure to the JSON file
    if json_data:
        save_json(json_data, OUTPUT_JSON)
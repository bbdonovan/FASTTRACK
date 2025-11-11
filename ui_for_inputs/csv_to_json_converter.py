import csv
import json

def load_csv_to_json(filename: str, root_name: str = "News Articles"):
    """
    Reads a CSV file created by fetch_news_to_csv, groups the articles
    by Country Code, then by Source ID, and returns the data in a
    nested JSON hierarchy suitable for visualization (e.g., D3.js).
    
    The resulting structure is: {name: root, children: [{name: Country, children: [{name: Source, children: [articles]}]}]}.

    :param filename: The path/name of the input CSV file.
    :param root_name: The name for the top-level node in the JSON tree.
    :return: A dictionary representing the nested JSON structure.
    """
    
    # Tier 1: Intermediate storage organized as {country: {source: [articles]}}
    data_by_country_and_source = {}
    
    print(f"Reading data from {filename}...")
    
    try:
        with open(filename, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                country = row.get('Country Code')
                source = row.get('Source ID')
                
                # We need valid country and source to group data
                if not country or not source or country == 'N/A' or source == 'N/A':
                    continue
                
                # Tier 3 (Article Details): The leaf node data
                article_node = {
                    "name": row.get('Title', 'Untitled Article'),
                    "url": row.get('Link', 'N/A'),           # *** Updated link field to "url" for D3
                    "pubDate": row.get('Publication Date', 'N/A'),
                    "value": 1  # Placeholder value to match the hierarchy example
                }
                
                # Safely create country level (Tier 1)
                if country not in data_by_country_and_source:
                    data_by_country_and_source[country] = {}
                
                # Safely create source level (Tier 2) and append the article
                if source not in data_by_country_and_source[country]:
                    data_by_country_and_source[country][source] = []
                    
                data_by_country_and_source[country][source].append(article_node)

    except FileNotFoundError:
        print(f"❌ Error: CSV file '{filename}' not found. Please ensure it exists and is correct.")
        return None
    except Exception as e:
        print(f"❌ An error occurred during CSV processing: {e}")
        return None

    # --- Convert to Nested JSON Hierarchy ---
    
    hierarchical_output = []

    # Outer Loop: Country (Tier 2)
    for country, sources in data_by_country_and_source.items():
        country_node = {
            "name": country,
            "children": [] # List of sources
        }
        
        # Middle Loop: Source (Tier 3)
        for source, articles in sources.items():
            source_node = {
                "name": source,
                "children": articles # List of articles (leaves)
            }
            country_node["children"].append(source_node)
            
        hierarchical_output.append(country_node)
        
    # Final Root Node (Tier 1)
    final_json_structure = {
        "name": root_name,
        "children": hierarchical_output
    }
    
    return final_json_structure

def save_json(data, output_filename: str = 'news_data.json'):
    """
    Writes the Python dictionary to a JSON file.
    """
    if data is None:
        return
        
    print(f"\nWriting hierarchical data to {output_filename}...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Use indent=4 for clean, readable output matching the user's example format
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ JSON file saved successfully: {output_filename}")


if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. DEFINE THE INPUT CSV FILE PATH
    # Ensure a file named 'political_news_combined.csv' exists in this directory 
    # and has the columns 'Title', 'Source ID', 'Link', and 'Country Code'.
    INPUT_CSV_FILE = 'political_news_combined.csv'
    OUTPUT_JSON_FILE = 'political_news_tree.json'

    # 2. Load and transform the data
    tree_data = load_csv_to_json(INPUT_CSV_FILE, root_name="Combined Political News")
    
    # 3. Save the result to a JSON file
    save_json(tree_data, OUTPUT_JSON_FILE)
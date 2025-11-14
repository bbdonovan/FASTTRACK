import os
import sys
import json
import time
import requests
import csv
from flask import Flask, render_template, request, send_from_directory
from requests.exceptions import Timeout, RequestException
import newsdata_io as nd_io
import newsapi_org as na_org
from csv_to_json_converter import load_csv_to_json, save_json

# --- Setup and Initialization ---
app = Flask(__name__)

# --- API Key Configuration and Validation ---
API_KEY = os.environ.get("GEMINI_API_KEY", "")
API_KEY = API_KEY.strip().strip('"')
if not API_KEY:
    print("\n" + "="*50)
    print("FATAL ERROR: GEMINI_API_KEY environment variable is not set.")
    print("Please set it in your terminal before running the app.")
    print("Example: export GEMINI_API_KEY='YOUR_API_KEY'")
    print("="*50 + "\n")
    if __name__ == '__main__':
        raise RuntimeError("GEMINI_API_KEY is missing. Cannot start application.")

# --- Constants for News Source Selection ---
NEWS_SOURCES = [
    "All",
    "NewsData.io",
    "NewsApi.org"
]
# --- Constants ---
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
TIMEOUT_SECONDS = 60

# System instruction for generating the main response paragraph
SYSTEM_INSTRUCTION_TEXT = (
    "You are a sophisticated historical researcher. Your task is to write a single, cohesive, "
    "well-written response (5-7 sentences) that synthesizes the user's idea with the specified "
    "countries and time context. Do not include a title or introduction."
)

# System instruction for generating structured news articles (Call 2)
SYSTEM_INSTRUCTION_NEWS = (
    "You are a news aggregator. Based on the user's selected countries and time frame, "
    "generate exactly 5 recent and relevant news articles headlines/titles from reputable news sources. "
    "Output MUST be a JSON array of objects, where each object has a 'title' key."
)


# --- Helper Functions ---

def load_country_data():
    """Loads the country list from the local JSON file."""
    try:
        with open('countries.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: countries.json not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in countries.json.")
        return []

# FIX: Function returns a DICTIONARY in ALL cases to prevent tuple unpacking errors.
def generate_content_with_retry(prompt, system_instruction, response_schema=None, use_search=False):
    """
    Makes a POST request to the Gemini API with exponential backoff.
    Returns generated text and list of sources (if requested).
    """
    url = f"{API_BASE_URL}?key={API_KEY}"
    max_retries = 5
    initial_delay = 1

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }

    headers = {'Content-Type': 'application/json'}

    if response_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    
    if use_search:
        payload["tools"] = [{"google_search": {}}]

    for attempt in range(max_retries):
        try:
            # Use a slightly longer timeout for generation calls
            response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
            
            # Check for non-2xx status codes (400, 500, etc.)
            response.raise_for_status()

            # Process the response
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            # Extract text
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
            
            # Extract sources if search grounding was used
            sources = []
            if use_search:
                grounding_metadata = candidate.get('groundingMetadata')
                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources = [
                        {
                            'uri': attr.get('web', {}).get('uri'),
                            'title': attr.get('web', {}).get('title')
                        }
                        for attr in grounding_metadata['groundingAttributions']
                        if attr.get('web', {}).get('uri') and attr.get('web', {}).get('title')
                    ]
            
            # Log sources for debugging (new feature)
            #try:
            #    # Use os.getcwd() for logging stability
            #    log_path = os.path.join(os.getcwd(), 'gemini_sources_log.json')
            #    with open(log_path, 'w', encoding='utf-8') as log_file:
            #        json.dump(sources, log_file, indent=2)
            #    print(f"Successfully logged {len(sources)} sources to gemini_sources_log.json")
            #except Exception as e:
            #    print(f"Warning: Could not log sources: {e}")

            # SUCCESS: Return a dictionary
            return {"text": text, "sources": sources}
            
        except Timeout:
            print(f"Attempt {attempt + 1} failed: Request timed out.")
        except RequestException as e:
            # Check specifically for 400 Client Error (often an invalid API key)
            print(f"Attempt {attempt + 1} failed: {e}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed due to unexpected error: {e}")

        # Exponential backoff
        if attempt < max_retries - 1:
            delay = initial_delay * (2 ** attempt)
            time.sleep(delay)

    print("Max retries reached. Failing.")
    # FAILURE: Return a dictionary
    return {"text": "Error: Could not generate content after multiple retries.", "sources": []}


# --- Flask Routes ---

@app.route('/country_news_tree.json')
def serve_country_news_tree_json():
    """Route to serve the country_news_tree.json file for the visualization."""
    return send_from_directory(app.root_path, 'country_news_tree.json', mimetype='application/json')

@app.route('/', methods=['GET', 'POST'])
def index():
    country_data = load_country_data()
    
    # Define available time frames (FIXED structure)
    time_frames = [
        {"code": "nl", "name": "No Limit"},
        {"code": "6m", "name": "Last 6 Months"},
        {"code": "1y", "name": "Last Year"},
        {"code": "3y", "name": "Last 3 Years"},
    ]

    # Initialize all variables for rendering
    generated_response = None
    selected_codes = []
    news_countries = []
    selected_time_code = "nl"
    selected_news_source=[]
    selected_news_source_temp=[]
    gemini_prompt = ""
    # FIX: Initialize input_topic here for the first GET request context
    input_topic = ''

    if request.method == 'POST':
        # 1. Retrieve and process form data
        selected_codes = request.form.getlist('countries')
        selected_time_code = request.form.get('time_frame')
        topic = request.form.get('topic', '')
        print('selected_codes:',selected_codes)

        # Retrieve the new source preference input
        selected_news_source = request.form.getlist('news_source')
        print('*****SELECTED_NEWS_SOURCE*************',selected_news_source, selected_news_source_temp)
        if (selected_news_source == []) or (selected_news_source == ['All']):
            selected_news_source_temp = NEWS_SOURCES[1:]
        else:
            selected_news_source_temp = selected_news_source.copy()
        print('*****SELECTED_NEWS_SOURCE_TEMP*************',selected_news_source, selected_news_source_temp)

        input_topic = topic # <--- ADDED LINE: ensures topic persists in textarea
        
        # Map selected codes to full country names
        selected_names = [d['name'] for d in country_data if d['code'] in selected_codes]
        time_name = next((tf['name'] for tf in time_frames if tf['code'] == selected_time_code), "No Limit")

        # API CALLS ADDED ~11/10/25
        # Call news apis for each country code + topic and store results in csv file
        news_countries = [item for item in selected_codes]
        news_countries.append('US')
        news_countries = list(set(news_countries)) #so it doesn't repeat 'US' if already there
        write_mode = 'w' # set output to start with a new file
        if 'NewsData.io' in selected_news_source_temp:
            for code in news_countries:
                query = f'({input_topic}) AND ({' OR '.join(selected_names)})'
                print('newsdata.io query:', query,' ...to country code...', code)
                nd_io.fetch_news_to_csv(
                    query=query,
                    country_code=code,
                    filename=f'newsdata_output.csv',
                    max_articles=25,
                    from_date='2025-11-01',
                    to_date='2025-11-14', 
                    write_mode=write_mode
                )
                write_mode = 'a' # set output to append to the current csv file

        if 'NewsApi.org' in selected_news_source_temp:
            for code in news_countries:
                query = f'({input_topic}) AND ({' OR '.join(selected_names)})'
                print('newsapi.org query:', query,' ...to country code...', code)
                na_org.fetch_news_from_newsapi_org_to_csv(
                    query=query,
                    country_code=code,
                    filename=f'newsdata_output.csv',
                    max_articles=25,
                    from_date='2025-11-01',
                    to_date='2025-11-14', 
                    write_mode=write_mode
                )
                write_mode = 'a' # set output to append to the current csv file

        # Load and transform the data
        tree_data = load_csv_to_json('newsdata_output.csv', root_name="API Search Results")
    
        # Save the result to a JSON file
        save_json(tree_data, 'country_news_tree.json')
       

        # 2. Construct the core prompt
        country_context = f"Ensure the response heavily features elements related to the following countries: {', '.join(selected_names)}." if selected_names else ""
        if time_name != "No Limit":
            time_context = f"Focus the context of the writing on information and events that occurred within the {time_name}." if time_name else ""
        else:
            time_context = ""
        
        gemini_prompt = (
            f"{country_context} {time_context} "
            f"Elaborate on this idea: '{topic}'"
        ).strip()

        # 3. Call 1: Generate Main Response and Grounded Sources
        main_response = generate_content_with_retry(
            gemini_prompt,
            SYSTEM_INSTRUCTION_TEXT,
            use_search=True
        )
        # Unpack result from the returned DICTIONARY
        generated_response = main_response.get("text")
        #all_sources = main_response.get("sources")
        
        # 4. Filter sources into general and Yahoo Finance
        #for source in all_sources:
        #    uri = source.get('uri', '').lower()
        #    if 'finance.yahoo.com' in uri:
        #        yahoo_sources.append(source)
        #    else:
        #        general_sources.append(source)
            
    # 6. Render Template with all variables
    return render_template(
        'index.html',
        country_data=country_data,
        time_frames=time_frames,
        generated_response=generated_response,
        selected_codes=selected_codes,
        selected_time_code=selected_time_code,
        gemini_prompt=gemini_prompt,
        input_topic=input_topic, # Pass the topic back to the template
    	news_sources=NEWS_SOURCES,             # <--- ADDED
    	selected_news_source=selected_news_source # <--- ADDED
    )

# --- Main Runner ---
if __name__ == '__main__':
    # Sanity check for API Key before running
    if not API_KEY:
        print("----------------------------------------------------------------------")
        print("ERROR: GEMINI_API_KEY environment variable is NOT set.")
        print("Please set your API key before running the server.")
        print("   e.g., export GEMINI_API_KEY='YOUR_KEY_HERE'")
        print("----------------------------------------------------------------------")
    else:
        app.run(host='0.0.0.0', port=5000)

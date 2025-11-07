import os
import sys
import json
import time
import requests
from flask import Flask, render_template, request, send_from_directory
from requests.exceptions import Timeout, RequestException

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
    "No Preference",
    "New York Times",
    "The Guardian",
    "Fox News",
    "The Wall Street Journal",
    "Bloomberg",
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
            try:
                # Use os.getcwd() for logging stability
                log_path = os.path.join(os.getcwd(), 'gemini_sources_log.json')
                with open(log_path, 'w', encoding='utf-8') as log_file:
                    json.dump(sources, log_file, indent=2)
                print(f"Successfully logged {len(sources)} sources to gemini_sources_log.json")
            except Exception as e:
                print(f"Warning: Could not log sources: {e}")

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


#def generate_news_articles(final_prompt):
def generate_news_articles(final_prompt, source_preference): # <--- New parameter
    """Generates a structured list of relevant news articles."""

    # Modify prompt to include the source preference
    if source_preference != "No Preference":
        source_instruction = f"from the news site: {source_preference}"
    else:
        source_instruction = "from reputable financial news sources"

    prompt = (
        f"Based on the following topic and context, identify 5 popular and recent news articles "
        f"{source_instruction}. Provide a brief, single-sentence summary for each article in the description field."
    )
#    prompt = f"Based on the following topic and context, identify 5 popular and recent news articles from reliable global news sources (like Reuters, #AP, Financial Times, BBC, etc.): {final_prompt}. Provide a brief, single-sentence summary for each article in the description field."

    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "The title of the news article."},
                "description": {"type": "STRING", "description": "A brief summary of the article."},
                "url": {"type": "STRING", "description": "The URL of the article. MUST start with http or https."}
            },
            "propertyOrdering": ["title", "description", "url"]
        }
    }

    # FIX: Explicitly pass all required arguments
    news_response = generate_content_with_retry(
        prompt, 
        SYSTEM_INSTRUCTION_NEWS, 
        response_schema=response_schema, 
        use_search=True
    )
    
    # FIX: Access response content using .get("text") because the function returns a DICTIONARY.
    if news_response.get("text") and not news_response.get("text").startswith("Error:"):
        try:
            return json.loads(news_response.get("text"))
        except json.JSONDecodeError:
            print("Warning: Failed to parse structured JSON response for news articles.")
            return []
    
    return []


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
    selected_time_code = "nl"
    gemini_prompt = ""
    general_sources = []
    yahoo_sources = []
    news_articles = []
    # FIX: Initialize input_topic here for the first GET request context
    input_topic = ''
    selected_news_source = NEWS_SOURCES[0] # <--- ADDED FIX: Initialize here

    if request.method == 'POST':
        # 1. Retrieve and process form data
        selected_codes = request.form.getlist('countries')
        selected_time_code = request.form.get('time_frame')
        topic = request.form.get('topic', '')

        # 1. Retrieve the new source preference input
        selected_news_source = request.form.get('news_source', NEWS_SOURCES[0]) # <--- ADDED & FIXED
        
        # FIX: Store retrieved topic back into the variable for persistence
        input_topic = topic # <--- ADDED LINE: ensures topic persists in textarea

        # Map selected codes to full country names
        selected_names = [d['name'] for d in country_data if d['code'] in selected_codes]
        time_name = next((tf['name'] for tf in time_frames if tf['code'] == selected_time_code), "No Limit")
        
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
        all_sources = main_response.get("sources")
        
        # 4. Filter sources into general and Yahoo Finance
        for source in all_sources:
            uri = source.get('uri', '').lower()
            if 'finance.yahoo.com' in uri:
                yahoo_sources.append(source)
            else:
                general_sources.append(source)

        # 5. Call 2: Generate Structured News Articles
        news_articles = generate_news_articles(gemini_prompt, selected_news_source) # <--- FIXED CALL
            
    # 6. Render Template with all variables
    return render_template(
        'index.html',
        country_data=country_data,
        time_frames=time_frames,
        generated_response=generated_response,
        selected_codes=selected_codes,
        selected_time_code=selected_time_code,
        gemini_prompt=gemini_prompt,
        general_sources=general_sources,
        yahoo_sources=yahoo_sources,
        news_articles=news_articles,
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

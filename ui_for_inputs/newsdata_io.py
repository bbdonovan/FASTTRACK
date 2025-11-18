import csv
import time
from newsdataapi import NewsDataApiClient
from newsdataapi.newsdataapi_exception import NewsdataException

# --- GLOBAL CONFIGURATION ---
# IMPORTANT: Replace this placeholder with your actual API key
API_KEY = 'pub_9619f9337fc44ef593a99b6e83fbf937'
client = NewsDataApiClient(apikey=API_KEY)

# Define the fields (column headers) we want to write to the CSV
# ADDED: 'API Source'
CSV_FIELDNAMES = ['Title', 'Publication Date', 'Link', 'Snippet', 'Source ID', 'Country Code', 'API Source']

def fetch_news_to_csv(query: str, country_code: str, filename: str, language_code: str = 'en', max_articles: int = 0, write_mode: str = 'w', from_date: str = None, to_date: str = None):
    """
    Fetches news articles from NewsData.io based on query and country, 
    and writes all paginated results to a CSV file.

    :param query: Keywords or phrases to search for (supports AND/OR/NOT).
    :param country_code: The 2-letter ISO country code (e.g., 'us', 'gb').
    :param filename: The path/name of the CSV file to create/overwrite.
    :param language_code: The 2-letter ISO language code (default is 'en').
    :param max_articles: Maximum total number of articles to fetch. 0 means no limit.
    :param write_mode: File mode to use: 'w' for write (overwrite/replace) or 'a' for append.
    :param from_date: Start date for the search (YYYY-MM-DD format). Optional, but IGNORED by latest_api.
    :param to_date: End date for the search (YYYY-MM-DD format). Optional, but IGNORED by latest_api.
    """
    
    # Validation for write mode
    if write_mode not in ['w', 'a']:
        print(f"❌ Invalid write_mode '{write_mode}'. Must be 'w' or 'a'. Defaulting to 'w'.")
        write_mode = 'w'
        
    print(f"\n--- Starting fetch for: '{query}' in {country_code.upper()} ({language_code.upper()}) ---")
    
    # Log date range if provided and warn user that latest_api ignores it
    date_range_str = ""
    if from_date or to_date:
        date_range_str = f" (Date Range: {from_date or 'Start'} to {to_date or 'End'})"
        print("  ⚠️ WARNING: 'from_date' and 'to_date' are NOT supported by the free tier's latest_api(). Filtering will be ignored.")
    print(f"  Search Query: {query}{date_range_str}")


    ALL_ARTICLES = []
    next_page_token = None
    page_number = 1
    
    while True:
        try:
            # 1. API Call with Pagination, Inputs, and Date Filters
            # Removed invalid date params (from_date, to_date) to prevent TypeError crash
            response = client.latest_api(
                q=query, 
                language=language_code, 
                country=country_code,
                page=next_page_token
            )
        
            # Error Handling for API issues (e.g., rate limits, invalid country code)
            if response.get('status') == 'error':
                 # Extract and print the specific error message
                error_message = response.get('results', {}).get('message', 'Unknown API Error')
                print(f"❌ API Error on Page {page_number}: {error_message}")
                break
            
            articles = response.get('results')

            if articles:
                articles_to_add = articles
                
                # --- LIMIT CHECK ---
                if max_articles > 0:
                    remaining_slots = max_articles - len(ALL_ARTICLES)
                    
                    if remaining_slots <= 0:
                        print(f"  🛑 Reached maximum limit of {max_articles} articles.")
                        break
                        
                    # If the current page has more articles than slots remaining, slice it.
                    if len(articles) > remaining_slots:
                        articles_to_add = articles[:remaining_slots]
                        
                # Add the determined set of articles to the main list
                ALL_ARTICLES.extend(articles_to_add)
                
                # Print status
                print(f"  Page {page_number}: Fetched {len(articles)} articles. Added {len(articles_to_add)}. Total collected: {len(ALL_ARTICLES)}")

                # Check if we hit the limit after adding
                if max_articles > 0 and len(ALL_ARTICLES) >= max_articles:
                    print(f"  🛑 Reached maximum limit of {max_articles} articles.")
                    break
                
                # Update token and page number
                next_page_token = response.get('nextPage')
                page_number += 1
                
                if not next_page_token:
                    print("  ➡️ Finished fetching all available pages (No nextPage token).")
                    break 
                
                # Pause to respect rate limits (prevents overwhelming the free tier)
                time.sleep(1) 
            
            else:
                print(f"  ❌ No more articles found on Page {page_number} or query finished.")
                break 

        except NewsdataException as e:
            # Catches errors raised by the Python client itself
            print(f"❌ Client Exception: An error occurred during the API call: {e}")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            break

    # --- 2. Write ALL Collected Data to CSV ---
    if not ALL_ARTICLES:
        print(f"No articles collected for '{query}'. Skipping CSV write.")
        return 

    print(f"\n--- Writing {len(ALL_ARTICLES)} Articles to CSV (Mode: {write_mode.upper()}) ---")

    # Use the provided write_mode parameter
    # 'newline=""' is used to prevent extra blank rows
    with open(filename, write_mode, encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        
        # Write header ONLY if we are in write/replace mode ('w').
        # If in append mode ('a'), we assume the header already exists.
        if write_mode == 'w':
             writer.writeheader()
        
        written_count = 0
        
        for article in ALL_ARTICLES:
            # Retrieve content, use empty string if it's None, then clean newlines
            content = article.get('content')
            clean_snippet = str(content) if content is not None else 'N/A'
            clean_snippet = clean_snippet.replace('\n', ' ')
            
            data_row = {
                'Title': article.get('title', 'N/A'),
                'Publication Date': article.get('pubDate', 'N/A'),
                'Link': article.get('link', 'N/A'),
                # FIXED: The problematic line now uses the pre-cleaned snippet:
                'Snippet': clean_snippet, 
                'Source ID': article.get('source_id', 'N/A'),
                'Country Code': country_code,
                # Hardcode the API source name
                'API Source': 'newsdata.io'
            }
            writer.writerow(data_row)
            written_count += 1

    print(f"✅ CSV FINALIZED: Wrote {written_count} articles to {filename}")

# --- Example Usage ---
if __name__ == '__main__':
    
    # Example 1: Search for Tesla or Rivian news in the US (No Limit, OVERWRITE)
    fetch_news_to_csv(
        query='tesla OR rivian',
        country_code='us',
        filename='us_vehicles_news.csv',
        language_code='en'
    )

    # Example 2: Search for tech news in Germany (No Limit, OVERWRITE)
    fetch_news_to_csv(
        query='technology AND AI NOT samsung',
        country_code='de',
        filename='de_tech_news.csv',
        language_code='de'
    )
    
    # Example 3: Limit the fetch to 15 articles (OVERWRITE)
    fetch_news_to_csv(
        query='apple OR microsoft',
        country_code='gb',
        filename='gb_limited_tech.csv',
        language_code='en',
        max_articles=15
    )

    # Example 4: Search with a specific date range (start date only)
    # This will now RUN but the date filter will be ignored as it's not supported by latest_api()
    fetch_news_to_csv(
        query='inflation',
        country_code='us',
        filename='us_inflation_oct_2025.csv',
        from_date='2025-10-01' 
    )

    # Example 5: Search with a specific date range (start and end date)
    # This will also RUN but the date filters will be ignored.
    fetch_news_to_csv(
        query='stock market',
        country_code='in',
        filename='india_stock_market_report.csv',
        from_date='2025-11-01', 
        to_date='2025-11-08'   
    )

    # Example 6: Search for specific sports teams in the UK (OVERWRITE)
    fetch_news_to_csv(
        query='"Manchester United" OR "Liverpool FC"',
        country_code='gb',
        filename='uk_football_news.csv'
    )
    
    # Example 7: Demonstrate the 'a' (append) mode with different countries.
    print("\n--- Running APPEND Demo with Country Code ---")
    
    # First call (US) - uses default mode 'w' (writes header and data)
    fetch_news_to_csv(
        query='election OR government',
        country_code='us',
        filename='political_news_combined.csv',
        max_articles=3
    )
    
    # Second call (GB) - uses mode 'a' (adds data to the end, skipping header)
    fetch_news_to_csv(
        query='election OR government',
        country_code='gb',
        filename='political_news_combined.csv',
        max_articles=3,
        write_mode='a' 
    )
    print("APPEND Demo Finished. Check 'political_news_combined.csv' for combined data with country tags.")
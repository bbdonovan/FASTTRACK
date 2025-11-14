import csv
import time
import os
from newsapi import NewsApiClient, newsapi_exception # Requires 'pip install newsapi-python'

# You MUST replace this placeholder with a valid NewsAPI.org key
NEWSAPI_ORG_API_KEY = 'd8ffe57b76284e338f9170e1699e339c' 

def fetch_news_from_newsapi_org_to_csv(
    query: str,
    country_code: str,
    filename: str,
    write_mode: str = 'w',
    max_articles: int = 0,
    from_date: str = None,
    to_date: str = None,
):
    """
    Fetches news articles using the NewsAPI.org /v2/everything endpoint, 
    including advanced search operators (AND/OR/NOT).
    
    The NewsAPI.org Free Tier limits include:
    - Search up to one month back.
    - Max 100 articles returned per single request.
    - Only provides headlines, links, and short snippets (not full content).

    :param query: Keywords or phrases (e.g., 'stock market AND inflation').
    :param country_code: 2-letter country code (e.g., 'us', 'gb').
    :param filename: The name of the CSV file to write to.
    :param write_mode: 'w' for overwrite (default), 'a' for append.
    :param max_articles: Maximum number of articles to fetch (0 for no limit).
    :param from_date: Start date for search (YYYY-MM-DD format, max 30 days old).
    :param to_date: End date for search (YYYY-MM-DD format, max 30 days old).
    """

    print(f"\n--- Starting fetch for NewsAPI.org: '{query}' in {country_code.upper()} ---")

    # Initialize the client (using api_key, not apikey)
    try:
        client = NewsApiClient(api_key=NEWSAPI_ORG_API_KEY)
    except newsapi_exception.NewsAPIException as e:
        print(f"❌ Error initializing NewsAPI client: {e}")
        return

    # MODIFIED: Added 'API Source'
    CSV_FIELDNAMES = ['Title', 'Publication Date', 'Link', 'Snippet', 'Source ID', 'Country Code', 'API Source']
    ALL_ARTICLES = []
    
    # NewsAPI.org uses pagination based on page number and page size.
    current_page = 1
    page_size = 100 # Maximum page size allowed by the API
    total_fetched = 0
    total_results = 1 # Initialize > 0 to start the loop

    while total_fetched < total_results and (max_articles == 0 or total_fetched < max_articles):
        try:
            # 1. Fetch data using the 'everything' endpoint
            response = client.get_everything(
                q=query,
                language='en',
                # NewsAPI uses from_param and to_param for date filtering in the Python client
                from_param=from_date,
                to=to_date,
                page=current_page,
                # FIX: Changed 'pageSize' to the correct snake_case 'page_size'
                page_size=page_size 
            )
            
            # 2. Handle Errors and Status
            if response['status'] != 'ok':
                if response['code'] == 'apiKeyInvalid':
                    print(f"❌ Error: API Key is invalid or missing.")
                elif response['code'] == 'maximumResultsReached':
                    print("⚠️ Warning: Reached maximum results allowed by your plan or API limit.")
                else:
                    print(f"❌ API Error: {response['message']}")
                break

            # 3. Process the Response
            articles_found = response['articles']
            total_results = response['totalResults']

            if not articles_found:
                print("⚠️ No more articles found for this query or no articles matched the criteria.")
                break

            articles_to_append = articles_found
            
            # Check if we need to limit the final page append
            if max_articles > 0 and (total_fetched + len(articles_found)) > max_articles:
                limit_needed = max_articles - total_fetched
                articles_to_append = articles_found[:limit_needed]

            ALL_ARTICLES.extend(articles_to_append)
            total_fetched += len(articles_to_append)

            print(f"  ✅ Page {current_page} fetched ({len(articles_to_append)} articles). Total collected: {total_fetched}")

            # Check if we should stop early due to max_articles limit
            if max_articles > 0 and total_fetched >= max_articles:
                break
                
            current_page += 1
            
            # Rate limit buffer
            time.sleep(1) 

        except newsapi_exception.NewsAPIException as e:
            print(f"❌ An error occurred during API call: {e}")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            break


    # --- 4. Write ALL Collected Data to CSV ---
    if not ALL_ARTICLES:
        print(f"No articles collected for '{query}'. Skipping CSV write.")
        return

    print(f"\n--- Writing {len(ALL_ARTICLES)} Articles to CSV ({'Append' if write_mode == 'a' else 'Replace'}) ---")

    # Determine if we need to write the header
    write_header = (write_mode == 'w' or (write_mode == 'a' and not os.path.exists(filename)))

    with open(filename, write_mode, encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        
        if write_header:
            writer.writeheader()
        
        written_count = 0
        
        for article in ALL_ARTICLES:
            # NewsAPI returns source as a nested dictionary: {'id': 'cnn', 'name': 'CNN'}
            source_info = article.get('source', {})
            
            data_row = {
                'Title': article.get('title', 'N/A'),
                'Publication Date': article.get('publishedAt', 'N/A'),
                'Link': article.get('url', 'N/A'),
                # NewsAPI.org provides a 'description' and 'content' snippet. We use description.
                'Snippet': article.get('description', 'N/A').replace('\n', ' '), 
                # Use the source name for consistency with NewsData.io's output
                'Source ID': source_info.get('name', 'N/A'), 
                'Country Code': country_code.upper(), # Use the input country code
                # NEW LINE: Hardcode the API source name
                'API Source': 'newsapi.org'
            }
            
            writer.writerow(data_row)
            written_count += 1

    print(f"✅ FINAL RESULT: Successfully wrote {written_count} articles to {filename}")

if __name__ == '__main__':
    import os
    
    # --- DEMO 1: Basic Search and Overwrite (using 'w') ---
    fetch_news_from_newsapi_org_to_csv(
        query='spaceX OR NASA',
        country_code='us',
        filename='newsapi_space_news.csv',
        write_mode='w',
        max_articles=20, # Limit the run for demonstration
        from_date='2025-10-01' # NOTE: Free tier max search back is ~30 days
    )

    # --- DEMO 2: Combined Search and Append (using 'a') ---
    # Create the combined file name
    COMBINED_FILE = 'newsapi_combined_data.csv'
    if os.path.exists(COMBINED_FILE):
        os.remove(COMBINED_FILE) # Start fresh for the demo

    # Run 1: Fetch US tech news and REPLACE
    fetch_news_from_newsapi_org_to_csv(
        query='technology AND startup',
        country_code='us',
        filename=COMBINED_FILE,
        write_mode='w',
        max_articles=10
    )

    # Run 2: Fetch UK finance news and APPEND
    fetch_news_from_newsapi_org_to_csv(
        query='finance AND banking',
        country_code='gb',
        filename=COMBINED_FILE,
        write_mode='a',
        max_articles=10
    )
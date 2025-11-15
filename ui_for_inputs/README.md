Summary:
1. This app accepts user inputs for country (can select multiple), time frame (still in work), api sources (can select multiple), and a topic.
2. The backend code uses the inputs to search for relevant articles from selected news api sources. --Note: The api calls are coded in newsdata_io.py and newsapi_org.py, which require python packages newsdataapi and newsapi-python, respectively.
3. Outputs are a csv file called newsdata_output.csv and a D3 collapsible tree displaying article titles (still in work) --Note: the D3 tree is fed by country_news_tree.json which is automatically generated from the csv output using csv_to_json_converter.py.
4. A display of an LLM prompt and Gemini response are also provided (requires GEMINI_API_KEY) as a test for connectivity with inputs and experimentally incorporating general LLM results.

At cmd prompt, do the following one time for setup:
1. go to local directory that holds app.py
2. set GEMINI_API_KEY=your_gemini_api_key (go to aistudio.google.com to get a free api key)
3. pip install Flask requests
4. pip install newsdataapi
5. pip install newsapi-python

To start the app:
1. python app.py
2. open Chrome and go to local url displayed in the command console (usually http://127.0.0.1:5000)

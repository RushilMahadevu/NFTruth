import requests
import os 
from dotenv import load_dotenv

load_dotenv()

OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY")

def get_opensea_collection(slug: str) -> dict:
    """
    Fetch collection data from OpenSea API v2
    slug is the collection's unique identifier (e.g., "boredapeyachtclub")
    """
    url = f"https://api.opensea.io/api/v2/collections/{slug}"
    headers = {"X-API-KEY": OPENSEA_API_KEY}
    
    response = requests.get(url, headers=headers) # Send GET request to OpenSea API
    
    if response.status_code == 200: # 200 means success
        return response.json()
    else:
        print(f"OpenSea API error: {response.status_code}")
        print(f"Response: {response.text}")
        return {}
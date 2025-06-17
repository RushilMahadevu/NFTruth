import requests
import os 
from dotenv import load_dotenv

load_dotenv()

OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY")
OPENSEA_BASE_URL = "https://api.opensea.io"

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

def get_opensea_collection_stats(collection_slug: str) -> dict:
    """
    Get collection statistics from OpenSea API
    """
    url = f"{OPENSEA_BASE_URL}/api/v2/collections/{collection_slug}/stats"
    
    headers = {
        "accept": "application/json",
        "x-api-key": OPENSEA_API_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching OpenSea stats for {collection_slug}: {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except Exception as e:
        print(f"Exception fetching OpenSea stats for {collection_slug}: {e}")
        return {}
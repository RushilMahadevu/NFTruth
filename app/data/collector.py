import requests
import os 
from dotenv import load_dotenv

load_dotenv()

OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_opensea_collection(slug: str) -> dict:
    """
    Fetch collection data from OpenSea API v2
    slug is the collection's unique identifier (e.g., "boredapeyachtclub")
    """
    url = f"https://api.opensea.io/api/v2/collections/{slug}"
    headers = {"X-API-KEY": OPENSEA_API_KEY}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"OpenSea API error: {response.status_code}")
        print(f"Response: {response.text}")
        return {}

# Example usage
if __name__ == "__main__":
    collection_data = get_opensea_collection("boredapeyachtclub")
    if collection_data:
        print("Collection found!")
        # Print some basic info if available
        if 'name' in collection_data:
            print(f"Collection Name: {collection_data['name']}")
        if 'description' in collection_data:
            print(f"Description: {collection_data['description'][:100]}...")
    else:
        print("No collection data retrieved.")
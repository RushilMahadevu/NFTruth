import requests
import os 
from dotenv import load_dotenv
import base64

load_dotenv()

OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

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

def get_wallet_transactions(address: str) -> dict:
    """
    Fetch wallet transactions from Etherscan API
    address is the Ethereum wallet address
    """
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account", # tells Etherscan we want account info
        "action": "txlist", # action to get transaction list
        "address": address, # Ethereum address to query
        "startblock": 0, # start from the beginning of the blockchain
        "endblock": 99999999, # end at the latest block
        "sort": "desc", # sort transactions by descending order (most recent first)
        "apikey": ETHERSCAN_API_KEY # your Etherscan API key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200: # 200 means success
        data = response.json()
        if data.get("status") == "1": # Check if the API call was successful 1 means success
            return data
        else:
            print(f"Etherscan API error: {data.get('message', 'Unknown error')}") # data.get('message') provides error details
            return {}
    else:
        print(f"Etherscan HTTP error: {response.status_code}")
        print(f"Response: {response.text}")
        return {}

def get_reddit_access_token() -> str:
    """
    Get OAuth access token from Reddit API
    Returns access token string or empty string if failed
    """
    auth_url = "https://www.reddit.com/api/v1/access_token"
    
    # Create basic auth header with client ID and secret
    auth_string = f"{REDDIT_CLIENT_ID}:{REDDIT_CLIENT_SECRET}" # Combine client ID and secret
    auth_bytes = auth_string.encode('ascii') # Encode to bytes
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii') # Base64 encode the bytes
    
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "User-Agent": REDDIT_USER_AGENT
    }
    
    data = {
        "grant_type": "client_credentials"
    }
    
    response = requests.post(auth_url, headers=headers, data=data)
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data.get("access_token", "")
    else:
        print(f"Reddit auth error: {response.status_code}")
        print(f"Response: {response.text}")
        return ""

def get_reddit_hype(query: str, limit: int = 50) -> dict:
    """
    Analyze Reddit hype for a given query (collection name or stock symbol).
    Returns post count, avg upvotes, comment count, and positive/negative keyword hits.
    """
    access_token = get_reddit_access_token()
    if not access_token:
        return {}

    url = "https://oauth.reddit.com/search"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": REDDIT_USER_AGENT
    }

    params = {
        "q": query,
        "type": "link",
        "sort": "new",  # Capture recent sentiment spikes
        "limit": min(limit, 100)
    }

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Reddit API error: {response.status_code}")
        print(f"Response: {response.text}")
        return {}

    posts = response.json().get("data", {}).get("children", [])
    total_upvotes = 0
    total_comments = 0
    positive_keywords = ["undervalued", "moon", "bullish", "diamond hands", "buy"]
    negative_keywords = ["scam", "rug", "dump", "exit", "overhyped", "ponzi"]
    pos_hits = neg_hits = 0

    for post in posts:
        data = post.get("data", {})
        title = data.get("title", "").lower()
        body = data.get("selftext", "").lower()
        combined = title + " " + body

        total_upvotes += data.get("ups", 0)
        total_comments += data.get("num_comments", 0)

        pos_hits += sum(kw in combined for kw in positive_keywords)
        neg_hits += sum(kw in combined for kw in negative_keywords)

    return {
        "reddit_post_count": len(posts),
        "reddit_total_upvotes": total_upvotes,
        "reddit_total_comments": total_comments,
        "reddit_positive_hits": pos_hits,
        "reddit_negative_hits": neg_hits,
        "reddit_sentiment_score": pos_hits - neg_hits
    }

if __name__ == "__main__":
    print(get_reddit_hype("Bored Ape Yacht Club"))
    print(get_reddit_hype("AMC"))

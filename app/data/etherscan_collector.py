import requests
import os 
from dotenv import load_dotenv

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_etherscan_wallet_tx(address: str) -> dict:
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
    print(f"Fetching Etherscan transactions for {address}...")
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
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




# Example usage and test cases
if __name__ == "__main__":
    print("=== Testing OpenSea API ===")
    collection_data = get_opensea_collection("boredapeyachtclub")
    if collection_data:
        print("✅ OpenSea Collection found!")
        # Print some basic info if available
        if 'name' in collection_data:
            print(f"Collection Name: {collection_data['name']}")
        if 'description' in collection_data:
            print(f"Description: {collection_data['description'][:100]}...")
    else:
        print("❌ No OpenSea collection data retrieved.")
    
    print("\n=== Testing Etherscan API ===")
    # Using Vitalik Buterin's address as a test case (public knowledge)
    test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    tx_data = get_wallet_transactions(test_address)
    if tx_data and tx_data.get("result"):
        transactions = tx_data["result"]
        print(f"✅ Etherscan API working! Found {len(transactions)} transactions")
        if transactions:
            latest_tx = transactions[0]  # Most recent transaction (sorted desc)
            print(f"Latest transaction hash: {latest_tx.get('hash', 'N/A')}")
            print(f"Block number: {latest_tx.get('blockNumber', 'N/A')}")
            print(f"Value: {int(latest_tx.get('value', 0)) / 10**18:.4f} ETH")
    else:
        print("❌ No Etherscan transaction data retrieved.")
    
    print("\n=== API Key Status ===")
    print(f"OpenSea API Key configured: {'✅' if OPENSEA_API_KEY else '❌'}")
    print(f"Etherscan API Key configured: {'✅' if ETHERSCAN_API_KEY else '❌'}")
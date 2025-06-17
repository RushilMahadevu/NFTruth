import os
import sys
from dotenv import load_dotenv
from ..app.data.opensea_collector import get_opensea_collection, get_opensea_collection_stats

# Load environment variables
load_dotenv()

def test_opensea_collection(collection_slug: str):
    """
    Test if OpenSea can find and return data for a collection
    """
    print(f"🔍 Testing OpenSea API for collection: {collection_slug}")
    print("=" * 50)
    
    # Test collection data
    print("📦 Fetching collection data...")
    collection_data = get_opensea_collection(collection_slug)
    
    if collection_data:
        print("✅ Collection found!")
        print(f"   Name: {collection_data.get('name', 'N/A')}")
        print(f"   Total Supply: {collection_data.get('total_supply', 'N/A')}")
        print(f"   Verified: {collection_data.get('safelist_status', 'N/A')}")
        print(f"   Discord: {'Yes' if collection_data.get('discord_url') else 'No'}")
        print(f"   Twitter: {'Yes' if collection_data.get('twitter_username') else 'No'}")
    else:
        print("❌ Collection not found or API error")
        return False
    
    # Test stats data
    print("\n📊 Fetching collection stats...")
    stats_data = get_opensea_collection_stats(collection_slug)
    
    if stats_data and 'total' in stats_data:
        total_stats = stats_data['total']
        print("✅ Stats found!")
        print(f"   Floor Price: {total_stats.get('floor_price', 0)} ETH")
        print(f"   Total Volume: {total_stats.get('volume', 0):,.0f} ETH")
        print(f"   Owners: {total_stats.get('num_owners', 0):,}")
        print(f"   Average Price: {total_stats.get('average_price', 0):.2f} ETH")
    else:
        print("❌ Stats not found or API error")
        return False
    
    print("\n🎉 OpenSea API test successful!")
    return True

def main():
    """
    Interactive OpenSea collection tester
    """
    print("🌊 OpenSea Collection Tester")
    print("=" * 40)
    
    # Check if API key is set
    api_key = os.getenv("OPENSEA_API_KEY")
    if not api_key:
        print("❌ OPENSEA_API_KEY not found in .env file!")
        return
    else:
        print("✅ OpenSea API key found")
    
    print("\nEnter collection slugs to test (or 'quit' to exit)")
    print("Examples: boredapeyachtclub, cryptopunks, azuki, doodles-official")
    
    while True:
        collection_slug = input("\nCollection slug: ").strip()
        
        if collection_slug.lower() == 'quit':
            break
        
        if not collection_slug:
            print("Please enter a valid collection slug")
            continue
        
        print()
        success = test_opensea_collection(collection_slug)
        
        if not success:
            print("\n💡 Tips:")
            print("   - Make sure the collection slug is correct")
            print("   - Check OpenSea URL: opensea.io/collection/YOUR_SLUG")
            print("   - Some collections might be private or delisted")

if __name__ == "__main__":
    main()
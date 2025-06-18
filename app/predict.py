import json
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from data.opensea_collector import get_opensea_collection, get_opensea_collection_stats
from data.reddit_collector import RedditDataCollector
from models.model import NFTAuthenticityModel
from opensea_collections import COLLECTION_SLUGS
from difflib import get_close_matches
import re
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

class NFTPredictor:
    """
    Use the trained model to predict NFT collection authenticity
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.reddit_collector = None
        self.load_model()
        self.setup_reddit_collector()
    
    def normalize_slug_input(self, user_input: str) -> tuple:
        """
        Normalize user input and find matching collection slug
        Returns: (normalized_slug, collection_name, confidence_score)
        """
        if not user_input:
            return None, None, 0.0
        
        # Clean and normalize input
        cleaned_input = user_input.strip().lower()
        cleaned_input = re.sub(r'[^a-zA-Z0-9\s-]', '', cleaned_input)  # Remove special chars except hyphens
        cleaned_input = re.sub(r'\s+', ' ', cleaned_input)  # Normalize whitespace
        
        # Direct slug match (case insensitive)
        for collection_name, slug in COLLECTION_SLUGS.items():
            if cleaned_input == slug.lower():
                return slug, collection_name, 1.0
        
        # Direct collection name match (case insensitive)
        for collection_name, slug in COLLECTION_SLUGS.items():
            if cleaned_input == collection_name.lower():
                return slug, collection_name, 1.0
        
        # Fuzzy matching on collection names
        collection_names = list(COLLECTION_SLUGS.keys())
        name_matches = get_close_matches(
            cleaned_input, 
            [name.lower() for name in collection_names], 
            n=3, 
            cutoff=0.6
        )
        
        if name_matches:
            # Find the original collection name
            for collection_name in collection_names:
                if collection_name.lower() == name_matches[0]:
                    return COLLECTION_SLUGS[collection_name], collection_name, 0.8
        
        # Fuzzy matching on slugs
        slugs = list(COLLECTION_SLUGS.values())
        slug_matches = get_close_matches(
            cleaned_input, 
            slugs, 
            n=3, 
            cutoff=0.6
        )
        
        if slug_matches:
            # Find the collection name for this slug
            for collection_name, slug in COLLECTION_SLUGS.items():
                if slug == slug_matches[0]:
                    return slug, collection_name, 0.7
        
        # Partial matching - check if input is contained in any collection name or slug
        for collection_name, slug in COLLECTION_SLUGS.items():
            if cleaned_input in collection_name.lower() or cleaned_input in slug.lower():
                return slug, collection_name, 0.6
            # Also check reverse - if collection name/slug is contained in input
            if collection_name.lower() in cleaned_input or slug.lower() in cleaned_input:
                return slug, collection_name, 0.5
        
        # If no match found, return the cleaned input as-is
        return cleaned_input, None, 0.0
    
    def get_suggestions(self, user_input: str, num_suggestions: int = 5) -> list:
        """
        Get multiple suggestions for a user input
        Returns list of (slug, collection_name, confidence) tuples
        """
        if not user_input:
            return []
        
        cleaned_input = user_input.strip().lower()
        cleaned_input = re.sub(r'[^a-zA-Z0-9\s-]', '', cleaned_input)
        
        suggestions = []
        
        # Get fuzzy matches for collection names
        collection_names = list(COLLECTION_SLUGS.keys())
        name_matches = get_close_matches(
            cleaned_input, 
            [name.lower() for name in collection_names], 
            n=num_suggestions, 
            cutoff=0.4
        )
        
        for match in name_matches:
            for collection_name in collection_names:
                if collection_name.lower() == match:
                    suggestions.append((
                        COLLECTION_SLUGS[collection_name], 
                        collection_name, 
                        0.8
                    ))
                    break
        
        # Get fuzzy matches for slugs
        slugs = list(COLLECTION_SLUGS.values())
        slug_matches = get_close_matches(
            cleaned_input, 
            slugs, 
            n=num_suggestions, 
            cutoff=0.4
        )
        
        for match in slug_matches:
            for collection_name, slug in COLLECTION_SLUGS.items():
                if slug == match:
                    # Avoid duplicates
                    if not any(s[0] == slug for s in suggestions):
                        suggestions.append((slug, collection_name, 0.7))
                    break
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[:num_suggestions]
    
    def setup_reddit_collector(self):
        """Initialize Reddit collector with credentials from .env"""
        try:
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT")
            
            if client_id and client_secret and user_agent:
                self.reddit_collector = RedditDataCollector(client_id, client_secret, user_agent)
                print("❤️ Reddit collector initialized")
            else:
                print("⚠️ Reddit credentials not found in .env file")
                self.reddit_collector = None
        except Exception as e:
            print(f"⚠️ Could not initialize Reddit collector: {e}")
            self.reddit_collector = None
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load('model_outputs/best_nft_model.pkl')
            self.scaler = joblib.load('model_outputs/feature_scaler.pkl')
            
            # Load metadata to get feature names
            with open('model_outputs/model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('features_used', [])
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure to train the model first by running: python3 app/models/model.py")
    
    def predict_collection(self, collection_slug: str) -> dict:
        """
        Predict authenticity for a given NFT collection
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        print(f"🔍 Analyzing collection: {collection_slug}")
        
        try:
            # Collect data for this collection
            collection_data = get_opensea_collection(collection_slug)
            stats_data = get_opensea_collection_stats(collection_slug)
            
            # Collect Reddit data if available
            reddit_data = {}
            if self.reddit_collector:
                try:
                    reddit_data = self.reddit_collector.collect_targeted_data(collection_slug)
                except Exception as e:
                    print(f"⚠️ Reddit data collection failed: {e}")
                    reddit_data = {'total_mentions': 0, 'avg_sentiment': 0.5, 'total_engagement': 0}
            else:
                reddit_data = {'total_mentions': 0, 'avg_sentiment': 0.5, 'total_engagement': 0}
            
            if not collection_data or not stats_data:
                return {
                    "collection": collection_slug,
                    "error": "Could not fetch collection data from OpenSea"
                }
            
            # Extract features (similar to ml_data_transformer)
            features = self.extract_features(collection_data, stats_data, reddit_data)
            
            # Engineer additional features
            features_df = pd.DataFrame([features])
            engineered_features = self.engineer_features(features_df)
            
            # Scale features
            features_scaled = self.scaler.transform(engineered_features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Prepare result
            result = {
                "collection": collection_slug,
                "prediction": "Legitimate" if prediction == 1 else "Suspicious",
                "confidence": {
                    "legitimate": float(probability[1]) if len(probability) > 1 else 0.0,
                    "suspicious": float(probability[0]) if len(probability) > 0 else 0.0
                },
                "features_analyzed": features,
                "risk_score": float(1 - probability[1]) if len(probability) > 1 else 1.0,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "collection": collection_slug,
                "error": f"Prediction failed: {str(e)}"
            }
    
    def extract_features(self, collection_data, stats_data, reddit_data):
        """Extract features from collection data"""
        features = {}
        
        # Basic collection features
        features['total_supply'] = collection_data.get('total_supply', 0)
        features['is_verified'] = 1 if collection_data.get('safelist_status') == 'verified' else 0
        features['has_discord'] = 1 if collection_data.get('discord_url') else 0
        features['has_twitter'] = 1 if collection_data.get('twitter_username') else 0
        features['trait_offers_enabled'] = 1 if collection_data.get('trait_offers_enabled') else 0
        features['collection_offers_enabled'] = 1 if collection_data.get('collection_offers_enabled') else 0
        
        # Stats features
        if stats_data and 'total' in stats_data:
            total_stats = stats_data['total']
            features['floor_price'] = float(total_stats.get('floor_price', 0) or 0)
            features['market_cap'] = float(total_stats.get('market_cap', 0) or 0)
            features['total_volume'] = float(total_stats.get('volume', 0) or 0)
            features['num_owners'] = int(total_stats.get('num_owners', 0) or 0)
            features['average_price'] = float(total_stats.get('average_price', 0) or 0)
        else:
            features.update({
                'floor_price': 0.0, 'market_cap': 0.0, 'total_volume': 0.0,
                'num_owners': 0, 'average_price': 0.0
            })
        
        # Reddit features
        features['reddit_mentions'] = reddit_data.get('total_mentions', 0)
        features['reddit_sentiment'] = reddit_data.get('avg_sentiment', 0.5)
        features['reddit_engagement'] = reddit_data.get('total_engagement', 0)
        
        return features
    
    def engineer_features(self, df):
        """Engineer additional features (same as in model training)"""
        X = df.values.astype(float)
        
        additional_features = []
        
        floor_price = df['floor_price'].values
        market_cap = df['market_cap'].values
        total_volume = df['total_volume'].values
        num_owners = df['num_owners'].values
        average_price = df['average_price'].values
        
        # Volume per owner - safely handle division by zero
        volume_per_owner = np.zeros_like(num_owners)
        mask = num_owners > 0
        if np.any(mask):
            volume_per_owner[mask] = total_volume[mask] / num_owners[mask]
        additional_features.append(volume_per_owner)
        
        # Market cap to volume ratio - safely handle division by zero
        mc_volume_ratio = np.zeros_like(total_volume)
        mask = total_volume > 0
        if np.any(mask):
            mc_volume_ratio[mask] = market_cap[mask] / total_volume[mask]
        additional_features.append(mc_volume_ratio)
        
        # Price premium - safely handle division by zero
        price_premium = np.ones_like(floor_price)  # Default to 1 if division is invalid
        mask = floor_price > 0
        if np.any(mask):
            price_premium[mask] = average_price[mask] / floor_price[mask]
        additional_features.append(price_premium)
        
        # Social engagement score
        reddit_mentions = df['reddit_mentions'].values
        reddit_engagement = df['reddit_engagement'].values
        social_score = reddit_mentions + reddit_engagement
        additional_features.append(social_score)
        
        # Liquidity indicator - use np.maximum to prevent issues with negative values
        liquidity = np.sqrt(np.maximum(total_volume, 0) * np.maximum(num_owners, 0))
        additional_features.append(liquidity)
        
        if additional_features:
            additional_features = np.column_stack(additional_features)
            X = np.column_stack([X, additional_features])
        
        return X

    def parse_opensea_url(self, url: str) -> str:
        """
        Extract collection slug from OpenSea URL
        Examples:
        - https://opensea.io/collection/cryptopunks -> cryptopunks
        - https://opensea.io/collection/bored-ape-yacht-club -> bored-ape-yacht-club
        """
        try:
            # Clean the URL
            url = url.strip()
            
            # Handle various URL formats
            if 'opensea.io/collection/' in url:
                # Extract everything after /collection/
                slug = url.split('opensea.io/collection/')[1]
                # Remove any trailing parameters or fragments
                slug = slug.split('?')[0].split('#')[0].split('/')[0]
                return slug
            
            return None
        except Exception:
            return None
    
    def normalize_input(self, user_input: str) -> str:
        """
        Normalize any user input (slug, URL, or name) to a collection slug
        """
        if not user_input:
            return None
        
        user_input = user_input.strip()
        
        # Check if it's an OpenSea URL first
        if 'opensea.io/collection/' in user_input:
            slug = self.parse_opensea_url(user_input)
            if slug:
                return slug
        
        # Try to find a match in our known collections
        normalized_slug, collection_name, confidence = self.normalize_slug_input(user_input)
        
        # If we found a good match, use it
        if confidence >= 0.6:
            return normalized_slug
        
        # Otherwise, clean the input and use it as-is
        cleaned = user_input.lower().replace(' ', '-')
        cleaned = re.sub(r'[^a-z0-9-]', '', cleaned)  # Remove special chars except hyphens
        cleaned = re.sub(r'-+', '-', cleaned)  # Collapse multiple hyphens
        cleaned = cleaned.strip('-')  # Remove leading/trailing hyphens
        
        return cleaned

def main():
    """Test the predictor"""
    predictor = NFTPredictor()
    
    print("\n🤖 NFT Authenticity Predictor")
    print("=" * 40)
    print("\n💡 Enter collection name, slug, or OpenSea URL")
    print("Examples: 'CryptoPunks', 'bored-ape-yacht-club', 'https://opensea.io/collection/azuki'")
    print("Type 'help' for examples or 'quit' to exit")
    
    while True:
        user_input = input("\nEnter NFT collection (or 'quit' to exit): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'help':
            print("\n📚 EXAMPLES:")
            print("=" * 40)
            print("Collection Names:")
            print("  • CryptoPunks")
            print("  • Bored Ape Yacht Club")
            print("  • Mutant Ape Yacht Club")
            print("\nCollection Slugs:")
            print("  • cryptopunks")
            print("  • bored-ape-yacht-club")
            print("  • mutant-ape-yacht-club")
            print("\nOpenSea URLs:")
            print("  • https://opensea.io/collection/cryptopunks")
            print("  • https://opensea.io/collection/bored-ape-yacht-club")
            continue
        
        if not user_input:
            print("Please enter a collection name, slug, or URL")
            continue
        
        # Check if it's an OpenSea URL first
        if 'opensea.io/collection/' in user_input:
            normalized_slug = predictor.parse_opensea_url(user_input)
            collection_name = None  # Initialize collection_name for URL inputs
            confidence = 0.0
            if normalized_slug:
                print(f"✅ Extracted from URL: {normalized_slug}")
            else:
                print("❌ Could not parse OpenSea URL. Please try again.")
                continue
        else:
            # Try to find a match in our known collections
            normalized_slug, collection_name, confidence = predictor.normalize_slug_input(user_input)
            
            if confidence == 1.0:
                # Exact match found
                print(f"✅ Found: {collection_name} ({normalized_slug})")
            elif confidence >= 0.6:
                # Close match found - ask for confirmation
                print(f"🔍 Did you mean '{collection_name}' ({normalized_slug})? ({confidence*100:.0f}% match)")
                
                while True:
                    choice = input("Enter 'y' for yes, 'n' for no, or 's' to see more suggestions: ").strip().lower()
                    
                    if choice in ['y', 'yes']:
                        print(f"✅ Using: {collection_name}")
                        break
                    elif choice in ['n', 'no']:
                        print("🔄 Using your original input")
                        normalized_slug = predictor.normalize_input(user_input)
                        collection_name = None
                        confidence = 0.0
                        break
                    elif choice in ['s', 'suggestions']:
                        suggestions = predictor.get_suggestions(user_input, num_suggestions=5)
                        if suggestions:
                            print("\n🔍 Similar collections found:")
                            for i, (slug, name, conf) in enumerate(suggestions, 1):
                                print(f"   {i}. {name} ({slug}) - {conf*100:.0f}% match")
                            print(f"   {len(suggestions)+1}. Use original input: '{user_input}'")
                            
                            try:
                                selection = input(f"\nSelect 1-{len(suggestions)+1}: ").strip()
                                if selection.isdigit():
                                    sel_num = int(selection) - 1
                                    if 0 <= sel_num < len(suggestions):
                                        normalized_slug, collection_name, confidence = suggestions[sel_num]
                                        print(f"✅ Selected: {collection_name}")
                                        break
                                    elif sel_num == len(suggestions):
                                        normalized_slug = predictor.normalize_input(user_input)
                                        collection_name = None
                                        confidence = 0.0
                                        print(f"✅ Using original: {user_input}")
                                        break
                                    else:
                                        print("Invalid selection. Please try again.")
                                else:
                                    print("Please enter a number.")
                            except ValueError:
                                print("Please enter a valid number.")
                        else:
                            print("No additional suggestions found.")
                            normalized_slug = predictor.normalize_input(user_input)
                            collection_name = None
                            confidence = 0.0
                            break
                    else:
                        print("Please enter 'y', 'n', or 's'")
            elif confidence > 0.0:
                # Weak match found - show suggestions directly
                suggestions = predictor.get_suggestions(user_input, num_suggestions=5)
                if suggestions:
                    print(f"🔍 '{user_input}' not found. Did you mean one of these?")
                    for i, (slug, name, conf) in enumerate(suggestions, 1):
                        print(f"   {i}. {name} ({slug}) - {conf*100:.0f}% match")
                    print(f"   {len(suggestions)+1}. Use original input: '{user_input}'")
                    
                    while True:
                        try:
                            selection = input(f"\nSelect 1-{len(suggestions)+1} (or press Enter for original): ").strip()
                            if not selection:  # User pressed Enter
                                normalized_slug = predictor.normalize_input(user_input)
                                collection_name = None
                                confidence = 0.0
                                print(f"✅ Using original: {user_input}")
                                break
                            elif selection.isdigit():
                                sel_num = int(selection) - 1
                                if 0 <= sel_num < len(suggestions):
                                    normalized_slug, collection_name, confidence = suggestions[sel_num]
                                    print(f"✅ Selected: {collection_name}")
                                    break
                                elif sel_num == len(suggestions):
                                    normalized_slug = predictor.normalize_input(user_input)
                                    collection_name = None
                                    confidence = 0.0
                                    print(f"✅ Using original: {user_input}")
                                    break
                                else:
                                    print("Invalid selection. Please try again.")
                            else:
                                print("Please enter a number or press Enter.")
                        except ValueError:
                            print("Please enter a valid number.")
                    break
                else:
                    # No suggestions found
                    normalized_slug = predictor.normalize_input(user_input)
                    collection_name = None
                    print(f"🔄 No similar collections found. Analyzing: {user_input}")
            else:
                # No match at all
                normalized_slug = predictor.normalize_input(user_input)
                collection_name = None
                print(f"🔄 Analyzing: {normalized_slug}")
        
        if not normalized_slug:
            print("❌ Could not process input. Please try again.")
            continue
        
        # Proceed with prediction
        result = predictor.predict_collection(normalized_slug)
        
        print("\n" + "=" * 50)
        print("AUTHENTICITY ANALYSIS REPORT")
        print("=" * 50)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            print("\n💡 Troubleshooting:")
            print("   • Check if the collection exists on OpenSea")
            print("   • Try the exact collection name or OpenSea URL")
            print("   • Check your internet connection")
        else:
            prediction = result['prediction']
            emoji = "✅" if prediction == "Legitimate" else "⚠️"
            risk_score = result['risk_score']*100
            
            # Determine risk level category
            risk_level = "Low"
            if risk_score > 70:
                risk_level = "Very High"
            elif risk_score > 50:
                risk_level = "High"
            elif risk_score > 30:
                risk_level = "Medium"
            
            # Display collection info
            display_name = collection_name if collection_name else result['collection']
            
            # Header information
            print(f"{emoji} Collection: {display_name}")
            print(f"🔗 Slug: {result['collection']}")
            print(f"🌐 OpenSea: https://opensea.io/collection/{result['collection']}")
            print(f"📊 Prediction: {prediction}")
            print(f"🎯 Confidence: {result['confidence']['legitimate']*100:.1f}% legitimate")
            print(f"⚠️ Risk Score: {risk_score:.1f}% ({risk_level} Risk)")
            
            # Core metrics
            print("\n📈 KEY METRICS:")
            features = result['features_analyzed']
            print(f"   • Floor Price: {features['floor_price']} ETH")
            print(f"   • Total Volume: {features['total_volume']:,.0f} ETH")
            print(f"   • Owners: {features['num_owners']:,}")
            print(f"   • Market Cap: {features['market_cap']:,.0f} ETH")
            print(f"   • Average Price: {features['average_price']:.3f} ETH")
            
            # Trading metrics
            volume_per_owner = features['total_volume'] / max(features['num_owners'], 1)
            mc_volume_ratio = features['market_cap'] / max(features['total_volume'], 0.001)
            price_premium = features['average_price'] / max(features['floor_price'], 0.001)
            
            print("\n📊 TRADING ANALYSIS:")
            print(f"   • Volume per Owner: {volume_per_owner:.3f} ETH")
            print(f"   • Market Cap to Volume Ratio: {mc_volume_ratio:.2f}")
            print(f"   • Price Premium Ratio: {price_premium:.2f}")
            
            # Social and verification metrics
            print("\n🔍 TRUST INDICATORS:")
            print(f"   • Verified Collection: {'Yes ✓' if features['is_verified'] else 'No ✗'}")
            print(f"   • Social Media Presence:")
            print(f"     - Discord: {'Present ✓' if features['has_discord'] else 'Missing ✗'}")
            print(f"     - Twitter: {'Present ✓' if features['has_twitter'] else 'Missing ✗'}")
            print(f"   • Reddit Mentions: {features['reddit_mentions']}")
            print(f"   • Reddit Sentiment: {features['reddit_sentiment']:.2f} (0-1 scale)")
            
            # Risk assessment
            print("\n⚠️ RISK ASSESSMENT:")
            
            # Market risk based on liquidity
            liquidity = (features['total_volume'] * features['num_owners']) ** 0.5
            if liquidity < 10:
                print(f"   • Liquidity Risk: HIGH - Limited trading activity")
            else:
                print(f"   • Liquidity Risk: LOW - Healthy trading volume")
            
            # Ownership concentration risk
            ownership_concentration = 1 - (features['num_owners'] / max(features['total_supply'], 1))
            if ownership_concentration > 0.8:
                print(f"   • Ownership Concentration: HIGH ({ownership_concentration*100:.1f}%)")
            elif ownership_concentration > 0.5:
                print(f"   • Ownership Concentration: MEDIUM ({ownership_concentration*100:.1f}%)")
            else:
                print(f"   • Ownership Concentration: LOW ({ownership_concentration*100:.1f}%)")
            
            # Price risk
            if price_premium > 2:
                print(f"   • Price Premium Risk: HIGH ({price_premium:.2f}x floor)")
            elif price_premium > 1.5:
                print(f"   • Price Premium Risk: MEDIUM ({price_premium:.2f}x floor)")
            else:
                print(f"   • Price Premium Risk: LOW ({price_premium:.2f}x floor)")
            
            # Social media risk
            social_risk = "HIGH" if not (features['has_discord'] or features['has_twitter']) else \
                          "MEDIUM" if not (features['has_discord'] and features['has_twitter']) else "LOW"
            print(f"   • Social Media Risk: {social_risk}")
            
            print("\n🔮 PREDICTION CONFIDENCE:")
            if result['confidence']['legitimate'] > 0.9:
                print("   Very high confidence - strong legitimacy indicators")
            elif result['confidence']['legitimate'] > 0.7:
                print("   High confidence - multiple positive indicators")
            elif result['confidence']['legitimate'] > 0.5:
                print("   Moderate confidence - mixed indicators")
            else:
                print("   Low confidence - exercise caution")
                
            print("\n⚠️ DISCLAIMER: For informational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()
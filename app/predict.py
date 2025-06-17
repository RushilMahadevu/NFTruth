import json
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from data.opensea_collector import get_opensea_collection, get_opensea_collection_stats
from data.reddit_collector import RedditDataCollector
from models.model import NFTAuthenticityModel

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
    
    def setup_reddit_collector(self):
        """Initialize Reddit collector with credentials from .env"""
        try:
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT")
            
            if client_id and client_secret and user_agent:
                self.reddit_collector = RedditDataCollector(client_id, client_secret, user_agent)
                print("✅ Reddit collector initialized")
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

def main():
    """Test the predictor"""
    predictor = NFTPredictor()
    
    print("🤖 NFT Authenticity Predictor")
    print("=" * 40)
    
    while True:
        collection_slug = input("\nEnter NFT collection slug (or 'quit' to exit): ").strip()
        
        if collection_slug.lower() == 'quit':
            break
        
        if not collection_slug:
            print("Please enter a valid collection slug")
            continue
        
        result = predictor.predict_collection(collection_slug)
        
        print("\n" + "=" * 50)
        print("AUTHENTICITY ANALYSIS RESULT")
        print("=" * 50)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            prediction = result['prediction']
            emoji = "✅" if prediction == "Legitimate" else "⚠️"
            
            print(f"{emoji} Collection: {result['collection']}")
            print(f"📊 Prediction: {prediction}")
            print(f"🎯 Confidence: {result['confidence']['legitimate']*100:.1f}% legitimate")
            print(f"⚠️ Risk Score: {result['risk_score']*100:.1f}%")
            
            print("\n📈 Key Features:")
            features = result['features_analyzed']
            print(f"   Floor Price: {features['floor_price']} ETH")
            print(f"   Total Volume: {features['total_volume']:,.0f} ETH")
            print(f"   Owners: {features['num_owners']:,}")
            print(f"   Verified: {'Yes' if features['is_verified'] else 'No'}")
            print(f"   Social Media: Discord: {'Yes' if features['has_discord'] else 'No'}, Twitter: {'Yes' if features['has_twitter'] else 'No'}")

if __name__ == "__main__":
    main()
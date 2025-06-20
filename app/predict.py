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
from typing import Dict, List, Tuple

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
        self.prediction_history = []  # Track predictions for summary
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
    
    def _process_reddit_data(self, reddit_data: Dict) -> Dict:
        """Process raw Reddit data into required format"""
        total_posts = reddit_data.get('metadata', {}).get('total_posts', 0)
        total_comments = reddit_data.get('metadata', {}).get('total_comments', 0)
        
        if total_posts == 0:
            return {
                'total_mentions': 0,
                'avg_sentiment': 0.5,
                'total_engagement': 0
            }
        
        all_posts = reddit_data.get('all_posts', [])
        
        # Calculate engagement (upvotes + comments)
        total_engagement = 0
        sentiment_scores = []
        
        for post in all_posts:
            # Engagement = upvotes + comment count
            engagement = (post.get('ups', 0) + post.get('num_comments', 0))
            total_engagement += engagement
            
            # Simple sentiment analysis based on upvote ratio and score
            upvote_ratio = post.get('upvote_ratio', 0.5)
            score = post.get('score', 0)
            
            # Convert upvote ratio to sentiment (0.5 = neutral, 1.0 = very positive)
            # Also consider the score - higher scores generally indicate positive sentiment
            if score > 0:
                sentiment = min(1.0, upvote_ratio + (score / 1000) * 0.1)
            else:
                sentiment = max(0.0, upvote_ratio - 0.1)
            
            sentiment_scores.append(sentiment)
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        print(f"📊 Reddit metrics - Mentions: {total_posts}, Sentiment: {avg_sentiment:.2f}, Engagement: {total_engagement}")
        
        return {
            'total_mentions': total_posts,
            'avg_sentiment': avg_sentiment,
            'total_engagement': total_engagement
        }

    def get_confidence_tier(self, confidence: float) -> dict:
        """
        Get confidence tier information with badges and colors
        """
        if confidence >= 70:
            return {
                'tier': 'Trusted',
                'badge': '🟢',
                'color': 'green',
                'description': 'High confidence - Strong legitimacy indicators'
            }
        elif confidence >= 40:
            return {
                'tier': 'Caution',
                'badge': '🟡',
                'color': 'orange', 
                'description': 'Medium confidence - Mixed indicators, proceed carefully'
            }
        else:
            return {
                'tier': 'Suspicious',
                'badge': '🔴',
                'color': 'red',
                'description': 'Low confidence - Multiple risk factors detected'
            }
    
    def get_risk_level(self, risk_score: float) -> dict:
        """
        Get risk level information
        """
        if risk_score >= 70:
            return {'level': 'Very High', 'emoji': '🚨', 'color': 'red'}
        elif risk_score >= 50:
            return {'level': 'High', 'emoji': '⚠️', 'color': 'orange'}
        elif risk_score >= 30:
            return {'level': 'Medium', 'emoji': '⚡', 'color': 'yellow'}
        else:
            return {'level': 'Low', 'emoji': '✅', 'color': 'green'}
    
    def log_confidence_factors(self, features: dict, prediction_result: dict) -> dict:
        """
        Log specific factors that influenced confidence scoring
        """
        factors = {
            'positive_factors': [],
            'negative_factors': [],
            'caps_applied': [],
            'warnings': []
        }
        
        # Analyze factors based on features
        total_volume = features.get('total_volume', 0)
        is_verified = features.get('is_verified', 0)
        has_discord = features.get('has_discord', 0)
        has_twitter = features.get('has_twitter', 0)
        reddit_mentions = features.get('reddit_mentions', 0)
        reddit_sentiment = features.get('reddit_sentiment', 0.5)
        num_owners = features.get('num_owners', 0)
        floor_price = features.get('floor_price', 0)
        
        # Positive factors
        if total_volume > 100000:
            factors['positive_factors'].append(f"High trading volume ({total_volume:,.0f} ETH)")
        elif total_volume > 10000:
            factors['positive_factors'].append(f"Good trading volume ({total_volume:,.0f} ETH)")
        
        if is_verified:
            factors['positive_factors'].append("OpenSea verified collection ✓")
        
        if has_discord and has_twitter:
            factors['positive_factors'].append("Complete social media presence")
        elif has_discord or has_twitter:
            factors['positive_factors'].append("Partial social media presence")
        
        if reddit_mentions > 20:
            factors['positive_factors'].append(f"Strong Reddit community ({reddit_mentions} mentions)")
        elif reddit_mentions > 5:
            factors['positive_factors'].append(f"Active Reddit discussion ({reddit_mentions} mentions)")
        
        if reddit_sentiment > 0.7:
            factors['positive_factors'].append(f"Positive community sentiment ({reddit_sentiment:.2f})")
        
        if num_owners > 5000:
            factors['positive_factors'].append(f"Wide ownership distribution ({num_owners:,} owners)")
        
        # Negative factors
        if total_volume < 10:
            factors['negative_factors'].append(f"Very low trading volume ({total_volume:.2f} ETH)")
        elif total_volume < 100:
            factors['negative_factors'].append(f"Low trading volume ({total_volume:.2f} ETH)")
        
        if not is_verified:
            factors['negative_factors'].append("Not OpenSea verified")
        
        if not has_discord and not has_twitter:
            factors['negative_factors'].append("No social media presence")
        
        if reddit_mentions == 0:
            factors['negative_factors'].append("No Reddit community activity")
        elif reddit_mentions < 5:
            factors['negative_factors'].append(f"Limited Reddit activity ({reddit_mentions} mentions)")
        
        if reddit_sentiment < 0.4:
            factors['negative_factors'].append(f"Negative community sentiment ({reddit_sentiment:.2f})")
        
        if num_owners < 50:
            factors['negative_factors'].append(f"Concentrated ownership ({num_owners} owners)")
        
        if floor_price > 500:
            factors['negative_factors'].append(f"Extremely high floor price ({floor_price:.2f} ETH)")
        elif floor_price < 0.001:
            factors['negative_factors'].append(f"Suspiciously low floor price ({floor_price:.6f} ETH)")
        
        # Caps applied (reasons for limiting confidence)
        confidence = prediction_result.get('confidence', 0)
        
        if 'risk_flags' in prediction_result and prediction_result['risk_flags'] >= 3:
            factors['caps_applied'].append(f"Multiple red flags detected ({prediction_result['risk_flags']} flags)")
        
        if not is_verified and total_volume < 1000:
            factors['caps_applied'].append("Unverified collection with low volume")
        
        if not has_discord and not has_twitter and reddit_mentions == 0:
            factors['caps_applied'].append("Complete absence of community signals")
        
        if total_volume < 1 and reddit_mentions < 5:
            factors['caps_applied'].append("Minimal trading and community activity")
        
        # Warnings
        if confidence < 30:
            factors['warnings'].append("Very high risk - multiple red flags")
        elif confidence < 50:
            factors['warnings'].append("High risk - proceed with extreme caution")
        elif not prediction_result.get('minimum_criteria_met', True):
            factors['warnings'].append("Limited verification signals available")
        
        return factors
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all predictions made in this session
        """
        if not self.prediction_history:
            return "No predictions made in this session."
        
        summary_lines = [
            "🧠 PREDICTION SUMMARY REPORT",
            "=" * 60,
            f"Total Collections Analyzed: {len(self.prediction_history)}",
            ""
        ]
        
        # Add table header
        summary_lines.extend([
            f"{'Collection':<20} {'Prediction':<12} {'Confidence':<12} {'Risk':<8} {'Verdict':<8}",
            "-" * 60
        ])
        
        # Add each prediction
        for pred in self.prediction_history:
            collection = pred['collection'][:18] + '..' if len(pred['collection']) > 20 else pred['collection']
            prediction = pred['prediction']
            confidence = f"{pred['confidence']:.1f}%"
            risk_info = self.get_risk_level(pred['risk_score'])
            risk = risk_info['level'][:3]  # Abbreviated
            
            # Determine verdict based on confidence tier
            tier_info = self.get_confidence_tier(pred['confidence'])
            verdict = tier_info['badge']
            
            summary_lines.append(
                f"{collection:<20} {prediction:<12} {confidence:<12} {risk:<8} {verdict:<8}"
            )
        
        # Add statistics
        total_count = len(self.prediction_history)
        legitimate_count = sum(1 for p in self.prediction_history if p['prediction'] == 'Legitimate')
        suspicious_count = total_count - legitimate_count
        avg_confidence = sum(p['confidence'] for p in self.prediction_history) / total_count
        
        summary_lines.extend([
            "",
            "📊 SESSION STATISTICS:",
            f"   • Legitimate Collections: {legitimate_count} ({legitimate_count/total_count*100:.1f}%)",
            f"   • Suspicious Collections: {suspicious_count} ({suspicious_count/total_count*100:.1f}%)", 
            f"   • Average Confidence: {avg_confidence:.1f}%",
            "",
            "🔧 CONFIDENCE TIER BREAKDOWN:",
            f"   • 🟢 Trusted (70%+): {sum(1 for p in self.prediction_history if p['confidence'] >= 70)}",
            f"   • 🟡 Caution (40-69%): {sum(1 for p in self.prediction_history if 40 <= p['confidence'] < 70)}",
            f"   • 🔴 Suspicious (<40%): {sum(1 for p in self.prediction_history if p['confidence'] < 40)}",
        ])
        
        return "\n".join(summary_lines)

    def predict_collection(self, collection_slug: str) -> dict:
        """
        Predict authenticity for a given NFT collection with enhanced reporting
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        print(f"🔍 Analyzing collection: {collection_slug}")
        
        # Find the collection name from the slug
        collection_name = None
        for name, slug in COLLECTION_SLUGS.items():
            if slug == collection_slug:
                collection_name = name
                break
        
        try:
            # Collect data for this collection
            collection_data = get_opensea_collection(collection_slug)
            stats_data = get_opensea_collection_stats(collection_slug)
            
            # Collect Reddit data if available
            reddit_data = {}
            if self.reddit_collector:
                try:
                    # Use collection name if available, otherwise use slug
                    search_term = collection_name if collection_name else collection_slug
                    print(f"🔍 Searching Reddit for: '{search_term}'")
                    
                    reddit_raw_data = self.reddit_collector.collect_targeted_data(
                        query=search_term,
                        categories=['crypto_general', 'nft_specific', 'ethereum', 'trading_focused', 'tech_analysis', 'blockchain_general'],
                        time_filter='month',
                        posts_per_subreddit=15,
                        include_comments=False
                    )
                    
                    reddit_data = self._process_reddit_data(reddit_raw_data)
                except Exception as e:
                    print(f"⚠️ Reddit data collection failed: {e}")
                    reddit_data = {'total_mentions': 0, 'avg_sentiment': 0.5, 'total_engagement': 0}
            else:
                print("⚠️ Reddit collector not available")
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
            
            # Calculate enhanced metrics
            confidence = float(probability[1] * 100) if len(probability) > 1 else 0.0
            risk_score = float((1 - probability[1]) * 100) if len(probability) > 1 else 100.0
            
            # Get tier and risk information
            tier_info = self.get_confidence_tier(confidence)
            risk_info = self.get_risk_level(risk_score)
            
            # Log confidence factors
            confidence_factors = self.log_confidence_factors(features, {
                'confidence': confidence,
                'risk_flags': 0,  # You can calculate this based on features
                'minimum_criteria_met': True  # You can calculate this based on features
            })
            
            # Prepare enhanced result
            result = {
                "collection": collection_slug,
                "collection_name": collection_name,
                "prediction": "Legitimate" if prediction == 1 else "Suspicious",
                "confidence": confidence,
                "confidence_tier": tier_info,
                "risk_score": risk_score,
                "risk_level": risk_info,
                "features_analyzed": features,
                "confidence_factors": confidence_factors,
                "timestamp": pd.Timestamp.now().isoformat(),
                "opensea_url": f"https://opensea.io/collection/{collection_slug}"
            }
            
            # Add to prediction history
            self.prediction_history.append({
                'collection': collection_name or collection_slug,
                'prediction': result['prediction'],
                'confidence': confidence,
                'risk_score': risk_score,
                'timestamp': result['timestamp']
            })
            
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
    """Test the predictor with enhanced reporting"""
    predictor = NFTPredictor()
    
    print("\n🤖 NFT Authenticity Predictor 🤖")
    print("=" * 40)
    print("\n💡 Enter collection name, slug, or OpenSea URL")
    print("\nExamples: 🧠 'CryptoPunks' | 😍 'bored-ape-yacht-club' | 🔗 'https://opensea.io/collection/azuki'")
    print("\nNew Commands:")
    print("  • 'summary' - Show session summary")
    print("  • 'help' - Show examples")
    print("  • 'quit' - Exit")
    
    while True:
        user_input = input("\n👋 What's the NFT collection? ").strip()
        
        if user_input.lower() == 'quit':
            # Show final summary before quitting
            print("\n" + predictor.generate_summary_report())
            break
        
        if user_input.lower() == 'summary':
            print("\n" + predictor.generate_summary_report())
            continue
        
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
                print(f"\n🔍 Did you mean '{collection_name}' ({normalized_slug})? 🤔 ({confidence*100:.0f}% match)")
                
                while True:
                    choice = input("👍 y = yes | 👎 n = no | 💡 s = suggestions: ").strip().lower()
                    
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
                            print("\n🔍 Similar collections found 🧩:")
                            for i, (slug, name, conf) in enumerate(suggestions, 1):
                                print(f"   {i}. {name} ({slug}) - {conf*100:.0f}% match")
                            print(f"   {len(suggestions)+1}. 📝 Use original input: '{user_input}'")
                            
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
                        print(f"   {i}. {name} ({slug}) - 📈 {conf*100:.0f}% match")
                    print(f"   {len(suggestions)+1}. 📝 Use original input: '{user_input}'")
                    
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
        
        print("\n" + "=" * 70)
        print("🎉 AUTHENTICITY ANALYSIS COMPLETE 🚀")
        print("=" * 70)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            # Enhanced display with tiers and detailed factors
            prediction = result['prediction']
            confidence = result['confidence']
            tier_info = result['confidence_tier']
            risk_info = result['risk_level']
            
            # Header with tier badge
            print(f"{tier_info['badge']} Collection: {result.get('collection_name', result['collection'])}")
            print(f"🔗 Slug: {result['collection']}")
            print(f"🌐 OpenSea: {result['opensea_url']}")
            print(f"📊 Prediction: {prediction}")
            print(f"🎯 Confidence: {confidence:.1f}% ({tier_info['tier']})")
            print(f"⚠️ Risk Level: {risk_info['level']} {risk_info['emoji']} ({result['risk_score']:.1f}%)")
            print(f"💡 {tier_info['description']}")
            
            # Confidence factors breakdown
            factors = result['confidence_factors']
            
            if factors['positive_factors']:
                print(f"\n✅ POSITIVE INDICATORS ({len(factors['positive_factors'])}):")
                for factor in factors['positive_factors']:
                    print(f"   • {factor}")
            
            if factors['negative_factors']:
                print(f"\n❌ RISK FACTORS ({len(factors['negative_factors'])}):")
                for factor in factors['negative_factors']:
                    print(f"   • {factor}")
            
            if factors['caps_applied']:
                print(f"\n🔒 CONFIDENCE LIMITATIONS:")
                for cap in factors['caps_applied']:
                    print(f"   • {cap}")
            
            if factors['warnings']:
                print(f"\n⚠️ WARNINGS:")
                for warning in factors['warnings']:
                    print(f"   • {warning}")
            
            # Core metrics (existing code)
            features = result['features_analyzed']
            print("\n📈 KEY METRICS:")
            print(f"   • Floor Price: {features['floor_price']} ETH")
            print(f"   • Total Volume: {features['total_volume']:,.0f} ETH")
            print(f"   • Owners: {features['num_owners']:,}")
            print(f"   • Market Cap: {features['market_cap']:,.0f} ETH")
            print(f"   • Average Price: {features['average_price']:.3f} ETH")
            
            print("\n⚠️ DISCLAIMER: For informational purposes only. Not financial advice.")
            
            # Mini summary for this prediction
            print(f"\n📝 Added to session summary. Type 'summary' to see all {len(predictor.prediction_history)} predictions.")

if __name__ == "__main__":
    main()
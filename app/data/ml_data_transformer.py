import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from .opensea_collector import get_opensea_collection, get_opensea_collection_stats
from .etherscan_collector import get_etherscan_wallet_tx
from .reddit_collector import RedditDataCollector
import os
from dotenv import load_dotenv
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

@dataclass
class NFTScamFeatures:
    """Data class to hold extracted features for ML training"""
    # Collection metadata features
    collection_age_days: float
    total_supply: int
    floor_price_eth: float
    market_cap_eth: float
    holder_count: int
    total_volume_eth: float
    
    # Trading pattern features
    avg_daily_volume: float
    volume_volatility: float
    price_volatility: float
    whale_concentration: float
    wash_trading_score: float
    
    # Social sentiment features
    reddit_mention_count: int
    reddit_sentiment_score: float
    reddit_enthusiasm_score: float
    reddit_warning_mentions: int
    social_hype_ratio: float
    
    # Blockchain analysis features
    creator_wallet_age_days: float
    creator_transaction_count: int
    creator_balance_eth: float
    suspicious_transaction_patterns: float
    mint_distribution_score: float
    
    # Risk indicators
    rug_pull_risk_score: float
    pump_dump_risk_score: float
    overall_scam_probability: float
    
    # Labels for supervised learning
    is_verified: bool
    is_scam: bool  # This would be manually labeled for training data

class MLDataTransformer:
    def __init__(self):
        self.reddit_collector = RedditDataCollector(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        
        # Initialize VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Scam indicator keywords
        self.scam_keywords = [
            'scam', 'rugpull', 'rug pull', 'avoid', 'warning', 'fake', 'fraud',
            'stolen', 'phishing', 'honeypot', 'pump and dump', 'exit scam'
        ]
        
        self.hype_keywords = [
            'moon', 'diamond hands', 'hodl', 'to the moon', 'lfg', 'gm',
            'wagmi', 'alpha', 'degen', 'ape in', 'fomo'
        ]

    def transform_nft_data(self, collection_slug: str, creator_address: Optional[str] = None) -> NFTScamFeatures:
        """
        Transform raw data from all sources into ML-ready features
        
        Args:
            collection_slug: OpenSea collection identifier
            creator_address: Creator's Ethereum address (optional)
        
        Returns:
            NFTScamFeatures object with extracted features
        """
        print(f"Transforming data for collection: {collection_slug}")
        
        # Collect raw data from all sources
        opensea_data = get_opensea_collection(collection_slug)
        reddit_data = self._collect_reddit_data(collection_slug)
        etherscan_data = {}
        if creator_address:
            etherscan_data = get_etherscan_wallet_tx(creator_address)
        
        # Extract features
        collection_features = self._extract_collection_features(opensea_data)
        trading_features = self._extract_trading_features(opensea_data, etherscan_data)
        social_features = self._extract_social_features(reddit_data, collection_slug)
        blockchain_features = self._extract_blockchain_features(etherscan_data, creator_address)
        risk_features = self._calculate_risk_scores(
            collection_features, trading_features, social_features, blockchain_features
        )
        
        # Combine all features  
        return NFTScamFeatures(
            **collection_features,
            **trading_features,
            **social_features,
            **blockchain_features,
            **risk_features,
            is_verified=opensea_data.get('safelist_request_status') == 'verified',
            is_scam=False  # This needs to be manually labeled for training
        )

    def _collect_reddit_data(self, collection_slug: str) -> Dict:
        """Collect Reddit data for the NFT collection - SIMPLIFIED to avoid loops"""
        print(f"Collecting Reddit data for: {collection_slug}")
        
        # Use only one query to avoid duplication
        query = collection_slug.replace('-', ' ')
        
        try:
            # Simplified data collection - only search in NFT-specific subreddits
            reddit_data = self.reddit_collector.collect_targeted_data(
                query=query,
                categories=['nft_specific'],  # Only NFT subreddits to reduce API calls
                time_filter='week',  # Shorter time window
                posts_per_subreddit=10,  # Fewer posts per subreddit
                include_comments=False,  # Skip comments to speed up
                comment_limit=0
            )
            
            all_reddit_data = {
                'posts': reddit_data.get('all_posts', []),
                'comments': [],
                'total_mentions': len(reddit_data.get('all_posts', []))
            }
            
            print(f"Found {all_reddit_data['total_mentions']} Reddit mentions")
            return all_reddit_data
            
        except Exception as e:
            print(f"Error collecting Reddit data: {e}")
            return {
                'posts': [],
                'comments': [],
                'total_mentions': 0
            }

    def _extract_collection_features(self, opensea_data: Dict) -> Dict:
        """Extract collection-level features from OpenSea data"""
        if not opensea_data:
            return {
                'collection_age_days': 0,
                'total_supply': 0,
                'floor_price_eth': 0,
                'market_cap_eth': 0,
                'holder_count': 0,
                'total_volume_eth': 0
            }
        
        # Calculate collection age - FIX TIMEZONE ISSUE
        created_date = opensea_data.get('created_date', '')
        collection_age = 0
        if created_date:
            try:
                # Handle timezone-aware datetime
                if created_date.endswith('Z'):
                    created_date = created_date[:-1] + '+00:00'
                created = datetime.fromisoformat(created_date)
                
                # Make sure both datetimes are timezone-aware
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                
                now = datetime.now(timezone.utc)
                collection_age = (now - created).days
            except Exception as e:
                print(f"Error parsing created_date: {e}")
                collection_age = 0
        
        stats = opensea_data.get('stats', {})
        
        return {
            'collection_age_days': float(collection_age),
            'total_supply': int(stats.get('total_supply', 0)),
            'floor_price_eth': float(stats.get('floor_price', 0) or 0),
            'market_cap_eth': float(stats.get('market_cap', 0) or 0),
            'holder_count': int(stats.get('num_owners', 0)),
            'total_volume_eth': float(stats.get('total_volume', 0) or 0)
        }

    def _extract_trading_features(self, opensea_data: Dict, etherscan_data: Dict) -> Dict:
        """Extract trading pattern features"""
        if not opensea_data:
            return {
                'avg_daily_volume': 0,
                'volume_volatility': 0,
                'price_volatility': 0,
                'whale_concentration': 0,
                'wash_trading_score': 0
            }
        
        stats = opensea_data.get('stats', {})
        
        # Calculate daily volume (rough estimate)
        total_volume = float(stats.get('total_volume', 0) or 0)
        collection_age = max(1, self._get_collection_age_days(opensea_data))
        avg_daily_volume = total_volume / collection_age
        
        # Estimate volatility based on volume changes (simplified)
        one_day_volume = float(stats.get('one_day_volume', 0) or 0)
        seven_day_volume = float(stats.get('seven_day_volume', 0) or 0)
        thirty_day_volume = float(stats.get('thirty_day_volume', 0) or 0)
        
        volumes = [v for v in [one_day_volume, seven_day_volume/7, thirty_day_volume/30] if v > 0]
        volume_volatility = np.std(volumes) if len(volumes) > 1 else 0
        
        # Price volatility (simplified using floor price changes)
        one_day_change = float(stats.get('one_day_change', 0) or 0)
        seven_day_change = float(stats.get('seven_day_change', 0) or 0)
        thirty_day_change = float(stats.get('thirty_day_change', 0) or 0)
        
        price_changes = [one_day_change, seven_day_change, thirty_day_change]
        price_volatility = np.std(price_changes) if any(price_changes) else 0
        
        # Whale concentration (simplified)
        total_supply = int(stats.get('total_supply', 1))
        num_owners = int(stats.get('num_owners', 1))
        whale_concentration = 1 - (num_owners / max(total_supply, 1))
        
        # Wash trading score (based on volume to unique traders ratio)
        wash_trading_score = self._calculate_wash_trading_score(etherscan_data)
        
        return {
            'avg_daily_volume': float(avg_daily_volume),
            'volume_volatility': float(volume_volatility),
            'price_volatility': float(price_volatility),
            'whale_concentration': float(whale_concentration),
            'wash_trading_score': float(wash_trading_score)
        }

    def _extract_social_features(self, reddit_data: Dict, collection_name: str) -> Dict:
        """Extract social sentiment features from Reddit data"""
        posts = reddit_data.get('posts', [])
        comments = reddit_data.get('comments', [])
        
        total_mentions = len(posts)
        
        # Analyze sentiment
        all_text = []
        for post in posts:
            all_text.append(post.get('title', '') + ' ' + post.get('selftext', ''))
        for comment in comments:
            if comment:  # Skip None comments
                all_text.append(comment.get('body', ''))
        
        sentiment_score = self._analyze_sentiment(all_text)
        enthusiasm_score = self._calculate_enthusiasm_score(all_text)
        warning_mentions = self._count_warning_mentions(all_text)
        
        # Calculate social hype ratio
        total_engagement = sum(post.get('score', 0) + post.get('num_comments', 0) for post in posts)
        social_hype_ratio = total_engagement / max(total_mentions, 1)
        
        return {
            'reddit_mention_count': total_mentions,
            'reddit_sentiment_score': float(sentiment_score),
            'reddit_enthusiasm_score': float(enthusiasm_score),
            'reddit_warning_mentions': warning_mentions,
            'social_hype_ratio': float(social_hype_ratio)
        }

    def _extract_blockchain_features(self, etherscan_data: Dict, creator_address: Optional[str]) -> Dict:
        """Extract blockchain analysis features"""
        if not etherscan_data or not creator_address:
            return {
                'creator_wallet_age_days': 0,
                'creator_transaction_count': 0,
                'creator_balance_eth': 0,
                'suspicious_transaction_patterns': 0,
                'mint_distribution_score': 0
            }
        
        transactions = etherscan_data.get('result', [])
        
        # Calculate wallet age
        if transactions:
            oldest_tx = min(transactions, key=lambda x: int(x.get('timeStamp', 0)))
            wallet_creation = datetime.fromtimestamp(int(oldest_tx.get('timeStamp', 0)))
            wallet_age_days = (datetime.now() - wallet_creation).days
        else:
            wallet_age_days = 0
        
        # Transaction count
        tx_count = len(transactions)
        
        # Current balance (from latest transaction)
        balance_wei = 0
        if transactions:
            # This is simplified - in reality you'd need current balance API call
            latest_tx = max(transactions, key=lambda x: int(x.get('timeStamp', 0)))
            balance_wei = int(latest_tx.get('value', 0))
        
        balance_eth = balance_wei / 10**18
        
        # Suspicious patterns
        suspicious_score = self._analyze_suspicious_patterns(transactions)
        
        # Mint distribution (simplified)
        mint_score = self._calculate_mint_distribution_score(transactions)
        
        return {
            'creator_wallet_age_days': float(wallet_age_days),
            'creator_transaction_count': tx_count,
            'creator_balance_eth': float(balance_eth),
            'suspicious_transaction_patterns': float(suspicious_score),
            'mint_distribution_score': float(mint_score)
        }

    def _calculate_risk_scores(self, collection_features: Dict, trading_features: Dict, 
                             social_features: Dict, blockchain_features: Dict) -> Dict:
        """Calculate composite risk scores"""
        
        # Rug pull risk factors
        rug_pull_factors = [
            1 - min(collection_features['collection_age_days'] / 365, 1),  # New collections riskier
            min(blockchain_features['suspicious_transaction_patterns'], 1),
            1 - min(collection_features['holder_count'] / 1000, 1),  # Few holders riskier
            min(social_features['reddit_warning_mentions'] / 10, 1)
        ]
        rug_pull_risk = np.mean(rug_pull_factors)
        
        # Pump and dump risk factors
        pump_dump_factors = [
            min(trading_features['volume_volatility'] / 100, 1),
            min(trading_features['price_volatility'] / 100, 1),
            min(social_features['social_hype_ratio'] / 1000, 1),
            trading_features['wash_trading_score']
        ]
        pump_dump_risk = np.mean(pump_dump_factors)
        
        # Overall scam probability (weighted combination)
        overall_scam_prob = (0.4 * rug_pull_risk + 0.3 * pump_dump_risk + 
                           0.2 * min(social_features['reddit_warning_mentions'] / 5, 1) +
                           0.1 * trading_features['whale_concentration'])
        
        return {
            'rug_pull_risk_score': float(rug_pull_risk),
            'pump_dump_risk_score': float(pump_dump_risk),
            'overall_scam_probability': float(min(overall_scam_prob, 1.0))
        }

    def _get_collection_age_days(self, opensea_data: Dict) -> float:
        """Helper to get collection age in days"""
        created_date = opensea_data.get('created_date', '')
        if not created_date:
            return 0
        
        try:
            if created_date.endswith('Z'):
                created_date = created_date[:-1] + '+00:00'
            created = datetime.fromisoformat(created_date)
            
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            
            now = datetime.now(timezone.utc)
            return (now - created).days
        except Exception:
            return 0

    def _calculate_wash_trading_score(self, etherscan_data: Dict) -> float:
        """Calculate wash trading score based on transaction patterns"""
        if not etherscan_data:
            return 0
        
        transactions = etherscan_data.get('result', [])
        if not transactions:
            return 0
        
        # Look for circular transactions (simplified)
        from_addresses = [tx.get('from', '') for tx in transactions]
        to_addresses = [tx.get('to', '') for tx in transactions]
        
        # Count back-and-forth transactions
        circular_count = 0
        for i, tx in enumerate(transactions):
            from_addr = tx.get('from', '')
            to_addr = tx.get('to', '')
            
            # Look for reverse transaction within next 10 transactions
            for j in range(i+1, min(i+11, len(transactions))):
                next_tx = transactions[j]
                if (next_tx.get('from', '') == to_addr and 
                    next_tx.get('to', '') == from_addr):
                    circular_count += 1
                    break
        
        return min(circular_count / max(len(transactions), 1), 1.0)

    def _analyze_sentiment(self, texts: List[str]) -> float:
        """Use VADER sentiment analysis for more accurate sentiment scoring"""
        if not texts:
            return 0.5
        
        scores = []
        for text in texts:
            if text:
                score = self.sentiment_analyzer.polarity_scores(text)
                scores.append(score['compound'])  # Compound score ranges from -1 to 1
        
        if not scores:
            return 0.5
        
        # Convert from [-1, 1] to [0, 1]
        return (np.mean(scores) + 1) / 2

    def _calculate_enthusiasm_score(self, texts: List[str]) -> float:
        """Calculate enthusiasm/hype score"""
        if not texts:
            return 0
        
        enthusiasm_indicators = 0
        total_chars = 0
        
        for text in texts:
            if not text:
                continue
            total_chars += len(text)
            
            # Count enthusiasm indicators
            enthusiasm_indicators += text.count('!')
            enthusiasm_indicators += text.count('🚀')
            enthusiasm_indicators += text.count('💎')
            enthusiasm_indicators += len(re.findall(r'[A-Z]{2,}', text))  # ALL CAPS words
            
            # Count hype keywords
            for keyword in self.hype_keywords:
                enthusiasm_indicators += text.lower().count(keyword)
        
        return enthusiasm_indicators / max(total_chars / 100, 1)  # Per 100 characters

    def _count_warning_mentions(self, texts: List[str]) -> int:
        """Count mentions of scam warnings"""
        warning_count = 0
        for text in texts:
            if not text:
                continue
            text_lower = text.lower()
            for keyword in self.scam_keywords:
                warning_count += text_lower.count(keyword)
        return warning_count

    def _analyze_suspicious_patterns(self, transactions: List[Dict]) -> float:
        """Analyze transactions for suspicious patterns"""
        if not transactions:
            return 0
        
        suspicious_score = 0
        total_transactions = len(transactions)
        
        # Check for patterns like rapid sequential transactions
        rapid_tx_count = 0
        for i in range(1, len(transactions)):
            current_time = int(transactions[i].get('timeStamp', 0))
            prev_time = int(transactions[i-1].get('timeStamp', 0))
            
            # Transactions within 1 minute of each other
            if current_time - prev_time < 60:
                rapid_tx_count += 1
        
        rapid_tx_ratio = rapid_tx_count / max(total_transactions, 1)
        
        # Check for unusual gas prices (very high or very low)
        gas_prices = [int(tx.get('gasPrice', 0)) for tx in transactions if tx.get('gasPrice')]
        if gas_prices:
            avg_gas = np.mean(gas_prices)
            unusual_gas_count = sum(1 for price in gas_prices 
                                  if price > avg_gas * 3 or price < avg_gas * 0.3)
            unusual_gas_ratio = unusual_gas_count / len(gas_prices)
        else:
            unusual_gas_ratio = 0
        
        suspicious_score = (rapid_tx_ratio + unusual_gas_ratio) / 2
        return min(suspicious_score, 1.0)

    def _calculate_mint_distribution_score(self, transactions: List[Dict]) -> float:
        """Calculate how evenly mints were distributed (higher score = more even)"""
        if not transactions:
            return 0.5
        
        # This is simplified - would need to identify actual mint transactions
        # For now, analyze transaction value distribution
        values = [int(tx.get('value', 0)) for tx in transactions if int(tx.get('value', 0)) > 0]
        
        if not values:
            return 0.5
        
        # Calculate distribution evenness using coefficient of variation
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if mean_value == 0:
            return 0.5
        
        cv = std_value / mean_value
        # Lower coefficient of variation = more even distribution
        evenness_score = max(0, 1 - min(cv, 1))
        
        return evenness_score

    def create_training_dataset(self, collection_slugs: List[str], 
                              creator_addresses: Optional[List[str]] = None,
                              labels: Optional[List[bool]] = None) -> List[NFTScamFeatures]:
        """
        Create a dataset for ML training
        
        Args:
            collection_slugs: List of OpenSea collection slugs
            creator_addresses: List of creator addresses (optional)
            labels: List of scam labels (True = scam, False = legitimate)
        
        Returns:
            List of NFTScamFeatures objects
        """
        dataset = []
        
        if creator_addresses is None:
            creator_addresses = [None] * len(collection_slugs)
        
        if labels is None:
            labels = [False] * len(collection_slugs)  # Default to not scam
        
        for i, slug in enumerate(collection_slugs):
            try:
                features = self.transform_nft_data(
                    collection_slug=slug,
                    creator_address=creator_addresses[i] if i < len(creator_addresses) else None
                )
                
                # Set the label if provided
                if i < len(labels):
                    features.is_scam = labels[i]
                
                dataset.append(features)
                print(f"Processed {i+1}/{len(collection_slugs)}: {slug}")
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {slug}: {e}")
                continue
        
        return dataset

    def save_dataset(self, dataset: List[NFTScamFeatures], filename: str):
        """Save dataset to JSON file"""
        data = []
        for features in dataset:
            data.append(features.__dict__)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Dataset saved to {filename}")

    def extract_opensea_features(self, collection_slug: str) -> dict:
        """Extract features from OpenSea collection data"""
        try:
            # Get basic collection info
            collection_data = get_opensea_collection(collection_slug)
            # Get collection statistics
            stats_data = get_opensea_collection_stats(collection_slug)
            
            features = {}
            
            # Extract from collection data
            if collection_data:
                features.update({
                    'total_supply': collection_data.get('total_supply', 0),
                    'is_verified': 1 if collection_data.get('safelist_status') == 'verified' else 0,
                    'has_discord': 1 if collection_data.get('discord_url') else 0,
                    'has_twitter': 1 if collection_data.get('twitter_username') else 0,
                    'trait_offers_enabled': 1 if collection_data.get('trait_offers_enabled', False) else 0,
                    'collection_offers_enabled': 1 if collection_data.get('collection_offers_enabled', False) else 0,
                })
            
            # Extract from stats data (nested under 'total' key)
            if stats_data and 'total' in stats_data:
                total_stats = stats_data['total']
                features.update({
                    'floor_price': float(total_stats.get('floor_price', 0) or 0),
                    'market_cap': float(total_stats.get('market_cap', 0) or 0),
                    'total_volume': float(total_stats.get('volume', 0) or 0),  # Note: 'volume' not 'total_volume'
                    'num_owners': int(total_stats.get('num_owners', 0) or 0),
                    'average_price': float(total_stats.get('average_price', 0) or 0),
                })
            
            return features
            
        except Exception as e:
            print(f"Error extracting OpenSea features for {collection_slug}: {e}")
            return {}

    def transform_collection_data(self, collection_slug: str) -> List[Dict]:
        """
        Transform collection data into ML-ready format
        Returns a list of records for the collection
        """
        try:
            # Extract features from different sources
            opensea_features = self.extract_opensea_features(collection_slug)
            reddit_features = self.extract_reddit_features(collection_slug)
            
            # Combine all features
            combined_features = {
                **opensea_features,
                **reddit_features
            }
            
            # Create a record for this collection
            record = {
                'collection_slug': collection_slug,
                'features': combined_features,
                'timestamp': datetime.now().isoformat()
            }
            
            return [record]  # Return as list for consistency
            
        except Exception as e:
            print(f"Error transforming data for {collection_slug}: {e}")
            return []

    def extract_reddit_features(self, collection_slug: str) -> dict:
        """Extract features from Reddit data"""
        try:
            reddit_data = self._collect_reddit_data(collection_slug)
            
            return {
                'reddit_mentions': reddit_data.get('total_mentions', 0),
                'reddit_sentiment': 0.5,  # Placeholder - you can enhance this
                'reddit_engagement': len(reddit_data.get('posts', []))
            }
            
        except Exception as e:
            print(f"Error extracting Reddit features for {collection_slug}: {e}")
            return {
                'reddit_mentions': 0,
                'reddit_sentiment': 0.5,
                'reddit_engagement': 0
            }

    def debug_opensea_response(self, collection_slug: str):
        """Debug function to see what OpenSea API is returning"""
        print(f"\n{'='*50}")
        print(f"DEBUGGING OPENSEA API FOR: {collection_slug}")
        print(f"{'='*50}")
        
        try:
            # Test collection endpoint
            collection_data = get_opensea_collection(collection_slug)
            print(f"Collection data keys: {list(collection_data.keys()) if collection_data else 'None'}")
            
            # Test stats endpoint
            stats_data = get_opensea_collection_stats(collection_slug)
            print(f"Stats data keys: {list(stats_data.keys()) if stats_data else 'None'}")
            print(f"Stats data: {stats_data}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

# # Example usage
# if __name__ == "__main__":
#     transformer = MLDataTransformer()
    
#     # Debug each collection first
#     test_collections = ["boredapeyachtclub", "cryptopunks", "azuki"]
#     for collection in test_collections:
#         transformer.debug_opensea_response(collection)

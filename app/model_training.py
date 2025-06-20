import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

class RuleBasedNFTAuthenticityModel:
    """
    Rule-based NFT authenticity assessment system
    More reliable than ML for small datasets
    """
    
    def __init__(self):
        self.model_type = "Rule-Based"
        self.version = "1.0"
        self.created_at = datetime.now().isoformat()
        
        # Known scam indicators
        self.scam_keywords = [
            'fake', 'scam', 'copy', 'clone', 'knockoff',
            'quick', 'easy', 'money', 'pump', 'moon'
        ]
        
        # Known legitimate collections (high confidence)
        self.known_legitimate = {
            'cryptopunks': 0.95,
            'boredapeyachtclub': 0.95,
            'mutant-ape-yacht-club': 0.92,
            'azuki': 0.90,
            'doodles-official': 0.88,
            'clonex': 0.87,
            'meebits': 0.85,
            'world-of-women': 0.82,
            'pudgypenguins': 0.85,
            'coolcats': 0.80
        }
    
    def predict_collection(self, collection_features: Dict, collection_slug: str = None) -> Dict:
        """
        Rule-based authenticity prediction
        """
        
        # Extract key metrics
        total_volume = collection_features.get('total_volume', 0)
        is_verified = collection_features.get('is_verified', 0)
        reddit_mentions = collection_features.get('reddit_mentions', 0)
        reddit_sentiment = collection_features.get('reddit_sentiment', 0.5)
        has_discord = collection_features.get('has_discord', 0)
        has_twitter = collection_features.get('has_twitter', 0)
        num_owners = collection_features.get('num_owners', 0)
        floor_price = collection_features.get('floor_price', 0)
        total_supply = collection_features.get('total_supply', 0)
        reddit_engagement = collection_features.get('reddit_engagement', 0)
        
        # Check if it's a known collection first
        if collection_slug and collection_slug in self.known_legitimate:
            confidence_score = self.known_legitimate[collection_slug]
        else:
            # Start with neutral base
            confidence_score = 0.5
            
            # 1. VOLUME ANALYSIS (Strongest signal - up to ±40%)
            if total_volume >= 1000000:  # 1M+ ETH - legendary
                confidence_score += 0.40
            elif total_volume >= 500000:  # 500K+ ETH - major
                confidence_score += 0.35
            elif total_volume >= 100000:  # 100K+ ETH - established
                confidence_score += 0.25
            elif total_volume >= 50000:   # 50K+ ETH - solid
                confidence_score += 0.20
            elif total_volume >= 10000:   # 10K+ ETH - decent
                confidence_score += 0.15
            elif total_volume >= 1000:    # 1K+ ETH - active
                confidence_score += 0.08
            elif total_volume >= 100:     # 100+ ETH - minimal
                confidence_score += 0.03
            elif total_volume >= 10:      # 10+ ETH - very low
                confidence_score -= 0.10
            elif total_volume >= 1:       # 1+ ETH - almost none
                confidence_score -= 0.20
            else:                          # No volume - major red flag
                confidence_score -= 0.30
            
            # 2. VERIFICATION STATUS (±20%)
            if is_verified:
                confidence_score += 0.20
            else:
                confidence_score -= 0.25
            
            # 3. SOCIAL MEDIA PRESENCE (±15%)
            if has_discord and has_twitter:
                confidence_score += 0.12
            elif has_discord or has_twitter:
                confidence_score += 0.06
            else:
                confidence_score -= 0.15
            
            # 4. REDDIT ENGAGEMENT (±15%)
            if reddit_mentions >= 100:
                confidence_score += 0.15
            elif reddit_mentions >= 50:
                confidence_score += 0.12
            elif reddit_mentions >= 20:
                confidence_score += 0.08
            elif reddit_mentions >= 5:
                confidence_score += 0.04
            elif reddit_mentions == 0:
                confidence_score -= 0.12
            
            # 5. REDDIT SENTIMENT (±10%)
            if reddit_sentiment >= 0.8:
                confidence_score += 0.10
            elif reddit_sentiment >= 0.7:
                confidence_score += 0.06
            elif reddit_sentiment >= 0.6:
                confidence_score += 0.03
            elif reddit_sentiment <= 0.3:
                confidence_score -= 0.10
            elif reddit_sentiment <= 0.4:
                confidence_score -= 0.05
            
            # 6. OWNERSHIP DISTRIBUTION (±10%)
            if num_owners >= 10000:
                confidence_score += 0.10
            elif num_owners >= 5000:
                confidence_score += 0.08
            elif num_owners >= 1000:
                confidence_score += 0.05
            elif num_owners >= 100:
                confidence_score += 0.02
            elif num_owners < 10:
                confidence_score -= 0.15
            elif num_owners < 50:
                confidence_score -= 0.08
            
            # 7. PRICE SANITY CHECKS (±10%)
            if floor_price > 1000:  # Extremely high - suspicious
                confidence_score -= 0.10
            elif floor_price > 100:  # Very high - could be manipulation
                confidence_score -= 0.05
            elif floor_price < 0.00001:  # Essentially free - suspicious
                confidence_score -= 0.15
            elif floor_price < 0.001:  # Very cheap - somewhat suspicious
                confidence_score -= 0.08
            elif 0.1 <= floor_price <= 50:  # Reasonable range
                confidence_score += 0.03
            
            # 8. SUPPLY ANALYSIS (±5%)
            if total_supply > 100000:  # Very large supply - often suspicious
                confidence_score -= 0.08
            elif total_supply > 50000:  # Large supply
                confidence_score -= 0.03
            elif 1000 <= total_supply <= 10000:  # Standard range
                confidence_score += 0.03
            elif total_supply < 100:  # Very limited - could be exclusive or suspicious
                confidence_score -= 0.02
            
            # 9. ENGAGEMENT RATIO (±5%)
            if reddit_engagement > 50000:
                confidence_score += 0.05
            elif reddit_engagement > 10000:
                confidence_score += 0.03
            elif reddit_engagement == 0 and reddit_mentions > 0:
                confidence_score -= 0.03  # Mentions but no engagement - suspicious
        
        # 10. SCAM KEYWORD CHECK (Collection name/slug)
        if collection_slug:
            slug_lower = collection_slug.lower()
            for keyword in self.scam_keywords:
                if keyword in slug_lower:
                    confidence_score -= 0.25  # Heavy penalty for scam keywords
                    break
        
        # 11. SPECIAL PATTERNS
        # Zero volume but verified - suspicious (could be fake verification)
        if total_volume == 0 and is_verified:
            confidence_score -= 0.15
        
        # No social media AND no Reddit - major red flag
        if not has_discord and not has_twitter and reddit_mentions == 0:
            confidence_score -= 0.10
        
        # Very high concentration (few owners for supply)
        if num_owners > 0 and total_supply > 0:
            ownership_ratio = num_owners / total_supply
            if ownership_ratio < 0.01:  # Less than 1% of supply has unique owners
                confidence_score -= 0.15
        
        # Apply realistic bounds
        final_confidence = max(0.05, min(0.98, confidence_score))
        
        # Determine prediction
        prediction = 'Legitimate' if final_confidence >= 0.5 else 'Suspicious'
        risk_score = 1.0 - final_confidence
        
        # Calculate confidence level description
        if final_confidence >= 0.9:
            confidence_level = "Very high confidence"
        elif final_confidence >= 0.8:
            confidence_level = "High confidence"
        elif final_confidence >= 0.7:
            confidence_level = "Good confidence"
        elif final_confidence >= 0.6:
            confidence_level = "Moderate confidence"
        elif final_confidence >= 0.4:
            confidence_level = "Low confidence"
        else:
            confidence_level = "Very low confidence"
        
        return {
            'prediction': prediction,
            'confidence': float(final_confidence * 100),
            'risk_score': float(risk_score * 100),
            'legitimacy_probability': float(final_confidence),
            'confidence_level': confidence_level,
            'model_used': self.model_type,
            'analysis_breakdown': {
                'volume_score': self._get_volume_score(total_volume),
                'verification_score': 20 if is_verified else -25,
                'social_score': self._get_social_score(has_discord, has_twitter),
                'reddit_score': self._get_reddit_score(reddit_mentions, reddit_sentiment),
                'ownership_score': self._get_ownership_score(num_owners),
                'price_score': self._get_price_score(floor_price)
            }
        }
    
    def _get_volume_score(self, volume):
        """Get volume contribution to confidence"""
        if volume >= 1000000: return 40
        elif volume >= 500000: return 35
        elif volume >= 100000: return 25
        elif volume >= 50000: return 20
        elif volume >= 10000: return 15
        elif volume >= 1000: return 8
        elif volume >= 100: return 3
        elif volume >= 10: return -10
        elif volume >= 1: return -20
        else: return -30
    
    def _get_social_score(self, has_discord, has_twitter):
        """Get social media contribution to confidence"""
        if has_discord and has_twitter: return 12
        elif has_discord or has_twitter: return 6
        else: return -15
    
    def _get_reddit_score(self, mentions, sentiment):
        """Get Reddit contribution to confidence"""
        base_score = 0
        if mentions >= 100: base_score = 15
        elif mentions >= 50: base_score = 12
        elif mentions >= 20: base_score = 8
        elif mentions >= 5: base_score = 4
        elif mentions == 0: base_score = -12
        
        # Add sentiment bonus/penalty
        if sentiment >= 0.8: base_score += 10
        elif sentiment >= 0.7: base_score += 6
        elif sentiment >= 0.6: base_score += 3
        elif sentiment <= 0.3: base_score -= 10
        elif sentiment <= 0.4: base_score -= 5
        
        return base_score
    
    def _get_ownership_score(self, num_owners):
        """Get ownership distribution contribution to confidence"""
        if num_owners >= 10000: return 10
        elif num_owners >= 5000: return 8
        elif num_owners >= 1000: return 5
        elif num_owners >= 100: return 2
        elif num_owners < 10: return -15
        elif num_owners < 50: return -8
        else: return 0
    
    def _get_price_score(self, floor_price):
        """Get price sanity contribution to confidence"""
        if floor_price > 1000: return -10
        elif floor_price > 100: return -5
        elif floor_price < 0.00001: return -15
        elif floor_price < 0.001: return -8
        elif 0.1 <= floor_price <= 50: return 3
        else: return 0
    
    def save_model(self, filepath: str = 'model_outputs/rule_based_model.json'):
        """Save model configuration"""
        model_config = {
            'model_type': self.model_type,
            'version': self.version,
            'created_at': self.created_at,
            'known_legitimate': self.known_legitimate,
            'scam_keywords': self.scam_keywords
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"✅ Rule-based model saved: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str = 'model_outputs/rule_based_model.json'):
        """Load model configuration"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            model = cls()
            model.known_legitimate = config.get('known_legitimate', model.known_legitimate)
            model.scam_keywords = config.get('scam_keywords', model.scam_keywords)
            model.version = config.get('version', model.version)
            
            print(f"✅ Rule-based model loaded: {filepath}")
            return model
        except FileNotFoundError:
            print("⚠️ No saved model found, using defaults")
            return cls()

def main():
    """Test the rule-based model"""
    model = RuleBasedNFTAuthenticityModel()
    
    # Test cases
    test_cases = [
        {
            'name': 'Bored Ape Yacht Club',
            'slug': 'boredapeyachtclub',
            'features': {
                'total_volume': 1612804,
                'is_verified': 1,
                'has_discord': 1,
                'has_twitter': 1,
                'reddit_mentions': 26,
                'reddit_sentiment': 0.90,
                'num_owners': 5500,
                'floor_price': 11.64,
                'total_supply': 10000,
                'reddit_engagement': 6719
            }
        },
        {
            'name': 'DXTerminal',
            'slug': 'dxterminal',
            'features': {
                'total_volume': 774,
                'is_verified': 1,
                'has_discord': 0,
                'has_twitter': 1,
                'reddit_mentions': 0,
                'reddit_sentiment': 0.50,
                'num_owners': 4580,
                'floor_price': 0.00273,
                'total_supply': 15757,
                'reddit_engagement': 0
            }
        },
        {
            'name': 'Seduce the Nymphs',
            'slug': 'seduce-the-nymphs-collection-1',
            'features': {
                'total_volume': 0,
                'is_verified': 0,
                'has_discord': 0,
                'has_twitter': 0,
                'reddit_mentions': 0,
                'reddit_sentiment': 0.50,
                'num_owners': 5,
                'floor_price': 1e-14,
                'total_supply': 11,
                'reddit_engagement': 0
            }
        }
    ]
    
    print("🧪 Testing Rule-Based NFT Authenticity Model")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\n📊 Testing: {test['name']}")
        result = model.predict_collection(test['features'], test['slug'])
        
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Risk Score: {result['risk_score']:.1f}%")
        print(f"   Level: {result['confidence_level']}")
    
    # Save model
    model.save_model()

if __name__ == "__main__":
    main()
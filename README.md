# 🔍 NFTruth - AI-Powered NFT Authenticity Detector (python base)

> *Fighting NFT scams with machine learning, one collection at a time!* 🚀

## 🎯 What is NFTruth?

NFTruth is an intelligent system that analyzes NFT collections to determine their legitimacy and detect potential scams. Using ensemble machine learning algorithms trained on multi-source data (OpenSea marketplace data, Reddit social sentiment, and Ethereum blockchain metrics), it provides comprehensive risk assessments for NFT collections.

## 🏗️ System Architecture

```
NFTruth/
├── 🎯 app/
│   ├── 📊 data/
│   │   ├── opensea_collector.py      # OpenSea API integration & data collection
│   │   ├── reddit_collector.py       # Reddit OAuth + sentiment analysis pipeline
│   │   ├── etherscan_collector.py    # Ethereum blockchain analysis (placeholder)
│   │   └── ml_data_transformer.py    # Feature engineering & ML data preparation
│   ├── 🤖 models/
│   │   ├── model.py                  # Ensemble ML model implementation
│   │   ├── model_notebook.ipynb      # Technical documentation & explanation
│   │   └── opensea_known_legit.py    # Curated legitimate collections database
│   ├── 📈 model_training.py          # Synthetic data generation & training pipeline
│   ├── 🔮 predict.py                 # Prediction interface & risk assessment
│   └── 📋 opensea_collections.py     # Collection slug mappings
├── 🏆 model_outputs/
│   └── rule_based_model.json         # Rule-based baseline model
├── 📚 training_data/                 # Generated training datasets
├── 🧪 tests/
│   ├── test_model_setup.py          # ML functionality validation
│   └── test_opensea.py              # API connection testing
├── 📋 requirements.txt              # Python dependencies
└── 📖 README.md                     # This documentation
```

## 🧠 How The System Works

### 📊 Multi-Source Data Collection Pipeline

#### 🏪 **OpenSea API Integration** ([`opensea_collector.py`](app/data/opensea_collector.py))
```python
# Comprehensive collection metrics extraction
- Collection verification status (safelist_status)
- Trading statistics (total_volume, floor_price, market_cap)
- Social presence (Discord, Twitter links)
- Ownership metrics (total_supply, num_owners)
- Price dynamics (average_price, price_changes)
- Collection features (trait_offers_enabled, collection_offers_enabled)
```

#### 💬 **Reddit Social Intelligence** ([`reddit_collector.py`](app/data/reddit_collector.py))
```python
# Advanced social sentiment analysis pipeline
- OAuth 2.0 authentication with Reddit API
- Multi-subreddit targeted data collection:
  * crypto_general: ['cryptocurrency', 'crypto', 'CryptoMarkets']
  * nft_specific: ['NFT', 'NFTs', 'opensea', 'NFTsMarketplace']
  * ethereum: ['ethereum', 'ethtrader', 'ethfinance']
  * trading_focused: ['wallstreetbets', 'CryptoMoonShots']
- VADER sentiment analysis integration
- Scam keyword detection: ['scam', 'rugpull', 'fake', 'fraud']
- Hype indicator tracking: ['moon', 'diamond hands', 'hodl', 'wagmi']
```

#### ⛓️ **Blockchain Analysis** ([`etherscan_collector.py`](app/data/etherscan_collector.py))
```python
# Creator wallet and transaction analysis (framework ready)
- Wallet age and transaction history
- Suspicious pattern detection (wash trading)
- Creator balance and activity analysis
- Mint distribution pattern recognition
```

### 🔬 Advanced Feature Engineering ([`ml_data_transformer.py`](app/data/ml_data_transformer.py))

The [`MLDataTransformer`](app/data/ml_data_transformer.py) class transforms raw data into 20+ meaningful ML features:

#### 💰 **Market Intelligence Features**
```python
# Liquidity and market efficiency indicators
volume_per_owner = total_volume / num_owners        # Liquidity quality
market_efficiency = market_cap / total_volume       # Market maturity  
price_premium = average_price / floor_price         # Pricing structure
avg_daily_volume = total_volume / collection_age_days
```

#### 🗣️ **Social Sentiment Scoring**
```python
# Community engagement metrics using VADER sentiment analysis
social_score = reddit_mentions + reddit_engagement
sentiment_analysis = SentimentIntensityAnalyzer().polarity_scores()
scam_keyword_density = scam_mentions / total_mentions
hype_indicator = hype_keywords_count / total_discussion_volume
```

#### ⛓️ **Blockchain Forensics**
```python
# Creator and trading pattern analysis
creator_wallet_age = days_since_first_transaction
wash_trading_score = circular_transactions / total_transactions
mint_distribution_score = distribution_uniformity_analysis
whale_concentration = 1 - (num_owners / total_supply)
```

### 🤖 Ensemble Machine Learning Architecture ([`model.py`](app/models/model.py))

The [`NFTAuthenticityModel`](app/models/model.py) implements four specialized algorithms:

#### 🌳 **Random Forest Classifier**
- **Strength**: Handles complex feature interactions and mixed data types
- **Use Case**: Captures non-linear relationships between market metrics
- **Implementation**: `RandomForestClassifier(n_estimators=100, random_state=42)`

#### 🚀 **Gradient Boosting Classifier** 
- **Strength**: Sequential learning builds strong patterns from weak signals
- **Use Case**: Detects subtle scam indicators through iterative improvement
- **Implementation**: `GradientBoostingClassifier(random_state=42)`

#### 📈 **Logistic Regression**
- **Strength**: Interpretable linear relationships with feature scaling
- **Use Case**: Provides explainable risk factors and coefficients
- **Implementation**: `LogisticRegression(random_state=42)` with `StandardScaler`

#### 🎯 **Support Vector Machine (SVM)**
- **Strength**: Finds optimal decision boundaries in high-dimensional space
- **Use Case**: Separates legitimate from suspicious collections precisely
- **Implementation**: `SVC(probability=True, random_state=42)` with scaling

### 🏷️ Intelligent Labeling System

Since NFT authenticity ground truth is rare, the system uses a sophisticated scoring methodology in [`create_synthetic_labels()`](app/models/model.py):

```python
# Legitimacy scoring algorithm (from model.py)
def create_synthetic_labels(self, df):
    scores = []
    for _, row in df.iterrows():
        score = 0
        
        # Verification signals (+3 points)
        if row.get('is_verified', False): score += 3
        
        # Social presence (+1 point each)
        if row.get('has_discord', False): score += 1
        if row.get('has_twitter', False): score += 1
        
        # Market signals (+2 points each)  
        if row.get('floor_price', 0) > 0: score += 2
        if row.get('total_volume', 0) > 1000: score += 2
        
        # Community signals (+1 point each)
        if row.get('num_owners', 0) > 1000: score += 1
        if row.get('reddit_mentions', 0) > 0: score += 1
        
        # Whitelist verification (+3 points)
        collection_name = row.get('collection', '').lower()
        if any(legit in collection_name for legit in self.known_legitimate):
            score += 3
            
        scores.append(score)
    
    # Classification threshold: score >= 6 = legitimate
    return [1 if score >= 6 else 0 for score in scores]
```

### 📈 Synthetic Data Generation ([`model_training.py`](app/model_training.py))

The [`generate_synthetic_nft_data()`](app/model_training.py) function creates realistic training data:

```python
# Creates balanced dataset with realistic feature distributions
- 65% legitimate collections (default ratio)
- 35% suspicious collections
- Realistic correlations between features
- Saves timestamped JSON files in training_data/
```

## 🔍 Complete Feature Analysis

### 📊 **Market Intelligence (9 features)**
- `total_volume`, `floor_price`, `average_price`, `market_cap`
- `volume_per_owner`, `market_efficiency`, `price_premium`
- `avg_daily_volume`, `liquidity_indicator`

### 🏷️ **Collection Properties (8 features)**  
- `is_verified`, `safelist_status`, `has_discord`, `has_twitter`
- `trait_offers_enabled`, `collection_offers_enabled`
- `total_supply`, `num_owners`

### 💬 **Social Intelligence (6 features)**
- `reddit_mentions`, `reddit_engagement`, `social_score`
- `reddit_sentiment`, `scam_keyword_density`, `hype_indicator`

### ⛓️ **Blockchain Forensics (7 features)**
- `creator_wallet_age_days`, `creator_transaction_count`
- `wash_trading_score`, `suspicious_transaction_patterns`
- `mint_distribution_score`, `whale_concentration`
- `creator_balance_eth`

## 🎯 Prediction Interface ([`predict.py`](app/predict.py))

The system provides comprehensive risk analysis through the prediction interface:

```python
# Example prediction output
{
    "collection": "example-nft-collection",
    "prediction": "Legitimate" | "Suspicious", 
    "confidence": {
        "legitimate": 0.847,    # 84.7% confidence
        "suspicious": 0.153     # 15.3% risk
    },
    "risk_score": 0.153,        # Inverse of legitimacy probability
    "risk_level": "Low",        # Categorical assessment
    "model_used": "LogisticRegression",
    "features_analyzed": {...}, # All extracted features
    "timestamp": "2025-06-20T..."
}
```

## ⚠️ Risk Classification System

| Risk Level | Score Range | Characteristics | Action Recommended |
|------------|-------------|-----------------|-------------------|
| 🟢 **Low Risk** | 0-30% | Verified, high volume, strong community | ✅ Relatively safe to proceed |
| 🟡 **Medium Risk** | 31-50% | Mixed signals, some concerns | ⚠️ Proceed with caution |
| 🟠 **High Risk** | 51-70% | Multiple red flags detected | 🚨 High caution advised |
| 🔴 **Very High Risk** | 71-100% | Strong scam indicators | ❌ Avoid completely |

## 🛠️ Technical Implementation Stack

### **Core Dependencies** ([`requirements.txt`](requirements.txt))
```python
# Machine Learning & Data Processing
numpy, pandas, scikit-learn, joblib

# API & Web Functionality  
requests, python-dotenv

# Natural Language Processing
nltk, vaderSentiment, textblob

# Data Visualization
matplotlib, seaborn

# Date Handling
python-dateutil, pytz
```

### **External APIs Required**
- **🏪 OpenSea API** - Collection marketplace data
- **💬 Reddit API** - Social sentiment analysis (OAuth 2.0)
- **⛓️ Etherscan API** - Ethereum blockchain data

## 📊 Model Performance Metrics

Based on synthetic training data evaluation:

> Logistic Regression is by far the most optimal!

| Model | Key Strengths | Use Case |
|-------|---------------|----------|
| **🏆 Logistic Regression** | Interpretable, fast, linear separability | Primary classifier for NFT authenticity |
| 🌳 Random Forest | Feature importance, non-linear patterns | Complex interaction detection |
| 🚀 Gradient Boosting | Sequential improvement, weak signal boosting | Subtle scam pattern recognition |
| 🎯 SVM | Maximum margin, high-dimensional separation | Precise decision boundaries |

## 🧪 Testing Suite

### **Model Validation** ([`test_model_setup.py`](tests/test_model_setup.py))
- Model initialization and training pipeline validation
- Feature engineering functionality testing
- Prediction interface verification

### **API Integration** ([`test_opensea.py`](tests/test_opensea.py))
- OpenSea API connection and data extraction testing
- Rate limiting and error handling validation

## 📚 Known Legitimate Collections ([`opensea_known_legit.py`](app/models/opensea_known_legit.py))

Curated database of verified legitimate NFT collections:
- CryptoPunks, Bored Ape Yacht Club, Azuki
- Doodles, CloneX, Meebits, World of Women
- Used for ground truth labeling and validation

## 🔄 Data Pipeline Flow

1. **Collection Input** → Collection slug or name
2. **Data Collection** → Multi-source API calls (OpenSea, Reddit)
3. **Feature Engineering** → Transform raw data into ML features
4. **Model Prediction** → Ensemble voting across 4 algorithms
5. **Risk Assessment** → Confidence scores and risk categorization
6. **Output Generation** → Structured JSON response with analysis

## 🚀 Future Enhancement Roadmap

- [ ] 🔄 **Real-time monitoring** - Live collection tracking dashboard
- [ ] 🌐 **Web interface** - User-friendly analysis portal
- [ ] 🤝 **Community reporting** - Crowdsourced scam detection system

## ⚖️ Important Disclaimers

⚠️ **Investment Warning**: This tool provides risk assessments based on observable data patterns and should not be the sole factor in investment decisions. The NFT market is highly speculative and volatile.

🔬 **Research Tool**: NFTruth is designed as a research and educational tool to demonstrate machine learning applications in blockchain analysis.

📊 **Data Limitations**: Predictions are based on publicly available data and may not capture all risk factors or market dynamics.

🎓 **Educational Purpose**: This system demonstrates advanced ML techniques for blockchain analysis and should be used for learning and research purposes.

**Always conduct your own research (DYOR) before making any financial decisions** 🧠

---

*Built with ❤️ to make the NFT space safer for everyone* 🛡️

**⭐ Star this repository if you found it helpful!**
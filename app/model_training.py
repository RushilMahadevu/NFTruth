import numpy as np
import pandas as pd
import json
from datetime import datetime

def generate_synthetic_nft_data(n_samples=100, legit_ratio=0.6, seed=42):
    np.random.seed(seed)
    data = []
    legit_count = int(n_samples * legit_ratio)
    scam_count = n_samples - legit_count

    # Legitimate collections
    for _ in range(legit_count):
        floor_price = np.random.lognormal(mean=1, sigma=1.2)
        total_volume = np.random.lognormal(mean=8, sigma=1.5)
        market_cap = total_volume * np.random.uniform(0.8, 2.0)
        num_owners = np.random.randint(500, 20000)
        average_price = floor_price * np.random.uniform(0.8, 3.5)
        reddit_mentions = np.random.poisson(10)
        reddit_sentiment = np.random.beta(3, 1.5)
        reddit_engagement = np.random.poisson(100)
        total_supply = np.random.randint(1000, 20000)
        is_verified = np.random.choice([0, 1], p=[0.2, 0.8])
        has_discord = np.random.choice([0, 1], p=[0.3, 0.7])
        has_twitter = np.random.choice([0, 1], p=[0.1, 0.9])
        trait_offers_enabled = np.random.choice([0, 1], p=[0.5, 0.5])
        collection_offers_enabled = np.random.choice([0, 1], p=[0.3, 0.7])

        features = {
            "total_supply": int(total_supply),
            "is_verified": int(is_verified),
            "has_discord": int(has_discord),
            "has_twitter": int(has_twitter),
            "trait_offers_enabled": int(trait_offers_enabled),
            "collection_offers_enabled": int(collection_offers_enabled),
            "floor_price": float(floor_price),
            "market_cap": float(market_cap),
            "total_volume": float(total_volume),
            "num_owners": int(num_owners),
            "average_price": float(average_price),
            "reddit_mentions": int(reddit_mentions),
            "reddit_sentiment": float(reddit_sentiment),
            "reddit_engagement": int(reddit_engagement)
        }
        data.append({
            "features": features,
            "collection_slug": f"legit_collection_{np.random.randint(10000)}"
        })

    # Suspicious collections
    for _ in range(scam_count):
        floor_price = np.random.lognormal(mean=-1, sigma=1.5)
        total_volume = np.random.lognormal(mean=4, sigma=1.5)
        market_cap = total_volume * np.random.uniform(0.5, 1.5)
        num_owners = np.random.randint(5, 500)
        average_price = floor_price * np.random.uniform(0.5, 2.0)
        reddit_mentions = np.random.poisson(1)
        reddit_sentiment = np.random.beta(1.5, 3)
        reddit_engagement = np.random.poisson(10)
        total_supply = np.random.randint(10, 10000)
        is_verified = np.random.choice([0, 1], p=[0.95, 0.05])
        has_discord = np.random.choice([0, 1], p=[0.8, 0.2])
        has_twitter = np.random.choice([0, 1], p=[0.7, 0.3])
        trait_offers_enabled = np.random.choice([0, 1], p=[0.7, 0.3])
        collection_offers_enabled = np.random.choice([0, 1], p=[0.8, 0.2])

        features = {
            "total_supply": int(total_supply),
            "is_verified": int(is_verified),
            "has_discord": int(has_discord),
            "has_twitter": int(has_twitter),
            "trait_offers_enabled": int(trait_offers_enabled),
            "collection_offers_enabled": int(collection_offers_enabled),
            "floor_price": float(floor_price),
            "market_cap": float(market_cap),
            "total_volume": float(total_volume),
            "num_owners": int(num_owners),
            "average_price": float(average_price),
            "reddit_mentions": int(reddit_mentions),
            "reddit_sentiment": float(reddit_sentiment),
            "reddit_engagement": int(reddit_engagement)
        }
        data.append({
            "features": features,
            "collection_slug": f"suspicious_collection_{np.random.randint(10000)}"
        })

    # Shuffle data
    np.random.shuffle(data)
    return data

def save_synthetic_data(data, out_dir="training_data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{out_dir}/nft_training_data_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"data": data}, f, indent=2)
    print(f"✅ Synthetic NFT training data saved: {out_path}")

def main():
    print("🔬 Generating synthetic NFT training data for ML model...")
    data = generate_synthetic_nft_data(n_samples=120, legit_ratio=0.65)
    save_synthetic_data(data)
    print("Done.")

if __name__ == "__main__":
    main()
import json
import os
from datetime import datetime
from data.ml_data_transformer import MLDataTransformer

def main():
    """
    Main function to collect NFT data and prepare it for ML training
    """
    print("Starting NFT Data Collection for ML Training...")
    print("=" * 60)
    
    # Initialize the transformer
    transformer = MLDataTransformer()
    
    # Define collections to analyze
    collections = [
        "boredapeyachtclub",
        "cryptopunks", 
        "azuki",
        "doodles-official",
        "mutant-ape-yacht-club",
        "world-of-women",
        "clonex",
        "meebits"
    ]
    
    print(f"Collecting data for {len(collections)} collections...")
    
    # Collect all data
    all_data = []
    successful_collections = []
    failed_collections = []
    
    for i, collection_slug in enumerate(collections, 1):
        print(f"\n[{i}/{len(collections)}] Processing: {collection_slug}")
        print("-" * 40)
        
        try:
            # Get transformed data for this collection
            collection_data = transformer.transform_collection_data(collection_slug)
            
            if collection_data and len(collection_data) > 0:
                all_data.extend(collection_data)
                successful_collections.append(collection_slug)
                print(f"✅ Successfully collected {len(collection_data)} records")
            else:
                failed_collections.append(collection_slug)
                print(f"❌ No data collected for {collection_slug}")
                
        except Exception as e:
            failed_collections.append(collection_slug)
            print(f"❌ Error processing {collection_slug}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total collections processed: {len(collections)}")
    print(f"Successful: {len(successful_collections)}")
    print(f"Failed: {len(failed_collections)}")
    print(f"Total data points collected: {len(all_data)}")
    
    if failed_collections:
        print(f"\nFailed collections: {', '.join(failed_collections)}")
    
    # Save data to JSON files
    if all_data:
        save_training_data(all_data, successful_collections)
    else:
        print("\n❌ No data to save!")
    
    print("\n🎉 Data collection complete!")

def save_training_data(data, collections):
    """
    Save the collected data in multiple formats for ML training
    """
    print("\n" + "=" * 60)
    print("SAVING TRAINING DATA")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    data_dir = "training_data"
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save complete dataset
    complete_file = f"{data_dir}/nft_training_data_{timestamp}.json"
    with open(complete_file, 'w') as f:
        json.dump({
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_records": len(data),
                "collections": collections,
                "description": "Complete NFT dataset for ML training"
            },
            "data": data
        }, f, indent=2)
    print(f"✅ Saved complete dataset: {complete_file}")
    
    # 2. Save features only (for easier model training)
    features_only = []
    for record in data:
        if 'features' in record:
            features_record = {
                'collection_slug': record['collection_slug'],
                'token_id': record.get('token_id'),
                **record['features']
            }
            features_only.append(features_record)
    
    features_file = f"{data_dir}/nft_features_{timestamp}.json"
    with open(features_file, 'w') as f:
        json.dump({
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_records": len(features_only),
                "collections": collections,
                "description": "NFT features extracted for ML training"
            },
            "features": features_only
        }, f, indent=2)
    print(f"✅ Saved features dataset: {features_file}")
    
    # 3. Save summary statistics
    summary = generate_dataset_summary(data, collections)
    summary_file = f"{data_dir}/dataset_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved dataset summary: {summary_file}")
    
    # 4. Save latest version (overwrite)
    latest_complete = f"{data_dir}/nft_training_data_latest.json"
    with open(latest_complete, 'w') as f:
        json.dump({
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_records": len(data),
                "collections": collections,
                "description": "Latest NFT dataset for ML training"
            },
            "data": data
        }, f, indent=2)
    
    latest_features = f"{data_dir}/nft_features_latest.json"
    with open(latest_features, 'w') as f:
        json.dump({
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_records": len(features_only),
                "collections": collections,
                "description": "Latest NFT features for ML training"
            },
            "features": features_only
        }, f, indent=2)
    
    print(f"✅ Saved latest versions for easy access")
    
    print(f"\n📁 All files saved in: {os.path.abspath(data_dir)}")

def generate_dataset_summary(data, collections):
    """
    Generate summary statistics about the collected dataset
    """
    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_records": len(data),
            "total_collections": len(collections),
            "collections": collections
        },
        "statistics": {
            "records_per_collection": {},
            "feature_coverage": {},
            "data_quality": {}
        }
    }
    
    # Records per collection
    for collection in collections:
        collection_records = [d for d in data if d.get('collection_slug') == collection]
        summary["statistics"]["records_per_collection"][collection] = len(collection_records)
    
    # Feature coverage analysis
    if data:
        sample_features = data[0].get('features', {})
        feature_names = list(sample_features.keys())
        
        for feature in feature_names:
            non_null_count = sum(1 for d in data if d.get('features', {}).get(feature) is not None)
            coverage_percent = (non_null_count / len(data)) * 100
            summary["statistics"]["feature_coverage"][feature] = {
                "non_null_count": non_null_count,
                "coverage_percent": round(coverage_percent, 2)
            }
    
    # Data quality metrics
    complete_records = sum(1 for d in data if d.get('features') and len(d['features']) > 0)
    summary["statistics"]["data_quality"] = {
        "complete_records": complete_records,
        "completion_rate": round((complete_records / len(data)) * 100, 2) if data else 0,
        "average_features_per_record": round(sum(len(d.get('features', {})) for d in data) / len(data), 2) if data else 0
    }
    
    return summary

def debug_mode():
    """
    Run in debug mode to test API connections
    """
    print("Running in DEBUG mode...")
    print("=" * 60)
    
    transformer = MLDataTransformer()
    
    # Test a few collections
    test_collections = ["boredapeyachtclub", "cryptopunks", "azuki"]
    
    for collection in test_collections:
        print(f"\n🔍 Debugging: {collection}")
        transformer.debug_opensea_response(collection)
        
        # Test the full transformation
        print(f"\n🔄 Testing transformation for: {collection}")
        try:
            sample_data = transformer.transform_collection_data(collection)
            if sample_data:
                print(f"✅ Successfully transformed {len(sample_data)} records")
                print(f"Sample record keys: {list(sample_data[0].keys()) if sample_data else 'None'}")
                if sample_data[0].get('features'):
                    print(f"Feature keys: {list(sample_data[0]['features'].keys())}")
            else:
                print("❌ No data returned from transformation")
        except Exception as e:
            print(f"❌ Error in transformation: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for debug flag
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        debug_mode()
    else:
        main()
"""
Simple test script to verify the model setup works
"""
import json
import pandas as pd
import numpy as np

def test_data_loading():
    """Test if we can load the training data"""
    data_path = "training_data/nft_features_latest.json"
    
    print("Testing data loading...")
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        features_list = []
        for record in data.get('features', []):
            features_list.append(record)
        
        df = pd.DataFrame(features_list)
        
        print(f"✅ Successfully loaded {len(df)} records")
        print(f"✅ Found {len(df.columns)} features")
        print(f"✅ Features: {list(df.columns)}")
        
        # Show sample data
        print("\nSample data:")
        print(df.head())
        
        # Check for missing values
        print(f"\nMissing values per column:")
        print(df.isnull().sum())
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_basic_ml():
    """Test basic ML functionality"""
    print("\nTesting basic ML imports...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        print("✅ Successfully imported sklearn components")
        
        # Create dummy data
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Dummy model accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with ML: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing NFT Model Setup")
    print("="*40)
    
    success = True
    
    # Test data loading
    if not test_data_loading():
        success = False
    
    # Test ML functionality
    if not test_basic_ml():
        success = False
    
    if success:
        print("\n🎉 All tests passed! Ready to train the full model.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")

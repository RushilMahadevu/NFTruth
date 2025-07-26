import pandas as pd
import numpy as np
import json
import joblib
import os
from .opensea_known_legit import known_legit
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score,
   confusion_matrix, classification_report, roc_auc_score, roc_curve
)




@dataclass
class ModelResults:
   """Results from model training and evaluation"""
   model_name: str
   accuracy: float
   precision: float
   recall: float
   f1_score: float
   roc_auc: float
   confusion_matrix: np.ndarray
   feature_importance: Optional[Dict[str, float]] = None


class NFTAuthenticityModel:
   """
   ML Model for detecting NFT authenticity and scam probability
   """
  
   def __init__(self):
       self.models = {}
       self.best_model = None
       self.model_scores = {}
       self.feature_names = []
       self.scaler = StandardScaler()
       self.scalers = {}  # Add scalers dictionary
       self.results = {}  # Add results dictionary for storing ModelResults
       self.model_path = 'model_outputs/nft_authenticity_model.pkl' # pkl is a common format for saving models that can be loaded later


   def prepare_data(self):
       """
       Load and prepare data for training
       """
       print("Loading training data...")
      
       # Look for the latest training data file
       data_dir = "training_data"
       if not os.path.exists(data_dir):
           raise FileNotFoundError(f"Training data directory '{data_dir}' not found. Run data collection first!")
      
       # Find the latest data file
       data_files = [f for f in os.listdir(data_dir) if f.startswith('nft_training_data_') and f.endswith('.json')] # Filter for JSON files
       if not data_files:
           # Try the latest file
           latest_file = os.path.join(data_dir, 'nft_training_data_latest.json')
           if os.path.exists(latest_file):
               data_file = latest_file
           else:
               raise FileNotFoundError("No training data files found. Run data collection first!")
       else:
           # Get the most recent file
           data_files.sort(reverse=True)
           data_file = os.path.join(data_dir, data_files[0])
      
       print(f"Loading data from: {data_file}")
      
       # Load the data
       with open(data_file, 'r') as f:
           data = json.load(f)
      
       # Extract features and create labels
       features_list = []
       collections = []
      
       if 'data' in data:
           records = data['data']
       else:
           records = data
      
       for record in records:
           if 'features' in record:
               features_list.append(record['features'])
               collections.append(record.get('collection_slug', 'unknown'))
      
       if not features_list:
           raise ValueError("No feature data found in training file")
      
       print(f"Loaded {len(features_list)} records with {len(features_list[0])} features")
      
       # Convert to DataFrame for easier handling
       import pandas as pd
       df = pd.DataFrame(features_list) # DataFrame is a powerful data structure for handling tabular data (similar to a SQL table or Excel sheet)
      
       # Store feature names
       self.feature_names = list(df.columns)
       print(f"Features: {self.feature_names}") # Print feature names for debugging
      
       # Create labels based on heuristics (since we don't have ground truth)
       y = self.create_synthetic_labels(df, collections)
      
       print("Creating synthetic labels based on heuristics...")
       print(f"Label distribution:")
       unique, counts = np.unique(y, return_counts=True) # Get unique labels and their counts np.unique returns the unique values in an array, and return_counts=True returns the counts of each unique value
       # y is the labels array created by create_synthetic_labels method
       for label, count in zip(unique, counts): # zip combines two lists into a list of tuples
           label_name = "Legitimate" if label == 1 else "Suspicious"
           print(f"  {label_name} ({label}): {count} ({count/len(y)*100:.1f}%)") # Print label distribution by percentage
      
       # Feature engineering
       print("Preparing features...")
       X = self.engineer_features(df)
       print(f"Created {X.shape[1]} features after engineering") # X is a numpy array of features after engineering and .shape[1] gives the number of features
      
       return X, y, self.feature_names # This returns the processed features, labels, and feature names for further use


   def create_synthetic_labels(self, df, collections):
        """
        Create more realistic synthetic labels based on multiple criteria
        """
        labels = []
        
        for idx, row in df.iterrows():
            collection_name = collections[idx] if idx < len(collections) else "unknown"
            
            # Start with legitimate assumption
            score = 1
            
            # Verification gives strong positive signal
            if row.get('is_verified', 0) == 1:
                score += 1.5
            else:
                score -= 0.2
            
            # Social media presence
            if row.get('has_discord', 0) == 1:
                score += 0.2
            if row.get('has_twitter', 0) == 1:
                score += 0.2
            
            # Volume and market metrics (more nuanced)
            total_volume = row.get('total_volume', 0)
            num_owners = row.get('num_owners', 0)
            floor_price = row.get('floor_price', 0)
            
            # High volume collections are usually legitimate
            if total_volume > 1000000:  # 1M ETH
                score += 1
            elif total_volume > 500000:  # 500k ETH
                score += 0.9
            elif total_volume > 100000:  # 100k ETH
                score += 0.8
            elif total_volume > 10000:  # 10k ETH
                score += 0.7
            elif total_volume > 1000:  # 1k ETH
                score += 0.6
            elif total_volume > 500 :  # 500 ETH
                score += 0.5
            elif total_volume > 100:  # 100 ETH
                score += 0.3
            elif total_volume > 10:  # 10 ETH
                score += 0.1
            elif total_volume < 10:  # Very low volume is suspicious
                score -= 0.2
            
            # Owner distribution
            if num_owners > 5000:
                score += 0.9
            elif num_owners > 1000:
                score += 0.7
            elif num_owners > 500:
                score += 0.3
            elif num_owners < 100:
                score -= 0.2
            elif num_owners < 10:
                score -= 0.6
            
            # Floor price sanity check
            if floor_price > 100:  # Extremely high floor price
                score -= 0.1
            elif floor_price > 1:  # Reasonable floor price
                score += 0.1
            
            # Reddit sentiment
            reddit_sentiment = row.get('reddit_sentiment', 0.5)
            if reddit_sentiment > 0.7:
                score += 0.7
            elif reddit_sentiment > 0.5:
                score += 0.6
            elif reddit_sentiment == 0.5:
                score += 0.5
            elif reddit_sentiment < 0.3:
                score -= 0.2
            
            # Known legitimate collections
            if collection_name.lower() in known_legit:
                score = 0.99  # Force legitimate
            
            # Convert score to binary label with some randomness to avoid overfitting
            threshold = 0.6 + np.random.normal(0, 0.1)  # Add noise to threshold
            label = 1 if score > threshold else 0
            
            labels.append(label)
            print(f"  {collection_name}: Score={score:.2f}, Label={'Legitimate' if label == 1 else 'Suspicious'}")
        
        return np.array(labels)


   def engineer_features(self, df):
        """
        Create additional features from the raw data with better error handling
        """
        # Start with original features
        X = df.values.astype(float)
        
        # Create additional features with better error handling
        additional_features = []
        
        # Price-based features
        floor_price = df['floor_price'].values
        market_cap = df['market_cap'].values
        total_volume = df['total_volume'].values
        num_owners = df['num_owners'].values
        average_price = df['average_price'].values
        
        # Volume per owner (safe division)
        volume_per_owner = np.zeros_like(num_owners, dtype=float)
        valid_owners = num_owners > 0
        volume_per_owner[valid_owners] = total_volume[valid_owners] / num_owners[valid_owners]
        volume_per_owner = np.clip(volume_per_owner, 0, np.percentile(volume_per_owner[volume_per_owner > 0], 95) if np.any(volume_per_owner > 0) else 1000)
        additional_features.append(volume_per_owner)
        
        # Market cap to volume ratio (safe division)
        mc_volume_ratio = np.zeros_like(total_volume, dtype=float)
        valid_volume = total_volume > 0
        mc_volume_ratio[valid_volume] = market_cap[valid_volume] / total_volume[valid_volume]
        mc_volume_ratio = np.clip(mc_volume_ratio, 0, 10)  # Cap at reasonable value
        additional_features.append(mc_volume_ratio)
        
        # Price premium (safe division)
        price_premium = np.ones_like(floor_price, dtype=float)
        valid_floor = floor_price > 0
        price_premium[valid_floor] = average_price[valid_floor] / floor_price[valid_floor]
        price_premium = np.clip(price_premium, 0.1, 100)  # Reasonable bounds
        additional_features.append(price_premium)
        
        # Social engagement score (normalized)
        reddit_mentions = np.clip(df['reddit_mentions'].values, 0, 1000)
        reddit_engagement = np.clip(df['reddit_engagement'].values, 0, 100000)
        social_score = np.log1p(reddit_mentions + reddit_engagement * 0.01)  # Log scale
        additional_features.append(social_score)
        
        # Liquidity indicator (normalized)
        liquidity = np.log1p(np.sqrt(np.clip(total_volume, 0, 10000000) * np.clip(num_owners, 1, 100000)))
        additional_features.append(liquidity)
        
        # Combine all features
        if additional_features:
            additional_array = np.column_stack(additional_features)
            X = np.column_stack([X, additional_array])
        
        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=0.0)
        
        return X
  
   def train_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Train multiple ML models with better validation
        """
        print("\nTraining multiple models...")
        
        # Check if we have enough data
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = min(class_counts)
        
        print(f"Minimum class count: {min_class_count}")
        
        if min_class_count < 3 or len(X) < 20:
            print("⚠️ Warning: Dataset too small for reliable training!")
            print("⚠️ Results will be unreliable. Collect more data!")
            
        # Use stratified split if possible
        if min_class_count >= 2 and len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        else:
            # Use all data for training but warn about overfitting
            X_train, X_test = X, X
            y_train, y_test = y, y
            print("⚠️ Using all data for training - overfitting likely!")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models with regularization to prevent overfitting
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduced to prevent overfitting
                max_depth=5,      # Limited depth
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,      # Limited depth
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,           # Regularization
                max_iter=1000,
                random_state=42
            ),
            'Support Vector Classifier': SVC(
                kernel='linear',  # Linear kernel for simplicity
                C=1.0,           # Regularization
                probability=True, # Enable probability estimates
                random_state=42
            )
        }
        
        # Train models
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            try:
                if name in ['Logistic Regression']:
                    model.fit(X_train_scaled, y_train)
                    if len(set(y_test)) > 1:  # Can only predict if we have both classes
                        y_pred = model.predict(X_test_scaled)
                        y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_pred = model.predict(X_train_scaled)[:len(y_test)]
                        y_prob = model.predict_proba(X_train_scaled)[:len(y_test), 1]
                else:
                    model.fit(X_train, y_train)
                    if len(set(y_test)) > 1:
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)[:, 1]
                    else:
                        y_pred = model.predict(X_train)[:len(y_test)]
                        y_prob = model.predict_proba(X_train)[:len(y_test), 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Only calculate ROC AUC if we have both classes
                if len(set(y_test)) > 1:
                    roc_auc = roc_auc_score(y_test, y_prob)
                else:
                    roc_auc = 0.5  # Random performance for single class
                
                # Store model and results in both formats
                self.models[name] = model
                self.model_scores[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                }
                
                # Also store in results format for consistency
                self.results[name] = ModelResults(
                    model_name=name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    roc_auc=roc_auc,
                    confusion_matrix=confusion_matrix(y_test, y_pred)
                )
                
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1 Score: {f1:.3f}")
                print(f"  ROC AUC: {roc_auc:.3f}")
                
                if accuracy > 0.95 and len(X) < 50:
                    print(f"  ⚠️ WARNING: {name} may be overfitted!")
                
            except Exception as e:
                print(f"  ❌ Failed to train {name}: {str(e)}")
  
   def evaluate_models(self):
       """
       Print detailed evaluation of all models
       """
       print("\n" + "="*60)
       print("MODEL EVALUATION SUMMARY")
       print("="*60)
      
       # Create comparison DataFrame from model_scores instead of results
       comparison_data = []
       for name, scores in self.model_scores.items():
           comparison_data.append({
               'Model': name,
               'Accuracy': scores['accuracy'],
               'Precision': scores['precision'], 
               'Recall': scores['recall'],
               'F1 Score': scores['f1_score'],
               'ROC AUC': scores['roc_auc']
           })
      
       if not comparison_data:
           print("No model results to display")
           return None
           
       comparison_df = pd.DataFrame(comparison_data)
       comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
      
       print(comparison_df.to_string(index=False, float_format='%.3f'))
      
       # Find best model (avoid perfect scores as they indicate overfitting)
       best_model_name = None
       best_score = 0
       
       for name, scores in self.model_scores.items():
           f1 = scores['f1_score']
           # Penalize perfect scores as they likely indicate overfitting
           if f1 >= 0.999:
               adjusted_score = f1 - 0.1  # Penalty for perfect scores
           else:
               adjusted_score = f1
               
           if adjusted_score > best_score:
               best_score = adjusted_score
               best_model_name = name
       
       print(f"\n🏆 Best performing model: {best_model_name}")
      
       return best_model_name
  
   def save_models(self, save_dir: str = "saved_models"):
       """
       Save trained models and metadata
       """
       os.makedirs(save_dir, exist_ok=True)
      
       # Save models
       for name, model in self.models.items():
           model_path = f"{save_dir}/{name.lower().replace(' ', '_')}_model.pkl"
           joblib.dump(model, model_path)
           print(f"Saved {name} model to {model_path}")
      
       # Save scalers
       for name, scaler in self.scalers.items():
           scaler_path = f"{save_dir}/{name}_scaler.pkl"
           joblib.dump(scaler, scaler_path)
           print(f"Saved {name} scaler to {scaler_path}")
      
       # Save metadata
       metadata = {
           'feature_names': self.feature_names,
           'model_results': {
               name: {
                   'accuracy': result.accuracy,
                   'precision': result.precision,
                   'recall': result.recall,
                   'f1_score': result.f1_score,
                   'roc_auc': result.roc_auc
               }
               for name, result in self.results.items()
           },
           'training_date': datetime.now().isoformat()
       }
      
       metadata_path = f"{save_dir}/model_metadata.json"
       with open(metadata_path, 'w') as f:
           json.dump(metadata, f, indent=2)
       print(f"Saved metadata to {metadata_path}")
  
   def save_best_model(self):
       """
       Save the best trained model and metadata
       """
       if not self.best_model:
           print("❌ No best model to save")
           return
      
       try:
           # Create model directory
           os.makedirs('model_outputs', exist_ok=True)
          
           # Save best model
           best_model_path = 'model_outputs/best_nft_model.pkl'
           joblib.dump(self.best_model, best_model_path)
          
           # Save scaler if it exists
           if self.scalers:
               scaler_path = 'model_outputs/feature_scaler.pkl'
               joblib.dump(self.scalers['standard'], scaler_path)
               print(f"✅ Scaler saved: {scaler_path}")
          
           # Save metadata
           metadata = {
               'model_type': type(self.best_model).__name__,
               'training_date': datetime.now().isoformat(),
               'features_used': self.feature_names,
               'model_performance': {
                   name: {
                       'accuracy': result.accuracy,
                       'precision': result.precision,
                       'recall': result.recall,
                       'f1_score': result.f1_score,
                       'roc_auc': result.roc_auc
                   }
                   for name, result in self.results.items()
               },
               'preprocessing_steps': 'StandardScaler + Feature Engineering'
           }
          
           metadata_path = 'model_outputs/model_metadata.json'
           with open(metadata_path, 'w') as f:
               json.dump(metadata, f, indent=2)
          
           print(f"✅ Best model saved: {best_model_path}")
           print(f"✅ Metadata saved: {metadata_path}")
          
       except Exception as e:
           print(f"❌ Error saving model: {e}")
  
   def predict_collection(self, collection_features: Dict, model_name: str = 'Random Forest') -> Dict:
    """
    Predict authenticity for a new collection with enhanced factor tracking
    """
    if model_name not in self.models:
        raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
   
    model = self.models[model_name]
   
    # Prepare features properly
    df = pd.DataFrame([collection_features])
   
    # Ensure all expected features are present
    expected_features = ['total_supply', 'is_verified', 'has_discord', 'has_twitter',
                       'trait_offers_enabled', 'collection_offers_enabled', 'floor_price',
                       'market_cap', 'total_volume', 'num_owners', 'average_price',
                       'reddit_mentions', 'reddit_sentiment', 'reddit_engagement']
   
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
   
    # Reorder columns to match training
    df = df[expected_features]
   
    # Apply the same feature engineering as during training
    features = self.engineer_features(df)
   
    # Scale if needed
    if model_name in ['Logistic Regression', 'SVC'] and 'standard' in self.scalers:
        scaler = self.scalers['standard']
        features = scaler.transform(features)
   
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # Extract key metrics
    total_volume = collection_features.get('total_volume', 0)
    is_verified = collection_features.get('is_verified', 0)
    reddit_mentions = collection_features.get('reddit_mentions', 0)
    reddit_sentiment = collection_features.get('reddit_sentiment', 0.5)
    has_discord = collection_features.get('has_discord', 0)
    has_twitter = collection_features.get('has_twitter', 0)
    num_owners = collection_features.get('num_owners', 0)
    floor_price = collection_features.get('floor_price', 0)
    
    # Track scoring factors for transparency
    scoring_log = {
        'base_score': 0.5,
        'adjustments': [],
        'caps_applied': [],
        'final_factors': {}
    }
    
    # 🚨 STRICT SUSPICIOUS CATEGORY CHECK FIRST
    suspicious_flags = 0
    flag_details = []
    
    if total_volume == 0:
        suspicious_flags += 1
        flag_details.append("No trading volume")
    if reddit_mentions == 0:
        suspicious_flags += 1
        flag_details.append("No Reddit activity")
    if not has_discord and not has_twitter:
        suspicious_flags += 1
        flag_details.append("No social media presence")
    if not is_verified:
        suspicious_flags += 1
        flag_details.append("Not OpenSea verified")
    
    # If collection has ALL suspicious flags, cap at 30%
    if suspicious_flags >= 4:
        final_confidence = min(0.30, 0.15 + (total_volume / 1000) * 0.15)
        final_prediction = 'Suspicious'
        risk_score = 1.0 - final_confidence
        
        scoring_log['caps_applied'].append(f"Strict rules applied - {suspicious_flags} red flags: {', '.join(flag_details)}")
        
        return {
            'prediction': final_prediction,
            'confidence': float(final_confidence * 100),
            'risk_score': float(risk_score * 100),
            'legitimacy_probability': float(final_confidence),
            'model_used': model_name + " (Strict Rules)",
            'warning': 'High Risk - Multiple red flags detected',
            'red_flags': flag_details,
            'scoring_log': scoring_log
        }
    
    # Start with neutral base for normal collections
    confidence_score = 0.5
    
    # 🏆 VOLUME-BASED SCORING (Strongest signal)
    volume_boost = 0
    if total_volume > 1000000:
        volume_boost = 0.70
        scoring_log['adjustments'].append0(f"Major volume boost: +70% (1M+ ETH)")
    elif total_volume > 500000:
        volume_boost = 0.50
        scoring_log['adjustments'].append(f"High volume boost: +50% (500k+ ETH)")
    elif total_volume > 100000:
        volume_boost = 0.30
        scoring_log['adjustments'].append(f"Good volume boost: +30% (100k+ ETH)")
    elif total_volume > 50000:
        volume_boost = 0.20
        scoring_log['adjustments'].append(f"Solid volume boost: +20% (50k+ ETH)")
    elif total_volume > 10000:
        volume_boost = 0.15
        scoring_log['adjustments'].append(f"Decent volume boost: +15% (10k+ ETH)")
    elif total_volume > 1000:
        volume_boost = 0.08
        scoring_log['adjustments'].append(f"Active volume boost: +8% (1k+ ETH)")
    elif total_volume > 100:
        volume_boost = 0.03
        scoring_log['adjustments'].append(f"Minimal volume boost: +3% (100+ ETH)")
    elif total_volume > 10:
        volume_boost = -0.05
        scoring_log['adjustments'].append(f"Low volume penalty: -5% (10+ ETH)")
    elif total_volume > 1:
        volume_boost = -0.15
        scoring_log['adjustments'].append(f"Very low volume penalty: -15% (1+ ETH)")
    else:
        volume_boost = -0.25
        scoring_log['adjustments'].append(f"No volume penalty: -25%")
    
    confidence_score += volume_boost
    
    # 🛡️ VERIFICATION STATUS
    if is_verified:
        confidence_score += 0.50
        scoring_log['adjustments'].append("Verification boost: +50%")
    else:
        confidence_score -= 0.50
        scoring_log['adjustments'].append("Unverified penalty: -50%")
    
    # 📱 SOCIAL MEDIA PRESENCE
    if has_discord and has_twitter:
        confidence_score += 0.15
        scoring_log['adjustments'].append("Full social media boost: +15%")
    elif has_discord or has_twitter:
        confidence_score += 0.08
        scoring_log['adjustments'].append("Partial social media boost: +8%")
    else:
        confidence_score -= 0.20
        scoring_log['adjustments'].append("No social media penalty: -20%")
    
    # 🗣️ REDDIT ENGAGEMENT
    if reddit_mentions > 50:
        confidence_score += 0.40
        scoring_log['adjustments'].append(f"High Reddit activity: +40% ({reddit_mentions} mentions)")
    elif reddit_mentions > 20:
        confidence_score += 0.25
        scoring_log['adjustments'].append(f"Good Reddit activity: +25% ({reddit_mentions} mentions)")
    elif reddit_mentions > 5:
        confidence_score += 0.08
        scoring_log['adjustments'].append(f"Some Reddit activity: +8% ({reddit_mentions} mentions)")
    elif reddit_mentions > 0:
        confidence_score += 0.03
        scoring_log['adjustments'].append(f"Minimal Reddit activity: +3% ({reddit_mentions} mentions)")
    else:
        confidence_score -= 0.15
        scoring_log['adjustments'].append("No Reddit activity penalty: -15%")
    
    # 😊 REDDIT SENTIMENT
    if reddit_sentiment > 0.8:
        confidence_score += 0.10
        scoring_log['adjustments'].append(f"Very positive sentiment: +10% ({reddit_sentiment:.2f})")
    elif reddit_sentiment > 0.7:
        confidence_score += 0.06
        scoring_log['adjustments'].append(f"Positive sentiment: +6% ({reddit_sentiment:.2f})")
    elif reddit_sentiment > 0.6:
        confidence_score += 0.03
        scoring_log['adjustments'].append(f"Neutral-positive sentiment: +3% ({reddit_sentiment:.2f})")
    elif reddit_sentiment < 0.3:
        confidence_score -= 0.15
        scoring_log['adjustments'].append(f"Negative sentiment penalty: -15% ({reddit_sentiment:.2f})")
    elif reddit_sentiment < 0.4:
        confidence_score -= 0.08
        scoring_log['adjustments'].append(f"Poor sentiment penalty: -8% ({reddit_sentiment:.2f})")
    
    # Apply caps and final bounds
    red_flags = suspicious_flags
    minimum_criteria_met = is_verified or has_discord or has_twitter or reddit_mentions > 0
    
    if red_flags >= 3:
        confidence_score = min(confidence_score, 0.30)
        scoring_log['caps_applied'].append(f"High risk cap applied: {red_flags} red flags")
    elif is_verified and total_volume > 1000:
        confidence_score = max(0.40, min(0.95, confidence_score))
        scoring_log['caps_applied'].append("Verified + volume bounds: 40-95%")
    elif is_verified:
        confidence_score = max(0.35, min(0.85, confidence_score))
        scoring_log['caps_applied'].append("Verified bounds: 35-85%")
    else:
        confidence_score = max(0.05, min(0.75, confidence_score))
        scoring_log['caps_applied'].append("Unverified bounds: 5-75%")
    
    final_prediction = 'Legitimate' if confidence_score >= 0.55 else 'Suspicious'
    risk_score = 1.0 - confidence_score
    
    # Store final factors for reporting
    scoring_log['final_factors'] = {
        'total_volume': total_volume,
        'is_verified': is_verified,
        'social_media_complete': has_discord and has_twitter,
        'reddit_mentions': reddit_mentions,
        'reddit_sentiment': reddit_sentiment,
        'red_flags_count': red_flags,
        'minimum_criteria_met': minimum_criteria_met
    }
    
    result = {
        'prediction': final_prediction,
        'confidence': float(confidence_score * 100),
        'risk_score': float(risk_score * 100),
        'legitimacy_probability': float(confidence_score),
        'model_used': model_name + " (Enhanced)",
        'red_flags': red_flags,
        'red_flag_details': flag_details,
        'minimum_criteria_met': minimum_criteria_met,
        'scoring_log': scoring_log
    }
    
    # Add warnings
    if confidence_score < 0.30:
        result['warning'] = "Very High Risk - Multiple red flags detected"
    elif confidence_score < 0.50:
        result['warning'] = "High Risk - Proceed with extreme caution"
    elif not minimum_criteria_met:
        result['warning'] = "Caution - Limited verification signals"
        
    return result
  
   def generate_synthetic_data(self, df, target_samples=50):
       """
       Generate synthetic NFT data for training when we have too few samples
       """
       print(f"\n🔄 Generating synthetic data to reach {target_samples} samples...")
      
       synthetic_data = []
      
       # Generate legitimate collections (70% of synthetic data)
       legitimate_count = int(target_samples * 0.7)
       for _ in range(legitimate_count):
           record = {}
          
           # Add more variance to prevent overfitting
           record['total_supply'] = np.random.randint(500, 50000)
           record['is_verified'] = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% verified
           record['has_discord'] = np.random.choice([0, 1], p=[0.4, 0.6])  # 60% have discord
           record['has_twitter'] = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% have twitter
           record['trait_offers_enabled'] = np.random.choice([0, 1], p=[0.5, 0.5])
           record['collection_offers_enabled'] = np.random.choice([0, 1], p=[0.4, 0.6])
           record['floor_price'] = np.random.lognormal(0, 1.5)  # More realistic distribution
           record['market_cap'] = np.random.lognormal(8, 2)  # Log-normal for market cap
           record['total_volume'] = np.random.lognormal(9, 2)  # Log-normal for volume
           record['num_owners'] = np.random.randint(100, 15000)
           record['average_price'] = record['floor_price'] * np.random.uniform(0.8, 5.0)  # More variance
           record['reddit_mentions'] = np.random.poisson(5)  # Poisson distribution
           record['reddit_sentiment'] = np.random.beta(3, 2)  # Beta distribution (skewed positive)
           record['reddit_engagement'] = np.random.poisson(15)
           record['label'] = 1  # Legitimate
          
           synthetic_data.append(record)
      
       # Generate suspicious collections (30% of synthetic data)
       suspicious_count = target_samples - legitimate_count
       for _ in range(suspicious_count):
           record = {}
          
           # More realistic suspicious patterns with overlap to legitimate
           record['total_supply'] = np.random.randint(100, 200000)  # Can be very high or low
           record['is_verified'] = np.random.choice([0, 1], p=[0.9, 0.1])  # 90% unverified
           record['has_discord'] = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% have discord
           record['has_twitter'] = np.random.choice([0, 1], p=[0.5, 0.5])  # 50% have twitter
           record['trait_offers_enabled'] = np.random.choice([0, 1], p=[0.8, 0.2])
           record['collection_offers_enabled'] = np.random.choice([0, 1], p=[0.7, 0.3])
           record['floor_price'] = np.random.lognormal(-1, 1)  # Lower floor prices
           record['market_cap'] = np.random.lognormal(5, 2)  # Lower market cap
           record['total_volume'] = np.random.lognormal(6, 2)  # Lower volume
           record['num_owners'] = np.random.randint(5, 2000)  # Fewer owners
           record['average_price'] = record['floor_price'] * np.random.uniform(0.5, 3.0)
           record['reddit_mentions'] = np.random.poisson(1)  # Fewer mentions
           record['reddit_sentiment'] = np.random.beta(2, 3)  # Beta distribution (skewed negative)
           record['reddit_engagement'] = np.random.poisson(3)  # Lower engagement
           record['label'] = 0  # Suspicious
          
           synthetic_data.append(record)
      
       synthetic_df = pd.DataFrame(synthetic_data)
       
       # Add noise to prevent perfect separation
       for col in ['floor_price', 'market_cap', 'total_volume', 'average_price']:
           if col in synthetic_df.columns:
               noise = np.random.normal(0, synthetic_df[col].std() * 0.1, len(synthetic_df))
               synthetic_df[col] += noise
               synthetic_df[col] = np.maximum(synthetic_df[col], 0)  # Keep positive
       
       print(f"Generated {len(synthetic_df)} synthetic samples")
       print(f"   - Legitimate: {sum(synthetic_df['label'] == 1)}")
       print(f"   - Suspicious: {sum(synthetic_df['label'] == 0)}")
      
       return synthetic_df


def main():
   """
   Main function to run the NFT authenticity model
   """
   print("🚀 NFT Authenticity Model Training")
   print("=" * 50)
  
   # Initialize model
   model = NFTAuthenticityModel()
  
   try:
       # Load and prepare data
       X_processed, y, feature_names = model.prepare_data()
      
       if len(X_processed) < 10:
           print(f"Small dataset ({len(X_processed)} samples). Generating synthetic data...")
          
           # Create DataFrame from original features (not engineered ones) for synthetic generation
           import pandas as pd
           original_df = pd.DataFrame(X_processed[:, :len(feature_names)], columns=feature_names)
          
           # Generate synthetic data using the DataFrame method
           synthetic_df = model.generate_synthetic_data(original_df, target_samples=50)
          
           # Engineer features for synthetic data
           X_synthetic = model.engineer_features(synthetic_df)
           y_synthetic = model.create_synthetic_labels(synthetic_df, ['synthetic'] * len(synthetic_df))
          
           print(f"Original data shape: {X_processed.shape}")
           print(f"Synthetic data shape: {X_synthetic.shape}")
          
           # Ensure dimensions match
           if X_processed.shape[1] != X_synthetic.shape[1]:
               print(f"Dimension mismatch! Fixing...")
               min_features = min(X_processed.shape[1], X_synthetic.shape[1])
               X_processed = X_processed[:, :min_features]
               X_synthetic = X_synthetic[:, :min_features]
               print(f"Adjusted to {min_features} features")
          
           # Combine original and synthetic data
           X_processed = np.vstack([X_processed, X_synthetic])
           y = np.hstack([y, y_synthetic])
           print(f"Expanded to {len(X_processed)} samples")
      
       # Train models
       model.train_models(X_processed, y)
      
       # Evaluate models and get best model name
       best_model_name = model.evaluate_models()
       
       if best_model_name and best_model_name in model.models:
           model.best_model = model.models[best_model_name]
           model.save_best_model()
           print(f"\n✅ Model training complete!")
           print(f"Best model saved to model_outputs/")
       else:
           print("❌ No valid best model found")
  
   except Exception as e:
       print(f"Error in model training: {e}")
       import traceback
       traceback.print_exc()


if __name__ == "__main__":
   main()




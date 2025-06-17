import pandas as pd
import numpy as np
import json
import joblib
import os
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
        self.model_path = 'model_outputs/nft_authenticity_model.pkl'

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
        data_files = [f for f in os.listdir(data_dir) if f.startswith('nft_training_data_') and f.endswith('.json')]
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
        df = pd.DataFrame(features_list)
        
        # Store feature names
        self.feature_names = list(df.columns)
        print(f"Features: {self.feature_names}")
        
        # Create labels based on heuristics (since we don't have ground truth)
        y = self.create_synthetic_labels(df, collections)
        
        print("Creating synthetic labels based on heuristics...")
        print(f"Label distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = "Legitimate" if label == 1 else "Suspicious"
            print(f"  {label_name} ({label}): {count} ({count/len(y)*100:.1f}%)")
        
        # Feature engineering
        print("Preparing features...")
        X = self.engineer_features(df)
        print(f"Created {X.shape[1]} features after engineering")
        
        return X, y, self.feature_names

    def create_synthetic_labels(self, df, collections):
        """
        Create synthetic labels based on heuristics since we don't have ground truth
        """
        labels = []
        
        for idx, row in df.iterrows():
            score = 0
            
            # Verification status (strong indicator)
            if row.get('is_verified', 0) == 1:
                score += 3
            
            # Social media presence
            if row.get('has_discord', 0) == 1:
                score += 1
            if row.get('has_twitter', 0) == 1:
                score += 1
            
            # Market activity indicators
            if row.get('floor_price', 0) > 0:
                score += 2
            if row.get('total_volume', 0) > 1000:  # Significant volume
                score += 2
            if row.get('num_owners', 0) > 1000:  # Good distribution
                score += 1
            
            # Reddit engagement
            if row.get('reddit_mentions', 0) > 0:
                score += 1
            
            # Known legitimate collections (you can expand this list)
            collection = collections[idx] if idx < len(collections) else ''
            known_legit = ['boredapeyachtclub', 'cryptopunks', 'azuki', 'doodles-official']
            if collection in known_legit:
                score += 3
            
            # Label as legitimate if score >= 6, suspicious otherwise
            labels.append(1 if score >= 6 else 0)
        
        return np.array(labels)

    def engineer_features(self, df):
        """
        Create additional features from the raw data
        """
        # Start with original features
        X = df.values.astype(float)
        
        # Create additional features
        additional_features = []
        
        # Price-based features
        floor_price = df['floor_price'].values
        market_cap = df['market_cap'].values
        total_volume = df['total_volume'].values
        num_owners = df['num_owners'].values
        average_price = df['average_price'].values
        
        # Volume per owner (handle division by zero)
        volume_per_owner = np.where(num_owners > 0, total_volume / num_owners, 0)
        volume_per_owner = np.nan_to_num(volume_per_owner, nan=0.0, posinf=0.0, neginf=0.0)
        additional_features.append(volume_per_owner)
        
        # Market cap to volume ratio (handle division by zero)
        mc_volume_ratio = np.where(total_volume > 0, market_cap / total_volume, 0)
        mc_volume_ratio = np.nan_to_num(mc_volume_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        additional_features.append(mc_volume_ratio)
        
        # Price premium (average price vs floor price, handle division by zero)
        price_premium = np.where(floor_price > 0, average_price / floor_price, 1)
        price_premium = np.nan_to_num(price_premium, nan=1.0, posinf=1.0, neginf=1.0)
        additional_features.append(price_premium)
        
        # Social engagement score
        reddit_mentions = df['reddit_mentions'].values
        reddit_engagement = df['reddit_engagement'].values
        social_score = reddit_mentions + reddit_engagement
        additional_features.append(social_score)
        
        # Liquidity indicator (based on volume and owners)
        liquidity = np.sqrt(total_volume * num_owners)
        additional_features.append(liquidity)
        
        # Combine all features
        if additional_features:
            additional_features = np.column_stack(additional_features)
            X = np.column_stack([X, additional_features])
        
        return X
    
    def train_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Train multiple ML models
        """
        print("\nTraining multiple models...")
        
        # Check if we have enough data for train-test split
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = min(class_counts)
        
        print(f"Minimum class count: {min_class_count}")
        
        if min_class_count < 2 or len(X) < 10:
            print("⚠️  Warning: Very small dataset detected!")
            print("Using all data for training (no test split)")
            X_train, X_test, y_train, y_test = X, X, y, y
            use_cross_validation = False
        else:
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            use_cross_validation = True
        
        # Scale features for algorithms that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=42, 
                probability=True, 
                class_weight='balanced'
            )
        }
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for certain models
            if name in ['Logistic Regression', 'SVM']:
                X_train_used = X_train_scaled
                X_test_used = X_test_scaled
            else:
                X_train_used = X_train
                X_test_used = X_test
            
            # Train model
            model.fit(X_train_used, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_used)
            y_pred_proba = model.predict_proba(X_test_used)[:, 1]
            
            # Calculate metrics
            if use_cross_validation and len(np.unique(y_test)) > 1:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
            else:
                # For small datasets, use training accuracy as a rough estimate
                print(f"⚠️  Using training accuracy for {name} due to small dataset")
                y_train_pred = model.predict(X_train_used)
                accuracy = accuracy_score(y_train, y_train_pred)
                precision = precision_score(y_train, y_train_pred, zero_division=0)
                recall = recall_score(y_train, y_train_pred, zero_division=0)
                f1 = f1_score(y_train, y_train_pred, zero_division=0)
            
            # ROC AUC (handle case where only one class is present)
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                roc_auc = 0.5  # Default for single class
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.feature_names, 
                    model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(
                    self.feature_names, 
                    abs(model.coef_[0])
                ))
            
            # Store results
            self.results[name] = ModelResults(
                model_name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                confusion_matrix=cm,
                feature_importance=feature_importance
            )
            
            # Store model
            self.models[name] = model
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            print(f"  ROC AUC: {roc_auc:.3f}")
    
    def evaluate_models(self):
        """
        Print detailed evaluation of all models
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1_score,
                'ROC AUC': result.roc_auc
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Find best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best performing model: {best_model_name}")
        
        return best_model_name
    
    def plot_results(self, save_dir: str = "model_results"):
        """
        Create visualizations of model performance
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Model Comparison Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            values = [getattr(self.results[model], metric) for model in models]
            
            bars = ax.bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax.set_title(f'{name} by Model')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance Plot (for Random Forest)
        if 'Random Forest' in self.results and self.results['Random Forest'].feature_importance:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            importance = self.results['Random Forest'].feature_importance
            features = list(importance.keys())
            values = list(importance.values())
            
            # Sort by importance
            sorted_data = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_data)
            
            # Plot top 15 features
            top_n = min(15, len(features))
            y_pos = np.arange(top_n)
            
            ax.barh(y_pos, values[:top_n], color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features[:top_n])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top Feature Importances (Random Forest)')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()
    
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
        Predict authenticity for a new collection
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare features
        df = pd.DataFrame([collection_features])
        df = df.reindex(columns=self.feature_names, fill_value=0)
        df = self.prepare_features(df)
        
        # Scale if needed
        if model_name in ['Logistic Regression', 'SVM']:
            scaler = self.scalers['standard']
            features = scaler.transform(df)
        else:
            features = df
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return {
            'is_legitimate': bool(prediction),
            'legitimacy_probability': float(probability[1]),
            'risk_score': float(1 - probability[1]),
            'confidence': float(max(probability)),
            'model_used': model_name
        }
    
    def generate_synthetic_data(self, df, target_samples=50):
        """
        Generate synthetic NFT data for training when we have too few samples
        """
        print(f"\n🔄 Generating synthetic data to reach {target_samples} samples...")
        
        synthetic_data = []
        
        # Define realistic ranges for each feature based on real NFT data
        feature_ranges = {
            'total_supply': (100, 100000),
            'is_verified': [0, 1],
            'has_discord': [0, 1],
            'has_twitter': [0, 1],
            'trait_offers_enabled': [0, 1],
            'collection_offers_enabled': [0, 1],
            'floor_price': (0.001, 50.0),
            'market_cap': (100, 1000000),
            'total_volume': (1000, 5000000),
            'num_owners': (50, 10000),
            'average_price': (0.01, 100.0),
            'reddit_mentions': (0, 20),
            'reddit_sentiment': (0.1, 0.9),
            'reddit_engagement': (0, 50)
        }
        
        # Generate legitimate collections (70% of synthetic data)
        legitimate_count = int(target_samples * 0.7)
        for _ in range(legitimate_count):
            record = {}
            
            # Higher values for legitimate collections
            record['total_supply'] = np.random.randint(1000, 20000)
            record['is_verified'] = 1  # Mostly verified
            record['has_discord'] = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% have discord
            record['has_twitter'] = np.random.choice([0, 1], p=[0.1, 0.9])  # 90% have twitter
            record['trait_offers_enabled'] = np.random.choice([0, 1], p=[0.3, 0.7])
            record['collection_offers_enabled'] = np.random.choice([0, 1], p=[0.2, 0.8])
            record['floor_price'] = np.random.uniform(0.5, 30.0)
            record['market_cap'] = np.random.uniform(5000, 500000)
            record['total_volume'] = np.random.uniform(10000, 2000000)
            record['num_owners'] = np.random.randint(500, 8000)
            record['average_price'] = record['floor_price'] * np.random.uniform(1.1, 3.0)
            record['reddit_mentions'] = np.random.randint(1, 15)
            record['reddit_sentiment'] = np.random.uniform(0.4, 0.8)  # More positive sentiment
            record['reddit_engagement'] = np.random.randint(1, 30)
            record['label'] = 1  # Legitimate
            
            synthetic_data.append(record)
        
        # Generate suspicious collections (30% of synthetic data)
        suspicious_count = target_samples - legitimate_count
        for _ in range(suspicious_count):
            record = {}
            
            # Suspicious patterns
            record['total_supply'] = np.random.randint(100, 100000)  # Can be very high
            record['is_verified'] = np.random.choice([0, 1], p=[0.8, 0.2])  # Mostly unverified
            record['has_discord'] = np.random.choice([0, 1], p=[0.6, 0.4])  # Less likely to have discord
            record['has_twitter'] = np.random.choice([0, 1], p=[0.4, 0.6])  # Less likely to have twitter
            record['trait_offers_enabled'] = np.random.choice([0, 1], p=[0.7, 0.3])
            record['collection_offers_enabled'] = np.random.choice([0, 1], p=[0.6, 0.4])
            record['floor_price'] = np.random.uniform(0.001, 5.0)  # Lower floor prices
            record['market_cap'] = np.random.uniform(10, 10000)  # Much lower market cap
            record['total_volume'] = np.random.uniform(100, 50000)  # Lower volume
            record['num_owners'] = np.random.randint(10, 1000)  # Fewer owners
            record['average_price'] = record['floor_price'] * np.random.uniform(0.8, 2.0)
            record['reddit_mentions'] = np.random.randint(0, 8)  # Fewer mentions
            record['reddit_sentiment'] = np.random.uniform(0.1, 0.6)  # More negative sentiment
            record['reddit_engagement'] = np.random.randint(0, 10)  # Lower engagement
            record['label'] = 0  # Suspicious
            
            synthetic_data.append(record)
        
        synthetic_df = pd.DataFrame(synthetic_data)
        print(f"✅ Generated {len(synthetic_df)} synthetic samples")
        print(f"   - Legitimate: {sum(synthetic_df['label'] == 1)}")
        print(f"   - Suspicious: {sum(synthetic_df['label'] == 0)}")
        
        return synthetic_df

    def create_visualizations(self, X, y, feature_names):
        """
        Create visualizations for model analysis (optional, can be skipped)
        """
        try:
            print("📊 Creating visualizations...")
            
            # Make it faster by using a simpler backend
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode
            
            # Create output directory
            os.makedirs('model_outputs/visualizations', exist_ok=True)
                
        except Exception as e:
            print(f"⚠️ Visualization creation failed (skipping): {e}")
            print("This is okay - the model training will continue without visualizations.")

            # 1. Feature correlation heatmap (simplified)
            if len(feature_names) <= 20:  # Only for reasonable number of features
                plt.figure(figsize=(8, 6))
                correlation_matrix = np.corrcoef(X.T)
                plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                plt.colorbar()
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.savefig('model_outputs/visualizations/correlation_matrix.png', dpi=100, bbox_inches='tight')
                plt.close()
                print("✅ Saved correlation matrix")
        
        # 2. Simple feature importance (if we have a trained model)
        if hasattr(self, 'best_model') and self.best_model:
            try:
                if hasattr(self.best_model, 'feature_importances_'):
                    importances = self.best_model.feature_importances_
                    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(indices)), importances[indices])
                    plt.title('Top 10 Feature Importances')
                    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
                    plt.tight_layout()
                    plt.savefig('model_outputs/visualizations/feature_importance.png', dpi=100, bbox_inches='tight')
                    plt.close()
                    print("✅ Saved feature importance plot")
            except Exception as e:
                print(f"⚠️ Could not create feature importance plot: {e}")
        
        print("📊 Visualizations complete!")

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
            print(f"⚠️ Small dataset ({len(X_processed)} samples). Generating synthetic data...")
            
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
                print(f"⚠️ Dimension mismatch! Fixing...")
                min_features = min(X_processed.shape[1], X_synthetic.shape[1])
                X_processed = X_processed[:, :min_features]
                X_synthetic = X_synthetic[:, :min_features]
                print(f"Adjusted to {min_features} features")
            
            # Combine original and synthetic data
            X_processed = np.vstack([X_processed, X_synthetic])
            y = np.hstack([y, y_synthetic])
            print(f"✅ Expanded to {len(X_processed)} samples")
        
        # Train models
        model.train_models(X_processed, y)
        
        # Evaluate models
        model.evaluate_models()
        
        # Create simplified visualizations (optional)
        choice = input("\n📊 Create visualizations? (y/n, default=n): ").lower().strip()
        if choice == 'y':
            model.create_visualizations(X_processed, y, feature_names)
        else:
            print("⏭️ Skipping visualizations")
        
        # Save the best model
        best_model_name = model.evaluate_models()
        model.best_model = model.models[best_model_name]
        model.save_best_model()
        
        print("\n🎉 Model training complete!")
        print(f"📁 Best model saved to model_outputs/")
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

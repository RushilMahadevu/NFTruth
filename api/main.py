import sys
import os
from datetime import datetime

# Add the parent directory to the path so we can import from app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import pandas as pd

# Debug: Print paths
print(f"🔍 Current working directory: {os.getcwd()}")
print(f"🔍 Project root: {project_root}")
print(f"🔍 Expected model path: {os.path.join(project_root, 'model_outputs', 'best_nft_model.pkl')}")
print(f"🔍 Model file exists: {os.path.exists(os.path.join(project_root, 'model_outputs', 'best_nft_model.pkl'))}")

from app.predict import NFTPredictor

app = FastAPI(
    title="NFTruth API",
    description="NFT Collection Authenticity Prediction API",
    version="1.0.0"
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Astro domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor once when the server starts
predictor = NFTPredictor()

# Pydantic models for request/response
class CollectionInput(BaseModel):
    input: str
    
class PredictionResponse(BaseModel):
    collection: str
    collection_name: Optional[str]
    prediction: str
    confidence: float
    confidence_tier: Dict[str, Any]
    risk_score: float
    risk_level: Dict[str, Any]
    features_analyzed: Dict[str, Any]
    confidence_factors: Dict[str, Any]
    timestamp: str
    opensea_url: str
    error: Optional[str] = None

class SuggestionResponse(BaseModel):
    suggestions: List[Dict[str, Any]]

class SummaryResponse(BaseModel):
    summary: str
    total_predictions: int
    session_stats: Dict[str, Any]

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "NFTruth API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "suggestions": "/suggestions",
            "summary": "/summary",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_collection(input_data: CollectionInput):
    """
    Predict NFT collection authenticity
    
    - **input**: Collection name, slug, or OpenSea URL
    """
    try:
        # Normalize the input
        normalized_slug = predictor.normalize_input(input_data.input)
        
        if not normalized_slug:
            raise HTTPException(
                status_code=400, 
                detail="Could not process the input. Please provide a valid collection name, slug, or OpenSea URL."
            )
        
        # Make prediction
        result = predictor.predict_collection(normalized_slug)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/suggestions")
def get_suggestions(query: str, num_suggestions: int = 5):
    """
    Get collection suggestions based on query
    
    - **query**: Partial collection name or slug
    - **num_suggestions**: Number of suggestions to return (default: 5)
    """
    try:
        if not query or len(query.strip()) < 2:
            return {
                "query": query,
                "suggestions": [],
                "total_found": 0,
                "message": "Query too short. Please enter at least 2 characters."
            }
        
        suggestions = predictor.get_suggestions(query.strip(), num_suggestions)
        
        formatted_suggestions = []
        for slug, name, confidence in suggestions:
            formatted_suggestions.append({
                "slug": slug,
                "name": name,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.0f}%",
                "opensea_url": f"https://opensea.io/collection/{slug}",
                "match_type": "name" if confidence >= 0.8 else "partial"
            })
        
        return {
            "query": query.strip(),
            "suggestions": formatted_suggestions,
            "total_found": len(formatted_suggestions),
            "available_collections": len(predictor.prediction_history) if hasattr(predictor, 'prediction_history') else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@app.get("/summary", response_model=SummaryResponse)
def get_session_summary():
    """
    Get summary of all predictions made in this session
    """
    try:
        summary_text = predictor.generate_summary_report()
        
        # Calculate session stats
        total_predictions = len(predictor.prediction_history)
        
        if total_predictions > 0:
            legitimate_count = sum(1 for p in predictor.prediction_history if p['prediction'] == 'Legitimate')
            avg_confidence = sum(p['confidence'] for p in predictor.prediction_history) / total_predictions
            
            # Confidence tier breakdown
            trusted_count = sum(1 for p in predictor.prediction_history if p['confidence'] >= 70)
            caution_count = sum(1 for p in predictor.prediction_history if 40 <= p['confidence'] < 70)
            suspicious_count = sum(1 for p in predictor.prediction_history if p['confidence'] < 40)
            
            session_stats = {
                "total_predictions": total_predictions,
                "legitimate_count": legitimate_count,
                "suspicious_count": total_predictions - legitimate_count,
                "legitimate_percentage": (legitimate_count / total_predictions) * 100,
                "average_confidence": avg_confidence,
                "confidence_tiers": {
                    "trusted": trusted_count,
                    "caution": caution_count,
                    "suspicious": suspicious_count
                }
            }
        else:
            session_stats = {
                "total_predictions": 0,
                "message": "No predictions made yet"
            }
        
        return {
            "summary": summary_text,
            "total_predictions": total_predictions,
            "session_stats": session_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.get("/collections")
def list_known_collections(limit: int = 50):
    """
    List known collections that can be analyzed
    
    - **limit**: Maximum number of collections to return (default: 50)
    """
    try:
        from app.opensea_collections import COLLECTION_SLUGS
        
        collections = []
        for i, (name, slug) in enumerate(COLLECTION_SLUGS.items()):
            if i >= limit:
                break
            collections.append({
                "name": name,
                "slug": slug,
                "opensea_url": f"https://opensea.io/collection/{slug}"
            })
        
        return {
            "collections": collections,
            "total_shown": len(collections),
            "total_available": len(COLLECTION_SLUGS)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    try:
        # Check if model is loaded
        model_status = "loaded" if predictor.model is not None else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "reddit_collector": "enabled" if predictor.reddit_collector else "disabled",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

# Add this new endpoint after your existing endpoints
@app.get("/model-stats")
def get_model_statistics():
    """
    Get model performance statistics including accuracy
    """
    try:
        import json
        import os
        
        # Load model metadata which contains training accuracy
        metadata_path = os.path.join(project_root, 'model_outputs', 'model_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "model_accuracy": metadata.get('accuracy', 'N/A'),
                "model_precision": metadata.get('precision', 'N/A'),  
                "model_recall": metadata.get('recall', 'N/A'),
                "model_f1_score": metadata.get('f1_score', 'N/A'),
                "model_type": metadata.get('best_model', 'Unknown'),
                "training_date": metadata.get('training_date', 'Unknown'),
                "features_used": len(metadata.get('features_used', [])),
                "training_samples": metadata.get('training_samples', 'N/A')
            }
        else:
            return {
                "error": "Model metadata not found. Please retrain the model.",
                "accuracy": "N/A"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model stats: {str(e)}")

# Health check endpoint
@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "nftruth-api",
        "version": "1.0.0",
        "platform": "glitch",
        "message": "NFTruth API is running on Glitch!"
    }

# Keep-alive endpoint for Glitch
@app.get("/ping")
async def ping():
    """Keep-alive endpoint"""
    return {"pong": True, "timestamp": datetime.now().isoformat()}

# Prediction model
class PredictionRequest(BaseModel):
    collection_name: str

@app.post("/predict")
async def predict_nft(request: PredictionRequest):
    """Predict NFT authenticity"""
    try:
        # Import your model (lazy loading to avoid startup issues)
        from app.models.model import NFTAuthenticityModel
        
        model = NFTAuthenticityModel()
        
        # Your prediction logic here
        result = {
            "collection_name": request.collection_name,
            "prediction": "legitimate",  # Replace with actual prediction
            "confidence": 0.85,  # Replace with actual confidence
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Reddit analysis endpoint
@app.get("/analyze/{collection_name}")
async def analyze_collection(collection_name: str):
    """Analyze NFT collection"""
    try:
        # Your Reddit analysis logic
        return {
            "collection_name": collection_name,
            "reddit_sentiment": 0.7,
            "mentions": 25,
            "analysis": "positive",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
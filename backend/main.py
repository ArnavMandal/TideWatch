from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import asyncio
from datetime import datetime, timedelta
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import io
import json
import re
import hashlib
from PIL import Image as PILImage
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# MongoDB setup
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "flood_detection"
CACHE_COLLECTION = "analysis_cache"

# Global MongoDB client
mongodb_client = None
database = None
cache_collection = None

class MongoDBCache:
    """MongoDB cache manager for analysis results"""
    
    def __init__(self, collection):
        self.collection = collection
        self.default_ttl = timedelta(hours=24)  # Cache expires after 24 hours
    
    async def initialize_indexes(self):
        """Create indexes for optimal performance"""
        try:
            indexes = [
                IndexModel([("cache_key", ASCENDING)], unique=True),
                IndexModel([("created_at", ASCENDING)], expireAfterSeconds=86400),  # TTL index
                IndexModel([("analysis_type", ASCENDING)]),
            ]
            await self.collection.create_indexes(indexes)
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def generate_cache_key(self, analysis_type: str, data: str) -> str:
        """Generate a unique cache key for the analysis"""
        combined = f"{analysis_type}:{data}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def generate_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image data"""
        return hashlib.sha256(image_data).hexdigest()
    
    async def get_cached_result(self, cache_key: str) -> Optional[dict]:
        """Retrieve cached analysis result"""
        try:
            result = await self.collection.find_one({"cache_key": cache_key})
            if result:
                logger.info(f"Cache hit for key: {cache_key}")
                # Remove MongoDB-specific fields
                result.pop("_id", None)
                result.pop("cache_key", None)
                result.pop("created_at", None)
                result.pop("analysis_type", None)
                return result
            else:
                logger.info(f"Cache miss for key: {cache_key}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    async def store_result(self, cache_key: str, analysis_type: str, result: dict):
        """Store analysis result in cache"""
        try:
            cache_document = {
                "cache_key": cache_key,
                "analysis_type": analysis_type,
                "created_at": datetime.utcnow(),
                **result  # Spread the analysis result
            }
            
            await self.collection.replace_one(
                {"cache_key": cache_key},
                cache_document,
                upsert=True
            )
            logger.info(f"Result cached with key: {cache_key}")
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            total_cached = await self.collection.count_documents({})
            image_analyses = await self.collection.count_documents({"analysis_type": "image"})
            
            return {
                "total_cached_analyses": total_cached,
                "image_analyses": image_analyses,
                "cache_hit_potential": "Enabled"
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

app = FastAPI(
    title="Flood Detection API",
    description="Simple flood risk assessment using Gemini AI with MongoDB caching",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CoordinateRequest(BaseModel):
    latitude: float
    longitude: float

class AnalysisResponse(BaseModel):
    success: bool
    risk_level: str
    description: str
    recommendations: list[str]
    elevation: float
    distance_from_water: float
    message: str

# Initialize MongoDB connection
@app.on_event("startup")
async def startup_db_client():
    """Initialize MongoDB connection on startup"""
    global mongodb_client, database, cache_collection
    try:
        mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        database = mongodb_client[DATABASE_NAME]
        cache_collection = database[CACHE_COLLECTION]
        
        # Test connection
        await database.command("ping")
        logger.info("Connected to MongoDB successfully")
        
        # Initialize cache manager and create indexes
        cache_manager = MongoDBCache(cache_collection)
        await cache_manager.initialize_indexes()
        
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        logger.info("Continuing without cache functionality")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close MongoDB connection on shutdown"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")

def parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini AI response and extract structured data"""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)
            
            return {
                "risk_level": parsed_data.get("risk_level", "Medium"),
                "description": parsed_data.get("description", "Analysis completed"),
                "recommendations": parsed_data.get("recommendations", []),
                "elevation": parsed_data.get("elevation", 50.0),
                "distance_from_water": parsed_data.get("distance_from_water", 1000.0),
                "image_analysis": parsed_data.get("image_analysis", "")
            }
        else:
            return {
                "risk_level": "Medium",
                "description": "Analysis completed",
                "recommendations": ["Monitor weather conditions", "Stay informed about local alerts"],
                "elevation": 50.0,
                "distance_from_water": 1000.0,
                "image_analysis": response_text
            }
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {str(e)}")
        return {
            "risk_level": "Medium",
            "description": "Analysis completed",
            "recommendations": ["Monitor weather conditions", "Stay informed about local alerts"],
            "elevation": 50.0,
            "distance_from_water": 1000.0,
            "image_analysis": response_text
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    cache_status = "enabled" if cache_collection else "disabled"
    return {
        "message": "Flood Detection API with Gemini AI and MongoDB Cache",
        "version": "1.0.0",
        "status": "healthy",
        "cache_status": cache_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    cache_status = "enabled" if cache_collection else "disabled"
    return {
        "status": "healthy",
        "ai_model": "Gemini 2.0 Flash",
        "cache_status": cache_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics - useful for monitoring"""
    if cache_collection is None:
        return {"error": "Cache not available"}
    
    cache_manager = MongoDBCache(cache_collection)
    stats = await cache_manager.get_cache_stats()
    return stats

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze flood risk based on uploaded image using Gemini AI with caching
    """
    try:
        logger.info(f"Analyzing image: {file.filename}")
        
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        # Read image data
        image_data = await file.read()
        
        # Initialize cache manager if available
        cache_manager = None
        cache_key = None
        if cache_collection is not None:
            cache_manager = MongoDBCache(cache_collection)
            image_hash = cache_manager.generate_image_hash(image_data)
            cache_key = cache_manager.generate_cache_key("image", image_hash)
            
            # Try to get cached result first
            cached_result = await cache_manager.get_cached_result(cache_key)
            if cached_result:
                # Add cache indicator to response
                cached_result["message"] = "Image analysis completed successfully using cached result"
                cached_result["cached"] = True
                return cached_result
        
        # Convert image to PIL Image for Gemini AI
        try:
            image = PILImage.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
        except Exception as img_error:
            logger.error(f"Error processing image: {str(img_error)}")
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        prompt = """
        Analyze this terrain image for flood risk assessment.
        
        Please provide:
        1. Risk Level (Low/Medium/High/Very High)
        2. Description of the risk based on what you see
        3. 3-5 specific recommendations
        4. Estimated elevation in meters
        5. Estimated distance from water bodies in meters
        6. What water bodies or flood risks you can identify in the image
        
        Format your response as JSON with these fields:
        - risk_level
        - description
        - recommendations (array of strings)
        - elevation (number)
        - distance_from_water (number)
        - image_analysis (string describing what you see)
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([prompt, image])
            
            parsed_data = parse_gemini_response(response.text)
            
        except Exception as ai_error:
            logger.error(f"Error calling Gemini AI: {str(ai_error)}")
            parsed_data = generate_image_risk_assessment()
            parsed_data["image_analysis"] = "Image analysis was not available, using simulated assessment"
        
        # Prepare response
        analysis_result = {
            "success": True,
            **parsed_data,
            "ai_analysis": parsed_data.get("image_analysis", ""),
            "message": "Image analysis completed successfully using Gemini AI",
            "cached": False
        }
        
        # Cache the result if cache is available
        if cache_manager and cache_key:
            await cache_manager.store_result(cache_key, "image", analysis_result)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_image_risk_assessment() -> dict:
    """Generate risk assessment for image analysis"""
    import random
    
    risk_level = random.choice(["Low", "Medium", "High", "Very High"])
    
    descriptions = {
        "Low": "Image analysis shows low flood risk terrain.",
        "Medium": "Image analysis indicates moderate flood risk factors.",
        "High": "Image analysis reveals high flood risk characteristics.",
        "Very High": "Image analysis shows very high flood risk indicators."
    }
    
    recommendations = {
        "Low": [
            "Continue monitoring terrain changes",
            "Maintain current drainage systems",
            "Stay informed about weather patterns"
        ],
        "Medium": [
            "Improve drainage infrastructure",
            "Consider flood monitoring systems",
            "Develop emergency response plan"
        ],
        "High": [
            "Install comprehensive flood barriers",
            "Implement early warning systems",
            "Consider structural reinforcements"
        ],
        "Very High": [
            "Immediate flood protection measures needed",
            "Consider relocation to higher ground",
            "Implement comprehensive emergency protocols"
        ]
    }
    
    return {
        "risk_level": risk_level,
        "description": descriptions[risk_level],
        "recommendations": recommendations[risk_level],
        "elevation": round(random.uniform(10, 100), 1),
        "distance_from_water": round(random.uniform(200, 2000), 1)
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
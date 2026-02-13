from fastapi import FastAPI, APIRouter, HTTPException
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
from huggingface_hub import InferenceClient
from fastapi.middleware.cors import CORSMiddleware


import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict

# -------------------------------------------------------------------
# ENV + DB
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

if not MONGO_URL or not DB_NAME:
    raise RuntimeError("MongoDB config missing in .env")

mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client[DB_NAME]

hf_client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HF_TOKEN")
)


# -------------------------------------------------------------------
# LIFESPAN
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    mongo_client.close()

# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------

app = FastAPI(
    title="Pre-Production Guardrail API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://film-production-backend.onrender.com",
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_router = APIRouter(prefix="/api")

# -------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------

class EmotionalBeat(BaseModel):
    moment: str
    intensity: str
    description: str

class RiskFlags(BaseModel):
    alignment_sensitive: bool = False
    reshoot_prone: bool = False
    casting_sensitive: bool = False
    schedule_slip_risk: bool = False
    effort_payoff_mismatch: bool = False

class BudgetImpact(BaseModel):
    risk_driver: str
    schedule_risk: str
    estimated_time_buffer: str
    estimated_cost_risk: str
    confidence: str

class ScheduleImpact(BaseModel):
    extra_takes_likely: bool
    reset_time_per_take_min: int
    coverage_risk: str
    suggested_buffer_minutes: int

class SceneAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    complexity_score: str
    risk_level: str
    emotional_beats: List[EmotionalBeat]
    pacing_density: int
    character_complexity: str
    narrative_importance: int
    visual_sensitivity: str
    risk_flags: RiskFlags
    effort_vs_payoff: str
    producer_notes: str
    recommended_actions: List[str]
    risk_timing: str
    analysis_confidence: str
    relative_complexity_rank: Optional[str] = "average"
    budget_impact: Optional[BudgetImpact] = None
    cost_drivers: List[str] = Field(default_factory=list)
    schedule_impact: Optional[ScheduleImpact] = None
    producer_actions: List[str] = Field(default_factory=list)

class Scene(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scene_number: str
    location: str
    time_of_day: Optional[str] = None
    characters: List[str]
    scene_text: str
    analysis: Optional[SceneAnalysis] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analyzed_at: Optional[datetime] = None

class SceneCreate(BaseModel):
    scene_number: str
    location: str
    time_of_day: Optional[str] = None
    characters: List[str]
    scene_text: str

class SceneResponse(Scene):
    created_at: str
    analyzed_at: Optional[str]

# -------------------------------------------------------------------
# JSON EXTRACTION (SAFE)
# -------------------------------------------------------------------

def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON found")

    candidate = text[start:end + 1]
    return json.loads(candidate)

# -------------------------------------------------------------------
# AI ANALYSIS (HARDENED)
# -------------------------------------------------------------------

async def analyze_scene_with_hf(scene: Scene) -> SceneAnalysis:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a film production risk analyst.\n"
                "Return ONLY valid JSON.\n"
                "Do NOT add new keys.\n"
                "Do NOT nest objects.\n"
                "If unsure, use safe defaults."
            )
        },
        {
            "role": "user",
            "content": (
                "Analyze this scene using only the predefined fields.\n\n"
                f"Scene:\n{scene.scene_text}"
            )
        }
    ]

    response = hf_client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=messages,
        max_tokens=1200,
        temperature=0
    )

    raw = response.choices[0].message["content"]
    logging.info("HF RAW OUTPUT:\n%s", raw)

    # ---------- SAFETY NET ----------
    try:
        data = extract_json(raw)
        return SceneAnalysis(**data)

    except Exception as e:
        logging.error("STRUCTURE FAILED – USING FALLBACK")
        logging.error(e)

        # Fallback guarantees no 500
        return SceneAnalysis(
            complexity_score="medium",
            risk_level="medium",
            emotional_beats=[],
            pacing_density=5,
            character_complexity="medium",
            narrative_importance=5,
            visual_sensitivity="medium",
            risk_flags=RiskFlags(),
            effort_vs_payoff="balanced",
            producer_notes="AI output malformed. Fallback analysis used.",
            recommended_actions=["Manual producer review recommended"],
            risk_timing="pre-production",
            analysis_confidence="low",
            relative_complexity_rank="average",
            budget_impact=None,
            cost_drivers=[],
            schedule_impact=None,
            producer_actions=[]
        )

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@api_router.get("/")
async def root():
    return {"status": "Pre-Production Guardrail API running"}

@api_router.post("/scenes", response_model=SceneResponse)
async def create_scene(scene_input: SceneCreate):
    scene = Scene(**scene_input.model_dump())
    scene.analysis = await analyze_scene_with_hf(scene)
    scene.analyzed_at = datetime.now(timezone.utc)

    doc = scene.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["analyzed_at"] = doc["analyzed_at"].isoformat()

    await db.scenes.insert_one(doc)

    return SceneResponse(
        **scene.model_dump(exclude={"created_at", "analyzed_at"}),
        created_at=scene.created_at.isoformat(),
        analyzed_at=scene.analyzed_at.isoformat()
    )

@api_router.get("/scenes", response_model=List[SceneResponse])
async def get_scenes():
    scenes = await db.scenes.find({}, {"_id": 0}).to_list(1000)
    return [SceneResponse(**scene) for scene in scenes]

@api_router.get("/scenes/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str):
    scene = await db.scenes.find_one({"id": scene_id}, {"_id": 0})
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return SceneResponse(**scene)


@api_router.post("/scenes/{scene_id}/reanalyze", response_model=SceneResponse)
async def reanalyze_scene(scene_id: str):
    scene_doc = await db.scenes.find_one({"id": scene_id}, {"_id": 0})
    if not scene_doc:
        raise HTTPException(status_code=404, detail="Scene not found")

    scene = Scene(**scene_doc)
    scene.analysis = await analyze_scene_with_hf(scene)
    scene.analyzed_at = datetime.now(timezone.utc)

    doc = scene.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["analyzed_at"] = doc["analyzed_at"].isoformat()

    await db.scenes.replace_one({"id": scene_id}, doc)

    return SceneResponse(
        **scene.model_dump(exclude={"created_at", "analyzed_at"}),
        created_at=scene.created_at.isoformat(),
        analyzed_at=scene.analyzed_at.isoformat()
    )

@api_router.delete("/scenes/{scene_id}")
async def delete_scene(scene_id: str):
    result = await db.scenes.delete_one({"id": scene_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Scene not found")
    return {"message": "Scene deleted"}

# -------------------------------------------------------------------
# MIDDLEWARE
# -------------------------------------------------------------------

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

"""
FastAPI Everything — One File
Run: uvicorn fastapi_everything:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query, Path, Header, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import time
import uuid
import jwt  # pip install PyJWT

# ════════════════════════════════════════════════════
# APP SETUP
# ════════════════════════════════════════════════════

app = FastAPI(
    title="ML Platform API",
    description="Full FastAPI example — models, auth, validation, background tasks",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# CORS — allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════
# PYDANTIC MODELS (Request / Response schemas)
# ════════════════════════════════════════════════════

class ModelType(str, Enum):
    classification = "classification"
    regression     = "regression"
    clustering     = "clustering"

class TrainRequest(BaseModel):
    name:          str             = Field(..., min_length=1, max_length=100, example="my_model")
    model_type:    ModelType       = Field(..., example="classification")
    learning_rate: float           = Field(0.001, gt=0, lt=1, example=0.001)
    epochs:        int             = Field(10, ge=1, le=1000, example=50)
    features:      List[str]       = Field(..., min_items=1, example=["age", "salary"])
    params:        Dict[str, Any]  = Field(default_factory=dict)

    @validator("name")
    def name_must_be_alphanumeric(cls, v):
        if not v.replace("_", "").isalnum():
            raise ValueError("name must be alphanumeric (underscores allowed)")
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "churn_model",
                "model_type": "classification",
                "learning_rate": 0.001,
                "epochs": 50,
                "features": ["age", "salary", "tenure"],
                "params": {"max_depth": 5}
            }
        }

class TrainResponse(BaseModel):
    job_id:     str
    model_name: str
    status:     str
    created_at: datetime

class PredictRequest(BaseModel):
    model_id:  str
    input_data: List[Dict[str, Any]] = Field(..., min_items=1)

class PredictResponse(BaseModel):
    model_id:    str
    predictions: List[Any]
    confidence:  Optional[List[float]] = None
    latency_ms:  float

class UserCreate(BaseModel):
    username: str  = Field(..., min_length=3, max_length=50)
    email:    str
    password: str  = Field(..., min_length=8)

class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"

class UserResponse(BaseModel):
    id:       str
    username: str
    email:    str
    created_at: datetime


# ════════════════════════════════════════════════════
# FAKE DB (in production use SQLAlchemy / MongoDB)
# ════════════════════════════════════════════════════

fake_users_db: Dict[str, dict] = {}
fake_models_db: Dict[str, dict] = {}
fake_jobs_db: Dict[str, dict] = {}


# ════════════════════════════════════════════════════
# AUTH — JWT
# ════════════════════════════════════════════════════

SECRET_KEY    = "your-secret-key-here"   # use env variable in production!
ALGORITHM     = "HS256"
TOKEN_EXPIRES = 30  # minutes

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def create_access_token(data: dict, expires_delta: int = TOKEN_EXPIRES) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=expires_delta)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = verify_token(token)
    user_id = payload.get("sub")
    user = fake_users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ════════════════════════════════════════════════════
# DEPENDENCIES
# ════════════════════════════════════════════════════

async def get_db():
    """Simulates DB session dependency"""
    db = {"connection": "active"}
    try:
        yield db
    finally:
        pass   # close connection here

def verify_api_key(x_api_key: str = Header(None)):
    """Optional API key dependency"""
    valid_keys = {"dev-key-123", "prod-key-456"}
    if x_api_key and x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

def get_pagination(
    page:     int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
):
    return {"skip": (page - 1) * per_page, "limit": per_page, "page": page}


# ════════════════════════════════════════════════════
# MIDDLEWARE
# ════════════════════════════════════════════════════

@app.middleware("http")
async def add_timing_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start) * 1000:.2f}ms"
    return response

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"[{datetime.now().isoformat()}] {request.method} {request.url.path}")
    return await call_next(request)


# ════════════════════════════════════════════════════
# STARTUP / SHUTDOWN EVENTS
# ════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    print("App starting — connecting to DB, loading models...")
    # load ML models into memory here
    # db = await create_db_pool()

@app.on_event("shutdown")
async def shutdown():
    print("App shutting down — cleanup...")


# ════════════════════════════════════════════════════
# BASIC ROUTES
# ════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """Health check"""
    return {"status": "ok", "time": datetime.now().isoformat(), "version": "1.0.0"}

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "db": "connected", "model": "loaded"}


# ════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════

@app.post("/auth/register", response_model=UserResponse, status_code=201, tags=["Auth"])
async def register(user: UserCreate):
    """Register a new user"""
    # Check if email already exists
    for u in fake_users_db.values():
        if u["email"] == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    new_user = {
        "id":         user_id,
        "username":   user.username,
        "email":      user.email,
        "password":   user.password,    # hash in production! use bcrypt
        "created_at": datetime.now(),
    }
    fake_users_db[user_id] = new_user
    return new_user

@app.post("/auth/login", response_model=Token, tags=["Auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Login — returns JWT token"""
    # find user
    user = None
    for u in fake_users_db.values():
        if u["username"] == form.username and u["password"] == form.password:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": user["id"], "username": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse, tags=["Auth"])
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current logged-in user"""
    return current_user


# ════════════════════════════════════════════════════
# MODEL ROUTES
# ════════════════════════════════════════════════════

@app.post("/models/train", response_model=TrainResponse, status_code=202, tags=["Models"])
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """Start model training (runs in background)"""

    def do_training(job_id: str, config: dict):
        print(f"Training job {job_id} started...")
        time.sleep(2)   # simulate training
        fake_jobs_db[job_id]["status"] = "completed"
        fake_jobs_db[job_id]["accuracy"] = 0.94
        print(f"Training job {job_id} done!")

    job_id = str(uuid.uuid4())
    fake_jobs_db[job_id] = {
        "job_id":     job_id,
        "model_name": request.name,
        "status":     "running",
        "created_at": datetime.now(),
        "user_id":    current_user["id"],
    }

    background_tasks.add_task(do_training, job_id, request.dict())

    return TrainResponse(
        job_id     = job_id,
        model_name = request.name,
        status     = "running",
        created_at = datetime.now(),
    )

@app.get("/models/jobs/{job_id}", tags=["Models"])
async def get_job_status(
    job_id: str = Path(..., description="Training job ID"),
    current_user: dict = Depends(get_current_user),
):
    """Check training job status"""
    job = fake_jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job

@app.post("/models/predict", response_model=PredictResponse, tags=["Models"])
async def predict(
    request: PredictRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run predictions using a trained model"""
    start = time.perf_counter()

    # Simulate prediction
    predictions = [1 if i % 2 == 0 else 0 for i in range(len(request.input_data))]
    confidence  = [0.92, 0.87, 0.95][:len(request.input_data)]

    latency = (time.perf_counter() - start) * 1000

    return PredictResponse(
        model_id    = request.model_id,
        predictions = predictions,
        confidence  = confidence,
        latency_ms  = round(latency, 2),
    )

@app.get("/models", tags=["Models"])
async def list_models(
    model_type: Optional[ModelType] = Query(None, description="Filter by type"),
    search:     Optional[str]       = Query(None, description="Search by name"),
    pagination: dict = Depends(get_pagination),
    current_user: dict = Depends(get_current_user),
):
    """List all models with filtering and pagination"""
    models = list(fake_models_db.values())

    if model_type:
        models = [m for m in models if m.get("type") == model_type]
    if search:
        models = [m for m in models if search.lower() in m.get("name", "").lower()]

    skip, limit = pagination["skip"], pagination["limit"]
    paginated = models[skip: skip + limit]

    return {
        "total":   len(models),
        "page":    pagination["page"],
        "results": paginated,
    }

@app.delete("/models/{model_id}", status_code=204, tags=["Models"])
async def delete_model(
    model_id: str = Path(..., description="Model ID to delete"),
    current_user: dict = Depends(get_current_user),
):
    """Delete a model"""
    if model_id not in fake_models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    del fake_models_db[model_id]
    # 204 = no content (nothing returned)


# ════════════════════════════════════════════════════
# FILE UPLOAD
# ════════════════════════════════════════════════════

@app.post("/upload/dataset", tags=["Files"])
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a CSV dataset"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    contents = await file.read()
    size_kb = len(contents) / 1024

    # Save to disk
    save_path = f"uploads/{file.filename}"
    # with open(save_path, "wb") as f:
    #     f.write(contents)

    return {
        "filename":  file.filename,
        "size_kb":   round(size_kb, 2),
        "saved_to":  save_path,
        "message":   "Upload successful",
    }


# ════════════════════════════════════════════════════
# CUSTOM EXCEPTION HANDLERS
# ════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error":   True,
            "code":    exc.status_code,
            "message": exc.detail,
            "path":    str(request.url.path),
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error":   True,
            "code":    500,
            "message": "Internal server error",
        }
    )


# ════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_everything:app", host="0.0.0.0", port=8000, reload=True)

# ────────────────────────────────────────────────────
# HOW TO TEST (in terminal):
#
# 1. Start server:
#    uvicorn fastapi_everything:app --reload
#
# 2. Register:
#    curl -X POST http://localhost:8000/auth/register \
#      -H "Content-Type: application/json" \
#      -d '{"username":"alice","email":"alice@test.com","password":"secret123"}'
#
# 3. Login:
#    curl -X POST http://localhost:8000/auth/login \
#      -d "username=alice&password=secret123"
#    → copy the access_token
#
# 4. Train model:
#    curl -X POST http://localhost:8000/models/train \
#      -H "Authorization: Bearer YOUR_TOKEN" \
#      -H "Content-Type: application/json" \
#      -d '{"name":"my_model","model_type":"classification","features":["age","salary"],"epochs":10}'
#
# 5. Predict:
#    curl -X POST http://localhost:8000/models/predict \
#      -H "Authorization: Bearer YOUR_TOKEN" \
#      -H "Content-Type: application/json" \
#      -d '{"model_id":"abc123","input_data":[{"age":25,"salary":50000}]}'
#
# 6. Open Swagger UI: http://localhost:8000/docs
# ────────────────────────────────────────────────────

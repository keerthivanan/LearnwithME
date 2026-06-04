"""
FastAPI Full Course — Taught Simply
====================================
Install: pip install fastapi uvicorn python-jose[cryptography] passlib bcrypt python-multipart
Run:     uvicorn fastapi_full_course:app --reload
Docs:    http://localhost:8000/docs   ← try all routes here
"""

# ════════════════════════════════════════════════════════════
# WHAT IS FASTAPI?
# ════════════════════════════════════════════════════════════
#
#  FastAPI = a Python framework to build APIs (web endpoints)
#
#  Why FastAPI over Flask/Django?
#  ✓ Fastest Python web framework
#  ✓ Auto-generates docs (Swagger UI) at /docs
#  ✓ Built-in data validation via Pydantic
#  ✓ Async support out of the box
#  ✓ Type hints everywhere = less bugs
#
#  Think of FastAPI as:
#  You build ENDPOINTS (URLs) that receive data, process it, return data.
#  Example: POST /predict → send features → get prediction back
#
# ════════════════════════════════════════════════════════════

from fastapi import (
    FastAPI, HTTPException, Depends, status,
    Query, Path, Header, BackgroundTasks,
    UploadFile, File, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import time, uuid, jwt, hashlib


# ════════════════════════════════════════════════════════════
# STEP 1 — CREATE THE APP
# ════════════════════════════════════════════════════════════

app = FastAPI(
    title="FastAPI Full Course",
    description="Learn FastAPI step by step",
    version="1.0.0"
)

# ════════════════════════════════════════════════════════════
# STEP 2 — YOUR FIRST ROUTE (GET)
# ════════════════════════════════════════════════════════════
#
#  Route = a URL + a function that handles it
#
#  @app.get("/")    → handles GET requests to "/"
#  @app.post("/")   → handles POST requests to "/"
#
#  Return a dict → FastAPI converts it to JSON automatically

@app.get("/")
def home():
    return {"message": "Hello World!"}
    # Response: {"message": "Hello World!"}

@app.get("/about")
def about():
    return {
        "app": "ML Platform",
        "version": "1.0",
        "author": "Keerthi"
    }

# ════════════════════════════════════════════════════════════
# STEP 3 — PATH PARAMETERS
# ════════════════════════════════════════════════════════════
#
#  Path parameter = part of the URL itself
#  URL:  /users/42      → user_id = 42
#  URL:  /items/laptop  → item_name = "laptop"

@app.get("/users/{user_id}")
def get_user(user_id: int):   # FastAPI auto-converts "42" → 42 (int)
    return {"user_id": user_id, "name": f"User {user_id}"}
    # GET /users/42  → {"user_id": 42, "name": "User 42"}
    # GET /users/abc → 422 Error (not an int!)

@app.get("/items/{item_name}")
def get_item(item_name: str):
    return {"item": item_name.upper()}
    # GET /items/laptop → {"item": "LAPTOP"}

# Path with validation
@app.get("/products/{product_id}")
def get_product(product_id: int = Path(..., ge=1, le=9999)):
    # ge=1 means >= 1, le=9999 means <= 9999
    return {"product_id": product_id}
    # GET /products/0   → 422 Error (must be >= 1)
    # GET /products/500 → {"product_id": 500}

# ════════════════════════════════════════════════════════════
# STEP 4 — QUERY PARAMETERS
# ════════════════════════════════════════════════════════════
#
#  Query parameter = comes after ? in the URL
#  URL:  /search?q=python&limit=10
#  URL:  /products?category=books&min_price=100

@app.get("/search")
def search(
    q:        str,                          # required
    limit:    int   = 10,                   # optional, default 10
    offset:   int   = 0,                    # optional, default 0
    active:   bool  = True,                 # optional, default True
    category: Optional[str] = None,         # optional, can be None
):
    return {
        "query":    q,
        "limit":    limit,
        "offset":   offset,
        "active":   active,
        "category": category,
    }
    # GET /search?q=python        → {query:python, limit:10, offset:0, active:true}
    # GET /search?q=ml&limit=5&category=ai → {query:ml, limit:5, category:ai}

# Query with validation
@app.get("/scores")
def get_scores(
    min_score: float = Query(0,   ge=0,   le=100),
    max_score: float = Query(100, ge=0,   le=100),
    sort_by:   str   = Query("score", regex="^(score|name|date)$"),
):
    return {"min": min_score, "max": max_score, "sort": sort_by}

# ════════════════════════════════════════════════════════════
# STEP 5 — REQUEST BODY (POST / PUT)
# ════════════════════════════════════════════════════════════
#
#  For POST/PUT → client sends data in the request BODY (JSON)
#  We define the shape using Pydantic BaseModel
#  FastAPI validates automatically!

class Item(BaseModel):
    name:        str
    price:       float
    in_stock:    bool = True
    tags:        List[str] = []

@app.post("/items")
def create_item(item: Item):
    # item is already validated and converted
    return {
        "message": "Item created!",
        "item":    item,
        "id":      str(uuid.uuid4()),
    }
    # POST /items
    # Body: {"name": "Laptop", "price": 999.99, "tags": ["electronics"]}
    # Response: {"message":"Item created!", "item":{...}, "id":"abc-123"}
    #
    # Body: {"name": "", "price": -1}  → 422 Validation Error (price must be number)

# ════════════════════════════════════════════════════════════
# STEP 6 — PYDANTIC MODELS (data validation)
# ════════════════════════════════════════════════════════════
#
#  Pydantic = validates + converts your data automatically
#  Think of it as a smart data class with rules

class UserRole(str, Enum):
    admin  = "admin"
    user   = "user"
    viewer = "viewer"

class UserCreate(BaseModel):
    username:  str   = Field(..., min_length=3, max_length=50, example="alice")
    email:     str   = Field(..., example="alice@gmail.com")
    password:  str   = Field(..., min_length=8)
    age:       int   = Field(..., ge=18, le=120)
    role:      UserRole = UserRole.user

    # Custom validator
    @validator("email")
    def email_must_have_at(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email — must contain @")
        return v.lower()   # normalize to lowercase

    @validator("username")
    def username_no_spaces(cls, v):
        if " " in v:
            raise ValueError("Username cannot have spaces")
        return v

class UserResponse(BaseModel):
    id:         str
    username:   str
    email:      str
    role:       UserRole
    created_at: datetime
    # Notice: no 'password' field → never returned to client!

@app.post("/users", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate):
    new_user = {
        "id":         str(uuid.uuid4()),
        "username":   user.username,
        "email":      user.email,
        "role":       user.role,
        "created_at": datetime.now(),
        # password is stored (hashed in real app) but NOT in response
    }
    return new_user
    # POST /users
    # Body: {"username":"alice","email":"ALICE@GMAIL.COM","password":"secret123","age":25}
    # Response: {"id":"...","username":"alice","email":"alice@gmail.com","role":"user","created_at":"..."}
    # (email is lowercased automatically, password not shown)

# ════════════════════════════════════════════════════════════
# STEP 7 — HTTP METHODS + STATUS CODES
# ════════════════════════════════════════════════════════════
#
#  GET    → fetch data          → 200 OK
#  POST   → create something    → 201 Created
#  PUT    → replace entirely    → 200 OK
#  PATCH  → partial update      → 200 OK
#  DELETE → delete something    → 204 No Content

# Fake database
db: Dict[str, dict] = {}

class Product(BaseModel):
    name:     str
    price:    float
    category: str

class ProductUpdate(BaseModel):
    name:     Optional[str]  = None   # all optional for PATCH
    price:    Optional[float]= None
    category: Optional[str]  = None

@app.get("/products", status_code=200)
def list_products():
    return {"products": list(db.values()), "total": len(db)}

@app.post("/products", status_code=201)
def add_product(product: Product):
    product_id = str(uuid.uuid4())
    db[product_id] = {"id": product_id, **product.dict()}
    return db[product_id]

@app.get("/products/{product_id}", status_code=200)
def get_one_product(product_id: str):
    if product_id not in db:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    return db[product_id]

@app.put("/products/{product_id}", status_code=200)
def replace_product(product_id: str, product: Product):
    # PUT = replace entire object
    if product_id not in db:
        raise HTTPException(status_code=404, detail="Product not found")
    db[product_id] = {"id": product_id, **product.dict()}
    return db[product_id]

@app.patch("/products/{product_id}", status_code=200)
def update_product(product_id: str, product: ProductUpdate):
    # PATCH = update only provided fields
    if product_id not in db:
        raise HTTPException(status_code=404, detail="Product not found")
    existing = db[product_id]
    update_data = product.dict(exclude_unset=True)  # only fields that were sent
    existing.update(update_data)
    db[product_id] = existing
    return db[product_id]

@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: str):
    if product_id not in db:
        raise HTTPException(status_code=404, detail="Product not found")
    del db[product_id]
    # 204 = no content, return nothing

# ════════════════════════════════════════════════════════════
# STEP 8 — ERROR HANDLING
# ════════════════════════════════════════════════════════════
#
#  HTTPException → raise this for expected errors (not found, forbidden)
#  Status codes to know:
#    200 = OK
#    201 = Created
#    204 = No Content
#    400 = Bad Request (wrong input)
#    401 = Unauthorized (not logged in)
#    403 = Forbidden (logged in but no permission)
#    404 = Not Found
#    409 = Conflict (already exists)
#    422 = Unprocessable Entity (validation failed)
#    500 = Internal Server Error

@app.get("/divide/{a}/{b}")
def divide(a: float, b: float):
    if b == 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot divide by zero"
        )
    return {"result": a / b}
    # GET /divide/10/2  → {"result": 5.0}
    # GET /divide/10/0  → 400 {"detail": "Cannot divide by zero"}

# Custom exception with extra info
@app.get("/strict/{value}")
def strict_endpoint(value: int):
    if value < 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "negative_value", "message": "Must be positive", "got": value}
        )
    return {"value": value}

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error":   exc.detail,
            "code":    exc.status_code,
        }
    )

# ════════════════════════════════════════════════════════════
# STEP 9 — DEPENDENCIES (Depends)
# ════════════════════════════════════════════════════════════
#
#  Dependencies = reusable functions injected into routes
#  Use for: auth checks, DB sessions, pagination, API key checks
#
#  Think of it like: before running the route function,
#  first run these helper functions.

def get_pagination(
    page:     int = Query(1,  ge=1,  description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
):
    return {
        "skip":  (page - 1) * per_page,
        "limit": per_page,
        "page":  page
    }

def verify_api_key(x_api_key: str = Header(None)):
    valid_keys = {"dev-key-123", "prod-key-456"}
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

@app.get("/paginated-items")
def get_paginated(pagination: dict = Depends(get_pagination)):
    all_items = [{"id": i, "name": f"Item {i}"} for i in range(1, 51)]
    skip, limit = pagination["skip"], pagination["limit"]
    return {
        "page":    pagination["page"],
        "results": all_items[skip: skip + limit],
    }
    # GET /paginated-items?page=2&per_page=5
    # → page 2, 5 items (items 6-10)

@app.get("/protected")
def protected_route(api_key: str = Depends(verify_api_key)):
    return {"message": "You are authorized!", "key": api_key}
    # GET /protected + Header: x-api-key: dev-key-123  → success
    # GET /protected + Header: x-api-key: wrong        → 403

# ════════════════════════════════════════════════════════════
# STEP 10 — AUTHENTICATION (JWT Tokens)
# ════════════════════════════════════════════════════════════
#
#  HOW JWT AUTH WORKS:
#  1. User sends username + password to /login
#  2. Server checks, creates a TOKEN (JWT), sends back
#  3. User stores the token
#  4. For every protected request, user sends the token in header
#     Authorization: Bearer eyJhbG...
#  5. Server verifies token, allows access

SECRET_KEY = "super-secret-key-change-in-production"
ALGORITHM  = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Simple user store (use a real DB in production)
users_store = {
    "alice": {"username": "alice", "password": "pass123", "role": "admin"},
    "bob":   {"username": "bob",   "password": "pass456", "role": "user"},
}

def hash_password(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()

def create_token(data: dict, expire_minutes: int = 60) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(minutes=expire_minutes)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please login again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload  = decode_token(token)
    username = payload.get("sub")
    user = users_store.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only!")
    return user

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    # OAuth2PasswordRequestForm expects: username + password (form data)
    user = users_store.get(form.username)
    if not user or user["password"] != form.password:
        raise HTTPException(status_code=401, detail="Wrong username or password")

    token = create_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}
    # POST /login  form: username=alice&password=pass123
    # → {"access_token": "eyJhbG...", "token_type": "bearer"}

@app.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}
    # GET /me  Header: Authorization: Bearer <token>
    # → {"username": "alice", "role": "admin"}

@app.get("/admin/dashboard")
def admin_dashboard(user: dict = Depends(require_admin)):
    return {"message": f"Welcome to admin dashboard, {user['username']}!"}
    # Only works if logged in as 'alice' (admin role)
    # 'bob' gets 403 Forbidden

# ════════════════════════════════════════════════════════════
# STEP 11 — BACKGROUND TASKS
# ════════════════════════════════════════════════════════════
#
#  Background tasks run AFTER the response is sent
#  Use for: sending emails, training models, logging, cleanup
#  Client gets immediate response, task runs in background

def send_email(email: str, subject: str):
    time.sleep(3)   # simulate slow email
    print(f"Email sent to {email}: {subject}")

def log_action(action: str, user: str):
    print(f"[LOG] {datetime.now()} | {user} | {action}")

@app.post("/register")
def register_user(
    username:         str,
    email:            str,
    background_tasks: BackgroundTasks,
):
    # Save user to DB (fast)
    # ...

    # Queue emails/logs (non-blocking — client doesn't wait for these)
    background_tasks.add_task(send_email, email, "Welcome to ML Platform!")
    background_tasks.add_task(log_action, "user_registered", username)

    return {"message": "Registered! Welcome email will arrive shortly."}
    # Client gets response IMMEDIATELY
    # Email is sent in background (doesn't slow down the API)

# ════════════════════════════════════════════════════════════
# STEP 12 — FILE UPLOAD
# ════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()       # read file bytes
    size_kb  = len(contents) / 1024

    # Validate file type
    if not file.filename.endswith((".csv", ".json", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV, JSON, TXT allowed")

    return {
        "filename":    file.filename,
        "content_type": file.content_type,
        "size_kb":     round(size_kb, 2),
        "message":     "File uploaded!"
    }

@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        results.append({"name": file.filename, "size_kb": len(contents)/1024})
    return {"files": results, "count": len(results)}

# Form data + file together
@app.post("/upload-with-info")
async def upload_with_info(
    description: str  = Form(...),
    category:    str  = Form(...),
    file:        UploadFile = File(...),
):
    contents = await file.read()
    return {
        "description": description,
        "category":    category,
        "filename":    file.filename,
        "size_kb":     round(len(contents)/1024, 2),
    }

# ════════════════════════════════════════════════════════════
# STEP 13 — MIDDLEWARE
# ════════════════════════════════════════════════════════════
#
#  Middleware runs on EVERY request and response
#  Use for: timing, logging, adding headers, auth checks

# CORS Middleware — allows frontend (React) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods
    allow_headers=["*"],    # allow all headers
)

# Custom middleware — adds timing header to every response
@app.middleware("http")
async def add_process_time(request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)                        # run the route
    elapsed  = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"  # add header
    return response

# Custom middleware — request logging
@app.middleware("http")
async def log_every_request(request, call_next):
    print(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"← {response.status_code}")
    return response

# ════════════════════════════════════════════════════════════
# STEP 14 — ASYNC ROUTES
# ════════════════════════════════════════════════════════════
#
#  Use 'async def' when your route:
#  - Calls an external API
#  - Reads from a database
#  - Does any I/O operation
#
#  Use normal 'def' for CPU-heavy work

import asyncio

@app.get("/async-example")
async def async_route():
    # While waiting for these, other requests can be served!
    result1, result2 = await asyncio.gather(
        asyncio.sleep(1),   # simulate DB query 1
        asyncio.sleep(1),   # simulate DB query 2
    )
    # Both finished in ~1 second, not 2!
    return {"message": "Fetched two things in parallel!"}

# ════════════════════════════════════════════════════════════
# STEP 15 — ROUTERS (organize large apps)
# ════════════════════════════════════════════════════════════
#
#  In real projects, split routes into multiple files
#  Then include them with a prefix

from fastapi import APIRouter

# models_router.py (separate file in real project)
models_router = APIRouter(prefix="/models", tags=["ML Models"])

@models_router.get("/")
def list_models():
    return {"models": ["ResNet", "BERT", "XGBoost"]}

@models_router.post("/{model_id}/train")
def train_model(model_id: str, user: dict = Depends(get_current_user)):
    return {"message": f"Training {model_id} started!", "user": user["username"]}

@models_router.get("/{model_id}/predict")
def predict(model_id: str, user: dict = Depends(get_current_user)):
    return {"model": model_id, "prediction": 0.94}

# Include router in main app
app.include_router(models_router)
# Now all routes available at: /models/, /models/{id}/train, etc.

# ════════════════════════════════════════════════════════════
# STEP 16 — STARTUP & SHUTDOWN EVENTS
# ════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup():
    print("App starting...")
    # Load ML model into memory
    # Connect to database
    # Start background workers
    print("ML model loaded, DB connected!")

@app.on_event("shutdown")
async def on_shutdown():
    print("App shutting down...")
    # Close DB connections
    # Save state
    print("Cleanup done.")

# ════════════════════════════════════════════════════════════
# HOW TO TEST YOUR API
# ════════════════════════════════════════════════════════════
#
#  Option 1: Swagger UI (easiest)
#  Open: http://localhost:8000/docs
#  Click any route → Try it out → Execute
#
#  Option 2: curl (terminal)
#  GET:    curl http://localhost:8000/
#  POST:   curl -X POST http://localhost:8000/items \
#            -H "Content-Type: application/json" \
#            -d '{"name":"Laptop","price":999}'
#
#  Option 3: Python requests
import requests   # pip install requests

def test_api():
    base = "http://localhost:8000"

    # Test home
    r = requests.get(f"{base}/")
    print(r.json())   # {"message": "Hello World!"}

    # Test path param
    r = requests.get(f"{base}/users/42")
    print(r.json())   # {"user_id": 42, "name": "User 42"}

    # Test query param
    r = requests.get(f"{base}/search?q=python&limit=5")
    print(r.json())

    # Test POST with body
    r = requests.post(f"{base}/items", json={"name":"Laptop","price":999.99})
    print(r.json())   # {"message":"Item created!", ...}

    # Test login + protected route
    r = requests.post(f"{base}/login", data={"username":"alice","password":"pass123"})
    token = r.json()["access_token"]

    r = requests.get(f"{base}/me", headers={"Authorization": f"Bearer {token}"})
    print(r.json())   # {"username": "alice", "role": "admin"}

# ════════════════════════════════════════════════════════════
# RUN THE APP
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_full_course:app", host="0.0.0.0", port=8000, reload=True)

# ════════════════════════════════════════════════════════════
# QUICK SUMMARY
# ════════════════════════════════════════════════════════════
#
#  @app.get("/path")      → fetch data
#  @app.post("/path")     → create data
#  @app.put("/path/{id}") → replace data
#  @app.patch("/path/{id}")→ partial update
#  @app.delete("/path/{id}")→ delete
#
#  Path params:  /users/{user_id}  → user_id in function args
#  Query params: /search?q=x       → q: str in function args
#  Body:         BaseModel class   → validated auto
#
#  Depends()    → reusable logic (auth, pagination, DB)
#  Middleware   → runs on every request
#  Background   → runs after response sent
#
#  /docs → auto Swagger UI, always open this!

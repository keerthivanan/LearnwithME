"""
Flask — Definitions + Full ML API
===================================
pip install flask scikit-learn numpy joblib PyJWT
Run: python flask_everything.py
Test at: http://localhost:5000

WHAT IS FLASK?
  → Lightweight Python web framework — build REST APIs and web apps
  → Minimal core — you add what you need (no built-in ORM, auth, etc.)
  → Great for: ML model serving, microservices, quick APIs
  → Compared to FastAPI: Flask is older, more battle-tested, sync by default
                          FastAPI is newer, async-first, auto docs with Swagger

WHAT IS A REST API?
  → Representational State Transfer — web API standard
  → Uses HTTP methods to define actions:
    GET    → retrieve data
    POST   → create new resource / send data
    PUT    → update existing resource
    DELETE → delete resource
  → Returns JSON responses with HTTP status codes
    200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, 500 Server Error
"""

from flask import Flask, request, jsonify
from functools import wraps
import numpy as np
import time
import uuid
import jwt
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"

# Train a model at startup — in production, load a saved model (joblib.load)
X, y = make_classification(n_samples=500, n_features=5, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_scaled, y)
print("Model trained and ready!")

users = {"admin": "password123", "user": "pass456"}   # demo user store


# ══════════════════════════════════════════════════════
# 1. DECORATORS — Middleware / Request Guards
# ══════════════════════════════════════════════════════
# WHAT IS A FLASK DECORATOR?
#   → Function that wraps another function to add behavior before/after
#   → Applied with @decorator syntax above route functions
#   → Use for: authentication, timing, logging, rate limiting
#
# WHAT IS JWT (JSON Web Token)?
#   → Compact token format: header.payload.signature (base64 encoded)
#   → Stateless auth — server doesn't store sessions
#   → After login: server creates JWT signed with SECRET_KEY
#   → Client sends JWT in: Authorization: Bearer <token> header
#   → Server verifies signature → trusts the claims (user ID, expiry)
#
# @wraps(f):
#   → Preserves the original function's name/docstring after wrapping
#   → Without it, Flask can't distinguish decorated routes (throws errors)

def require_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

def timer(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start    = time.time()
        response = f(*args, **kwargs)
        elapsed  = (time.time() - start) * 1000
        if hasattr(response, "headers"):
            response.headers["X-Response-Time"] = f"{elapsed:.2f}ms"
        return response
    return decorated


# ══════════════════════════════════════════════════════
# 2. BEFORE / AFTER REQUEST HOOKS
# ══════════════════════════════════════════════════════
# WHAT ARE REQUEST HOOKS?
#   → Functions that run automatically before/after EVERY request
#   → @app.before_request : runs before the route handler
#     - Use for: logging, auth checks, rate limiting
#   → @app.after_request  : runs after the route handler, before response sent
#     - Use for: adding headers (CORS), response logging

@app.before_request
def log_request():
    print(f"[{datetime.datetime.now()}] {request.method} {request.path}")

@app.after_request
def add_cors(response):
    # CORS = Cross-Origin Resource Sharing
    # → Allows browser to call API from a different domain (e.g., localhost:3000 → localhost:5000)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response


# ══════════════════════════════════════════════════════
# 3. ROUTES — API Endpoints
# ══════════════════════════════════════════════════════
# WHAT IS A ROUTE?
#   → Maps a URL path to a Python function
#   → @app.route("/path", methods=["GET", "POST"])
#   → request.get_json()    : parse JSON body from POST requests
#   → request.args.get("key"): get URL query params (?key=value)
#   → jsonify(dict)         : convert dict → JSON response with correct headers

@app.route("/")
def home():
    return jsonify({
        "app":     "ML Platform API",
        "version": "1.0",
        "status":  "running",
        "time":    datetime.datetime.now().isoformat(),
    })
# GET / → {"app": "ML Platform API", "version": "1.0", ...}

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "loaded"})
# GET /health → {"status": "healthy", "model": "loaded"}


# ── AUTH ROUTES ─────────────────────────────────────
# WHAT IS LOGIN/TOKEN AUTH?
#   → Client sends username + password → server verifies → returns JWT token
#   → Client stores JWT, sends it with every future request
#   → JWT contains: who you are (sub), when it expires (exp)

@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if users.get(username) != password:
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode({
        "sub": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, app.config["SECRET_KEY"], algorithm="HS256")

    return jsonify({"access_token": token, "token_type": "bearer"})
# POST /login  Body: {"username":"admin","password":"password123"}
# → {"access_token": "eyJhb...", "token_type": "bearer"}


# ── ML PREDICTION ROUTES ─────────────────────────────
# WHAT IS MODEL SERVING?
#   → Exposing a trained ML model as an API
#   → Client sends features → API returns prediction + confidence
#   → This is how ML models go from Jupyter notebook → production!

@app.route("/predict", methods=["POST"])
@require_token   # must have valid JWT
@timer           # adds X-Response-Time header
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = data["features"]
    if not isinstance(features, list):
        return jsonify({"error": "features must be a list"}), 400

    try:
        X_input  = np.array(features).reshape(1, -1)
        X_sc     = scaler.transform(X_input)
        pred     = int(model.predict(X_sc)[0])
        proba    = model.predict_proba(X_sc)[0].tolist()
        return jsonify({
            "prediction":  pred,
            "probability": {str(i): round(p, 4) for i, p in enumerate(proba)},
            "confidence":  round(max(proba), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# POST /predict + Authorization: Bearer <token>
# Body: {"features": [1.2, -0.5, 0.8, 1.1, -0.3]}
# → {"prediction": 1, "probability": {"0": 0.12, "1": 0.88}, "confidence": 0.88}

@app.route("/predict/batch", methods=["POST"])
@require_token
def predict_batch():
    data  = request.get_json()
    items = data.get("items", [])
    if not items:
        return jsonify({"error": "No items provided"}), 400

    results = []
    for item in items:
        X_input = np.array(item["features"]).reshape(1, -1)
        X_sc    = scaler.transform(X_input)
        pred    = int(model.predict(X_sc)[0])
        proba   = model.predict_proba(X_sc)[0].tolist()
        results.append({
            "id":         item.get("id", str(uuid.uuid4())),
            "prediction": pred,
            "confidence": round(max(proba), 4),
        })
    return jsonify({"results": results, "count": len(results)})


# ── CRUD ROUTES ──────────────────────────────────────
# WHAT IS CRUD?
#   → Create, Read, Update, Delete — the four basic database operations
#   → GET    /models         → list all models (Read)
#   → POST   /models         → create new model record (Create)
#   → GET    /models/<id>    → get one model (Read)
#   → PUT    /models/<id>    → update one model (Update)
#   → DELETE /models/<id>    → delete one model (Delete)
#   → Pagination: ?page=1&per_page=10 → return subset of results

models_store = {}

@app.route("/models", methods=["GET"])
@require_token
def list_models():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))
    all_models = list(models_store.values())
    start      = (page - 1) * per_page
    return jsonify({
        "total":   len(all_models),
        "page":    page,
        "results": all_models[start: start + per_page]
    })

@app.route("/models", methods=["POST"])
@require_token
def create_model_record():
    data     = request.get_json()
    model_id = str(uuid.uuid4())
    models_store[model_id] = {
        "id":         model_id,
        "name":       data.get("name"),
        "type":       data.get("type"),
        "created_at": datetime.datetime.now().isoformat(),
    }
    return jsonify(models_store[model_id]), 201   # 201 = Created

@app.route("/models/<model_id>", methods=["GET"])
@require_token
def get_model_record(model_id):
    if model_id not in models_store:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(models_store[model_id])

@app.route("/models/<model_id>", methods=["PUT"])
@require_token
def update_model_record(model_id):
    if model_id not in models_store:
        return jsonify({"error": "Model not found"}), 404
    models_store[model_id].update(request.get_json())
    return jsonify(models_store[model_id])

@app.route("/models/<model_id>", methods=["DELETE"])
@require_token
def delete_model_record(model_id):
    if model_id not in models_store:
        return jsonify({"error": "Model not found"}), 404
    del models_store[model_id]
    return "", 204   # 204 = No Content (success, nothing to return)


# ══════════════════════════════════════════════════════
# 4. ERROR HANDLERS
# ══════════════════════════════════════════════════════
# WHAT ARE ERROR HANDLERS?
#   → Custom responses for HTTP errors (404, 500, etc.)
#   → Without them, Flask returns HTML error pages (ugly for an API)
#   → @app.errorhandler(code) → runs when that HTTP error occurs

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found", "path": request.path}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": f"Method {request.method} not allowed"}), 405


# ══════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# TEST COMMANDS:
# python flask_everything.py
#
# curl http://localhost:5000/
#
# Login → get token:
# curl -X POST http://localhost:5000/login \
#   -H "Content-Type: application/json" \
#   -d '{"username":"admin","password":"password123"}'
#
# Predict (replace TOKEN with the access_token from above):
# curl -X POST http://localhost:5000/predict \
#   -H "Authorization: Bearer TOKEN" \
#   -H "Content-Type: application/json" \
#   -d '{"features": [1.2, -0.5, 0.8, 1.1, -0.3]}'

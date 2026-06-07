"""
Flask — Full ML API
====================
pip install flask scikit-learn numpy joblib
Run: python flask_everything.py
"""

from flask import Flask, request, jsonify, render_template_string
from functools import wraps
import numpy as np
import joblib
import time
import uuid
import jwt
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"

# ── Train a model at startup ───────────────
X, y = make_classification(n_samples=500, n_features=5, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_scaled, y)
print("Model trained and ready!")

# Fake user store
users = {"admin": "password123", "user": "pass456"}

# ════════════════════════════════════════════
# 1. DECORATORS — Middleware
# ════════════════════════════════════════════

def require_token(f):
    """JWT auth decorator"""
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
    """Add X-Response-Time header"""
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        response = f(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        if hasattr(response, "headers"):
            response.headers["X-Response-Time"] = f"{elapsed:.2f}ms"
        return response
    return decorated

# ════════════════════════════════════════════
# 2. BEFORE / AFTER REQUEST
# ════════════════════════════════════════════

@app.before_request
def log_request():
    print(f"[{datetime.datetime.now()}] {request.method} {request.path}")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

# ════════════════════════════════════════════
# 3. ROUTES
# ════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "app":     "ML Platform API",
        "version": "1.0",
        "status":  "running",
        "time":    datetime.datetime.now().isoformat(),
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "loaded"})

# ── Auth ──────────────────────────────────
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
    # → {"access_token": "eyJ...", "token_type": "bearer"}

# ── Predict ───────────────────────────────
@app.route("/predict", methods=["POST"])
@require_token
@timer
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = data["features"]

    if not isinstance(features, list):
        return jsonify({"error": "features must be a list"}), 400

    try:
        X_input  = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        pred     = int(model.predict(X_scaled)[0])
        proba    = model.predict_proba(X_scaled)[0].tolist()
        return jsonify({
            "prediction":  pred,
            "probability": {str(i): round(p, 4) for i, p in enumerate(proba)},
            "confidence":  round(max(proba), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # POST /predict + Bearer token
    # Body: {"features": [1.2, -0.5, 0.8, 1.1, -0.3]}
    # → {"prediction": 1, "probability": {"0": 0.12, "1": 0.88}, "confidence": 0.88}

# ── Batch predict ─────────────────────────
@app.route("/predict/batch", methods=["POST"])
@require_token
def predict_batch():
    data  = request.get_json()
    items = data.get("items", [])

    if not items:
        return jsonify({"error": "No items provided"}), 400

    results = []
    for item in items:
        X_input  = np.array(item["features"]).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        pred     = int(model.predict(X_scaled)[0])
        proba    = model.predict_proba(X_scaled)[0].tolist()
        results.append({
            "id":         item.get("id", str(uuid.uuid4())),
            "prediction": pred,
            "confidence": round(max(proba), 4),
        })

    return jsonify({"results": results, "count": len(results)})

# ── CRUD ──────────────────────────────────
models_store = {}

@app.route("/models", methods=["GET"])
@require_token
def list_models():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))
    models   = list(models_store.values())
    start    = (page - 1) * per_page
    return jsonify({
        "total":   len(models),
        "page":    page,
        "results": models[start: start + per_page]
    })

@app.route("/models", methods=["POST"])
@require_token
def create_model():
    data  = request.get_json()
    model_id = str(uuid.uuid4())
    models_store[model_id] = {
        "id":         model_id,
        "name":       data.get("name"),
        "type":       data.get("type"),
        "created_at": datetime.datetime.now().isoformat(),
    }
    return jsonify(models_store[model_id]), 201

@app.route("/models/<model_id>", methods=["GET"])
@require_token
def get_model(model_id):
    if model_id not in models_store:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(models_store[model_id])

@app.route("/models/<model_id>", methods=["PUT"])
@require_token
def update_model(model_id):
    if model_id not in models_store:
        return jsonify({"error": "Model not found"}), 404
    data = request.get_json()
    models_store[model_id].update(data)
    return jsonify(models_store[model_id])

@app.route("/models/<model_id>", methods=["DELETE"])
@require_token
def delete_model(model_id):
    if model_id not in models_store:
        return jsonify({"error": "Model not found"}), 404
    del models_store[model_id]
    return "", 204

# ── Error handlers ────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found", "path": request.path}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": f"Method {request.method} not allowed"}), 405

# ════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# ────────────────────────────────────────────
# TEST IT:
# python flask_everything.py
#
# curl http://localhost:5000/
#
# Login:
# curl -X POST http://localhost:5000/login \
#   -H "Content-Type: application/json" \
#   -d '{"username":"admin","password":"password123"}'
#
# Predict:
# curl -X POST http://localhost:5000/predict \
#   -H "Authorization: Bearer YOUR_TOKEN" \
#   -H "Content-Type: application/json" \
#   -d '{"features": [1.2, -0.5, 0.8, 1.1, -0.3]}'
# ────────────────────────────────────────────

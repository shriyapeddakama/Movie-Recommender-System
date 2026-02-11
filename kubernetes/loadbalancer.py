from flask import Flask, request, jsonify
import requests
import hashlib

app = Flask(__name__)

# Backend model service URLs
MODELS = {
    "als-model": "http://als-model:5000/predict",
    "hybrid-model": "http://hybrid-model:5000/predict"
}

def choose_model(user_id):
    """Deterministically assign user to als-model or hybrid-model based on hash."""
    user_hash = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    if user_hash % 2 == 0:
        return "als-model"
    else:
        return "hybrid-model"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_id = data.get("user_id", "default_user")
    model_key = choose_model(user_id)
    model_url = MODELS[model_key]

    try:
        resp = requests.post(model_url, json=data, timeout=5)
        prediction = resp.json()
    except Exception as e:
        return jsonify({"error": "Model backend error", "details": str(e)}), 500

    return jsonify({
        "model": model_key,
        "prediction": prediction
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")
intent_response = joblib.load("intent_response.joblib")

@app.route("/", methods=["GET"])
def home():
    return "Widad Bot is running locally."

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.json.get("message", "")
    if not incoming_msg:
        return jsonify({"response": "لم أستطع فهم رسالتك، هل يمكنك توضيحها؟"})

    msg_vec = vectorizer.transform([incoming_msg])
    predicted_intent = model.predict(msg_vec)[0]
    response = intent_response.get(predicted_intent, "لم أجد ردًا مناسبًا، هل يمكنك إعادة صياغة سؤالك؟")

    return jsonify({
        "intent": predicted_intent,
        "response": response
    })
if __name__ == "__main__":
    app.run(debug=True)

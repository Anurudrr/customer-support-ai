from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess
from src.escalation_decider import should_escalate
from src.reply_generator import ReplyGenerator

# Load lightweight model
MODEL_PATH = ROOT / "models" / "vercel_classifier.pkl"
clf = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

reply_gen = ReplyGenerator()

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "healthy", "service": "AI Customer Support NLP API"}).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req = json.loads(post_data.decode('utf-8'))
            message = req.get('message', '').strip()
            
            if not message:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Message is required"}).encode('utf-8'))
                return

            # Preprocess
            clean_text = preprocess(message)
            
            if clf is not None:
                probs = clf.predict_proba([clean_text])[0]
                classes = clf.classes_
                best_idx = np.argmax(probs)
                intent = classes[best_idx]
                confidence = float(probs[best_idx])
            else:
                intent = "unknown"
                confidence = 0.0

            # Escalation decision
            esc_result = should_escalate(message, intent, confidence)

            # Reply generation
            if esc_result.should_escalate:
                reply = reply_gen.generate_escalation(
                    reason=esc_result.reason,
                    message=message,
                    triggered_rule=esc_result.triggered_rule,
                )
            else:
                reply = reply_gen.generate(intent=intent, message=message)

            res = {
                "message": message,
                "intent": intent,
                "confidence": round(confidence, 4),
                "should_escalate": esc_result.should_escalate,
                "escalation_reason": esc_result.reason if esc_result.should_escalate else "",
                "triggered_rule": esc_result.triggered_rule,
                "anger_signals": esc_result.anger_signals,
                "reply": reply
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

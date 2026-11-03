"""Stands in for llama-server in tests: same command line, same HTTP protocol, no model.

Speaks the parts of llama-server's API the llamacpp backend uses -- /health
(503 while "loading", then 200), /props (with `modalities`), and
/v1/chat/completions (bearer-key auth, an `image_url` part, `grammar`,
`logprobs`) -- and records every launch and request to the JSON-lines file
named by FAKE_LLAMA_RECORD, so a test can assert on exactly what was sent.

Behaviour is chosen by FAKE_LLAMA_MODE, a comma-separated list of:

    load_delay=S   /health answers 503 for S seconds after start
    crash          print an "unknown model architecture" error and exit 1
    no_vision      /props reports modalities.vision = false
    no_logprobs    completions carry no logprobs
    error500       completions answer HTTP 500
    slow=S         completions take S seconds
    die_after=N    exit after answering N completions
    think          wrap every answer in a <think> block
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

STARTED = time.monotonic()
MODE: dict[str, str] = {}
for item in filter(None, os.environ.get("FAKE_LLAMA_MODE", "").split(",")):
    key, _, value = item.partition("=")
    MODE[key.strip()] = value.strip()

RECORD = os.environ.get("FAKE_LLAMA_RECORD")
API_KEY = os.environ.get("LLAMA_API_KEY", "")
_record_lock = threading.Lock()
_completions = 0

# Per-token probabilities every answer is scored with: geometric mean 0.8485.
TOKEN_PROBS = (0.9, 0.8)


def record(entry: dict) -> None:
    if not RECORD:
        return
    with _record_lock, open(RECORD, "a", encoding="utf-8") as sink:
        sink.write(json.dumps(entry, ensure_ascii=False) + "\n")


def answer_for(body: dict) -> str:
    """A plausible reading for whichever kind of cell the request is for."""
    text = ""
    for message in body.get("messages", []):
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                text = part["text"]
    grammar = body.get("grammar") or ""
    if grammar:
        return "7/8/2026" if '"/"' in grammar else "١٢"
    # Without a grammar the kind shows in the prompt (the Qwen prompts name it)
    # or, when every kind shares one prompt (Nuqta's), in the token budget the
    # shared decoding policy gives it: code 24, arabic_text 48, any 64.
    budget = body.get("max_tokens")
    if "رقم الفاتورة" in text or budget == 24:
        return "رقم الفاتورة : 00010"
    if "وصف منتج" in text or budget == 48:
        return "بوري شبابي زارا"
    return "نص"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {API_KEY}"

    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() - STARTED < float(MODE.get("load_delay", 0) or 0):
                return self._json(503, {"error": {"code": 503, "message": "Loading model"}})
            return self._json(200, {"status": "ok"})
        if not self._authorized():
            return self._json(401, {"error": {"code": 401, "message": "Invalid API Key"}})
        if self.path == "/props":
            return self._json(
                200,
                {"build_info": "b99999-fake", "modalities": {"vision": "no_vision" not in MODE, "audio": False}},
            )
        return self._json(404, {"error": {"code": 404, "message": "Not Found"}})

    def do_POST(self):
        global _completions
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if not self._authorized():
            return self._json(401, {"error": {"code": 401, "message": "Invalid API Key"}})
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"code": 404, "message": "Not Found"}})

        image_size = None
        for message in body.get("messages", []):
            for part in message.get("content", []):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    header, _, data = url.partition(",")
                    if header != "data:image/png;base64":
                        return self._json(400, {"error": {"message": f"unsupported image url {header}"}})
                    image_size = Image.open(io.BytesIO(base64.b64decode(data))).size
        if image_size is None:
            return self._json(400, {"error": {"code": 400, "message": "no image in request"}})

        record({"event": "completion", "body": {k: v for k, v in body.items() if k != "messages"},
                "prompt": [p.get("text") for m in body["messages"] for p in m["content"] if p.get("type") == "text"],
                "image_size": image_size})

        if "slow" in MODE:
            time.sleep(float(MODE["slow"]))
        if "error500" in MODE:
            return self._json(500, {"error": {"code": 500, "message": "boom", "type": "server_error"}})

        content = answer_for(body)
        if "think" in MODE:
            content = f"<think>looking at the crop</think>{content}"

        choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
        if body.get("logprobs") and "no_logprobs" not in MODE:
            # A real server reports 1.0 for every token when top_logprobs is 0;
            # mimic that so a client that forgets it is caught.
            probs = TOKEN_PROBS if body.get("top_logprobs", 20) >= 1 else (1.0, 1.0)
            choice["logprobs"] = {
                "content": [
                    {"token": f"t{i}", "logprob": math.log(p), "bytes": [], "top_logprobs": []}
                    for i, p in enumerate(probs)
                ]
            }
        self._json(200, {"object": "chat.completion", "choices": [choice]})

        _completions += 1
        if "die_after" in MODE and _completions >= int(MODE["die_after"]):
            threading.Thread(target=lambda: (time.sleep(0.1), os._exit(3)), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args, rest = parser.parse_known_args()

    record({"event": "launch", "argv": sys.argv[1:], "api_key_in_env": bool(API_KEY),
            "api_key_on_command_line": API_KEY in " ".join(sys.argv)})

    if "crash" in MODE:
        print("llama_model_load: error loading model: unknown model architecture: 'qwen35'", flush=True)
        sys.exit(1)

    print(f"main: server is listening on http://{args.host}:{args.port}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()

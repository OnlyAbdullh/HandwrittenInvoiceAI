"""Stands in for tools/qwen_worker.py in tests: same protocol, no model.

Each result's text is a digest of the image it was sent, so a caller can tell
whether it received the answer to its *own* request. A random delay before
every answer makes interleaving likely if the parent fails to serialize the
pipe. Kind "any" sleeps far past any test timeout, to exercise the timeout path.
"""

import hashlib
import json
import random
import sys
import time

print("READY", flush=True)

for line in sys.stdin:
    request = json.loads(line)
    results = []
    for item in request["items"]:
        if item.get("kind") == "any":
            time.sleep(30)
        time.sleep(random.uniform(0.0, 0.02))
        digest = hashlib.sha256(item["image"].encode()).hexdigest()[:16]
        results.append({"text": digest, "confidence": 0.9})
    print(json.dumps({"id": request["id"], "results": results}), flush=True)

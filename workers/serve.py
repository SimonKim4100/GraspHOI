"""Request loop of the model workers (standard library only)."""
import json
import os
import sys
import traceback


def serve(handle, reply_fd):
    """Announce "loaded", then answer requests with handle until quit."""
    with os.fdopen(int(reply_fd), "w", buffering=1) as reply:
        reply.write(json.dumps({"event": "loaded"}) + "\n")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                if request.get("cmd") == "quit":
                    break
                message = {"ok": True}
                message.update(handle(request))
                out = json.dumps(message)
            except Exception:
                out = json.dumps({"ok": False, "error": traceback.format_exc()})
            reply.write(out + "\n")

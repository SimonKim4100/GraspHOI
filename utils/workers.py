"""Clients of the model workers."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from human_recon.constants import (
    HOIDETR_CKPT, HOIDETR_CONFIG, HOIDETR_ROOT, MHR_FITTER, MHR_MAPPING, REPO_ROOT, SAM3D_BODY_CKPT,
    SAM3D_BODY_MHR_MODEL, SAM3D_BODY_ROOT, SAM3D_OBJECTS_ROOT, SMPLH_MODEL_PATH,
)

WORKERS_DIR = REPO_ROOT / "workers"
SAM3D_OBJECTS_SEED = 42
BODY_PARAMS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "betas", "trans")
LOG_TAIL_LINES = 40       # log lines attached to an error


class Worker:
    """Model worker process."""

    def __init__(self, script, args=()):
        self.name = Path(script).stem
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        self._log = tempfile.NamedTemporaryFile("w", prefix=f"grasphoi_{self.name}_", suffix=".log",
                                                delete=False)
        read_fd, write_fd = os.pipe()
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-I", "-u", str(script), str(write_fd), *[str(a) for a in args]],
                stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT, pass_fds=(write_fd,),
                env=env, cwd=str(REPO_ROOT), text=True)
        except BaseException:
            os.close(read_fd)
            self._log.close()
            os.unlink(self._log.name)
            raise
        finally:
            os.close(write_fd)
        self._reply = os.fdopen(read_fd, "r")
        print(f"[workers] starting {self.name} (log: {self._log.name})", flush=True)
        try:
            loaded = self._read()
            if loaded.get("event") != "loaded":
                raise RuntimeError(f"{self.name} worker sent {loaded} instead of its loaded event")
        except BaseException:
            self.close(timeout=0)
            raise
        print(f"[workers] {self.name} ready", flush=True)

    def _tail(self):
        """Last LOG_TAIL_LINES lines of the worker log."""
        self._log.flush()
        with open(self._log.name, errors="replace") as f:
            return "".join(f.readlines()[-LOG_TAIL_LINES:])

    def _read(self):
        line = self._reply.readline()
        if not line:
            try:
                code = self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                code = None
            raise RuntimeError(f"{self.name} worker exited (status {code}); log {self._log.name}:\n"
                               f"{self._tail()}")
        return json.loads(line)

    def call(self, request):
        """Send a request and return the reply; raises RuntimeError on failure."""
        if self._proc is None:
            raise RuntimeError(f"{self.name} worker is closed")
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except OSError:
            raise RuntimeError(f"{self.name} worker is not running; log {self._log.name}:\n"
                               f"{self._tail()}") from None
        reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(f"{self.name} worker failed on {request}:\n{reply.get('error', reply)}")
        return reply

    def close(self, timeout=60):
        """Quit the worker, killing it after timeout seconds."""
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.close()
            except OSError:
                proc.kill()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        self._reply.close()
        self._log.close()
        if code == 0:
            os.unlink(self._log.name)
        else:
            print(f"[workers] {self.name} worker exited with status {code}; log kept at {self._log.name}",
                  flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HoiDetr(Worker):
    """HOI-DETR worker (workers/hoidetr_worker.py)."""

    def __init__(self):
        super().__init__(WORKERS_DIR / "hoidetr_worker.py", (HOIDETR_ROOT, HOIDETR_CONFIG, HOIDETR_CKPT))

    def detect(self, image_path):
        """HOI-DETR record of the image."""
        return self.call({"image": os.path.abspath(image_path)})["record"]


class Sam3dBody(Worker):
    """SAM 3D Body worker (workers/sam3d_body_worker.py)."""

    def __init__(self):
        super().__init__(WORKERS_DIR / "sam3d_body_worker.py",
                         (SAM3D_BODY_ROOT, SAM3D_BODY_CKPT, SAM3D_BODY_MHR_MODEL, SMPLH_MODEL_PATH,
                          MHR_FITTER, MHR_MAPPING))

    def fit(self, image_path, bbox_xyxy, K):
        """SMPL-H fit of the person in bbox_xyxy; returns BODY_PARAMS (axis-angle, metres)."""
        box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
        cam = np.asarray(K, dtype=np.float32).reshape(3, 3)
        reply = self.call({"image": os.path.abspath(image_path),
                           "bbox_xyxy": [float(v) for v in box],
                           "cam_int": [[float(v) for v in row] for row in cam]})
        return {k: np.asarray(reply[k], dtype=np.float32) for k in BODY_PARAMS}


class Sam3dObjects(Worker):
    """SAM 3D Objects worker (workers/sam3d_objects_worker.py)."""

    def __init__(self):
        super().__init__(WORKERS_DIR / "sam3d_objects_worker.py", (SAM3D_OBJECTS_ROOT,))

    def reconstruct(self, image_rgb, mask_bool):
        """Mesh and layout of the masked object, as a dict."""
        with tempfile.TemporaryDirectory(prefix="grasphoi_sam3d_objects_") as tmp:
            inp, out = os.path.join(tmp, "input.npz"), os.path.join(tmp, "output.npz")
            np.savez(inp, image=image_rgb, mask=mask_bool, seed=SAM3D_OBJECTS_SEED)
            self.call({"input": inp, "output": out})
            with np.load(out, allow_pickle=False) as z:
                result = {k: z[k] for k in z.files}
        result["s"] = float(result["s"])
        return result

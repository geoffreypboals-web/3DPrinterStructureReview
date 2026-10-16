"""Smoke test: build the demo frame, analyse it without AI, check the numbers are sane.

    python test_smoke.py        (or: python -m pytest test_smoke.py)
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def test_demo_frame():
    with tempfile.TemporaryDirectory() as tmp:
        step = os.path.join(tmp, "demo_frame.step")
        subprocess.run([sys.executable, os.path.join(HERE, "examples", "make_demo_frame.py"), step], check=True)
        subprocess.run([sys.executable, os.path.join(HERE, "printer_stiffness.py"), step,
                        "--no-ai", "--edition", "none"], check=True)
        r = json.load(open(os.path.join(tmp, "demo_frame_stiffness", "results.json")))
        # 2020 cube, 400 x 400 x 450: about 0.3-0.4 N/um and a first mode near 50 Hz
        assert 0.1 < r["summary"]["accel_X"]["stiffness_N_per_um"] < 2, r["summary"]["accel_X"]
        frame_modes = [m["freq_hz"] for m in r["modes"] if not m["local"]]
        assert 20 < frame_modes[0] < 150, frame_modes
        assert r["whatifs"] and r["whatifs"][0]["avg_gain_pct"] > 0


if __name__ == "__main__":
    test_demo_frame()
    print("ok")

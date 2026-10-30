"""Cross-platform task runner (same targets as the Makefile): python run.py [data|eval|pc|insights|pcengine|predict|web|serve|test|app|all]"""
import subprocess
import sys

PY = sys.executable
TASKS = {
    "data": [[PY, "-m", "orbitguard.data"]],
    "pc": [[PY, "-m", "orbitguard.pc"]],
    "insights": [[PY, "-m", "orbitguard.insights"]],
    "eval": [[PY, "-m", "orbitguard.evaluate"], [PY, "-m", "orbitguard.pc"], [PY, "-m", "orbitguard.insights"]],
    "predict": [[PY, "-m", "orbitguard.predict_test"]],
    "test": [[PY, "-m", "pytest", "-q"]],
    "pcengine": [[PY, "-m", "orbitguard.pcnet"]],
    "web": [[PY, "-m", "orbitguard.export_web"]],
    "serve": [[PY, "-m", "http.server", "8000", "--directory", "web"]],
    "app": [[PY, "-m", "streamlit", "run", "app.py"]],
}
TASKS["all"] = TASKS["data"] + TASKS["eval"] + TASKS["pcengine"] + TASKS["predict"] + TASKS["web"] + TASKS["test"]

if __name__ == "__main__":
    for target in sys.argv[1:] or ["all"]:
        for cmd in TASKS[target]:
            print("$", " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=True)

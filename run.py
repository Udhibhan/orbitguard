"""Cross-platform task runner (same targets as the Makefile): python run.py [data|eval|pc|insights|predict|test|app|all]"""
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
    "app": [[PY, "-m", "streamlit", "run", "app.py"]],
}
TASKS["all"] = TASKS["data"] + TASKS["eval"] + TASKS["predict"] + TASKS["test"]

if __name__ == "__main__":
    for target in sys.argv[1:] or ["all"]:
        for cmd in TASKS[target]:
            print("$", " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=True)

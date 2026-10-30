from streamlit.testing.v1 import AppTest
pages = ["Triage queue", "Event detail", "24-hour warnings", "Model evaluation", "About / limits"]
for ds in [0, 1]:
    for p in pages:
        at = AppTest.from_file("../app.py", default_timeout=180)
        at.run()
        at.sidebar.radio(key="ds").set_value(at.sidebar.radio(key="ds").options[ds])
        at.sidebar.radio(key="page").set_value(p)
        at.run()
        errs = [e.value for e in at.exception]
        print(f"ds={ds} {p:18s} exceptions={len(errs)}", errs[:1] if errs else "")

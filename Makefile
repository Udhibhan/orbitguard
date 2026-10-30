PY ?= python

.PHONY: data eval predict test app all

data:
	$(PY) run.py data
eval:
	$(PY) run.py eval
predict:
	$(PY) run.py predict
test:
	$(PY) run.py test
app:
	$(PY) run.py app
all:
	$(PY) run.py all

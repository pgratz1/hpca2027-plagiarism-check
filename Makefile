# Local pairwise PDF similarity checker.  `make help` lists targets.
#
# Everything runs offline except `make env` (pip) and `make models` (one-time
# Hugging Face download).  compare_pdfs.py forces HF offline mode itself.

PY ?= $(HOME)/envs/hpca-plagiarism/bin/python
export PYTHONPATH := src

A ?=
B ?=
OUT ?=
ARGS ?=
MODEL ?= sentence-transformers/all-MiniLM-L6-v2

.PHONY: help env models compare test clean-cache clean-outputs

help:
	@echo "make env                      create ~/envs/hpca-plagiarism and install requirements.txt"
	@echo "make models [MODEL=...]       one-time download of the sentence-embedding model (network)"
	@echo "make compare A=x.pdf B=y.pdf [OUT=dir] [ARGS='--no-semantic']   compare two PDFs (offline)"
	@echo "make test                     run the test suite (offline; network is blocked in tests)"
	@echo "make clean-cache              delete per-PDF extraction caches"
	@echo "make clean-outputs            delete generated reports"

env:
	test -x $(PY) || /usr/bin/python3.13 -m venv $(HOME)/envs/hpca-plagiarism
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

models:
	$(PY) scripts/prefetch_models.py --model "$(MODEL)"

compare:
	@test -n "$(A)" -a -n "$(B)" || { echo "usage: make compare A=x.pdf B=y.pdf [OUT=dir] [ARGS='...']"; exit 2; }
	$(PY) scripts/compare_pdfs.py "$(A)" "$(B)" $(if $(OUT),-o "$(OUT)") $(ARGS)

test:
	$(PY) -m pytest -q tests

clean-cache:
	rm -rf cache/*

clean-outputs:
	rm -rf outputs/*

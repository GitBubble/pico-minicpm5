PYTHON ?= python3

.PHONY: test doctor source-check source-release sbom

test:
	PYTHONPATH=src $(PYTHON) -m pytest

doctor:
	PYTHONPATH=src $(PYTHON) -m pico_minicpm5.cli doctor

source-check:
	PYTHONPATH=src $(PYTHON) -m pico_minicpm5.cli release source --check-only

source-release:
	PYTHONPATH=src $(PYTHON) -m pico_minicpm5.cli release source --out artifacts

sbom:
	PYTHONPATH=src $(PYTHON) -m pico_minicpm5.cli release sbom --out artifacts/pico-minicpm5-$$(PYTHONPATH=src $(PYTHON) -c 'from pico_minicpm5 import __version__; print(__version__)').spdx.json

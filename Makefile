PYTHON ?= python

.PHONY: install editable cpu smoke

install:
	$(PYTHON) scripts/install_smart.py

editable:
	$(PYTHON) scripts/install_smart.py --editable

cpu:
	$(PYTHON) scripts/install_smart.py --editable --cpu-only --extras cpu,dev

smoke:
	$(PYTHON) -X utf8 tests/test_xai.py
	$(PYTHON) -X utf8 tests/test_monitor.py
	$(PYTHON) -X utf8 tests/test_deterministic.py

PYTHON ?= python3
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: all build core clean test install dev

all: build

build:
	$(PYTHON) build_native.py -j $(JOBS)

core:
	$(PYTHON) build_native.py -j $(JOBS) --no-llama

clean:
	rm -rf build gguf_train/_libs dist *.egg-info

test: build
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

install: build
	$(PYTHON) -m pip install --no-build-isolation --no-deps .

dev: build
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

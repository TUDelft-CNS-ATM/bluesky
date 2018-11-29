.PHONY: test install uninstall

PYTHON=python3

all: build dist

test:
	@echo "Running Tests using $(PYTHON) BlueSky"
	@TESTING=true PYEXEC=$(PYTHON) $(PYTHON) -m pytest -s bluesky/test

lint:
	@autopep8 --in-place -r bluesky/test
# the use of fixtures in pytest causes unused-argument and redefined-outer-name 'issues'
	@PYTHONPATH=`pwd`:${PYTHONPATH} $(PYTHON) -m pylint -d "invalid-name,bare-except,unused-argument,redefined-outer-name,too-many-arguments" bluesky/test || true

clean:
	rm -rf dist/ build/ *egg-info*
	find -type d -name '__pycache__' | xargs rm -rf

build: setup.py requirements.txt MANIFEST.in
	@"`which $(PYTHON)`" setup.py build

dist:
	@"`which $(PYTHON)`" setup.py sdist bdist_wheel

pypi:
	twine upload dist/*

install:
	@cd dist && "`which $(PYTHON)`" -m pip install bluesky*.whl

uninstall:
	@"`which $(PYTHON)`" -m pip uninstall bluesky-simulator
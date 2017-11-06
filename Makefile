.PHONY: test
test:
	@echo "Running Tests using Python2 BlueSky"
	PYEXEC=python2 python3 -m pytest -s bluesky/test
	@echo "Running Tests using Python3 BlueSky"
	PYEXEC=python3 python3 -m pytest -s bluesky/test

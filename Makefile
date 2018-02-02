.PHONY: test
test:
	@echo "Running Tests using Python2 BlueSky"
	TESTING=true PYEXEC=python2 python3 -m pytest -s bluesky/test
	@echo "Running Tests using Python3 BlueSky"
	TESTING=true PYEXEC=python3 python3 -m pytest -s bluesky/test

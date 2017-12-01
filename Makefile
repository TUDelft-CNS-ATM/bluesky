.PHONY: test
test:
	@echo "Running Tests using Python2 BlueSky"
	@TESTING=true PYEXEC=python2 python3 -m pytest -s bluesky/test
	@echo "Running Tests using Python3 BlueSky"
	@TESTING=true PYEXEC=python3 python3 -m pytest -s bluesky/test

lint:
	@autopep8 --in-place -r bluesky/test
# the use of fixtures in pytest causes unused-argument and redefined-outer-name 'issues'
	@PYTHONPATH=`pwd`:${PYTHONPATH} pylint3 -d "invalid-name,bare-except,unused-argument,redefined-outer-name,too-many-arguments" bluesky/test || true

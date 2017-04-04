init:
	echo "init: nothing to do"
	#pip install -r requirements.txt

test: init
	pytest --cov=rptree --cov-report=term-missing tests

.PHONY: init test

test :
	black .
	python -m flake8 altair_transform
	python -m mypy altair_transform
	rm -r build
	python setup.py build &&\
	  cd build/lib &&\
	  python -m pytest --pyargs altair_transform

test-coverage:
	python setup.py build &&\
	  cd build/lib &&\
	  python -m pytest --pyargs --doctest-modules --cov=altair_transform --cov-report term altair_transform

test-coverage-html:
	python setup.py build &&\
	  cd build/lib &&\
	  python -m pytest --pyargs --doctest-modules --cov=altair_transform --cov-report html altair_transform

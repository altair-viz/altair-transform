test :
	black --check .
	python -m flake8 altair_transform
	python -m mypy altair_transform
	rm -r build
	python setup.py build &&\
	  cd build/lib &&\
	  python -m pytest --pyargs altair_transform

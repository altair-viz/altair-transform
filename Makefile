test :
	python -m mypy altair_transform
	python -m pytest --pyargs altair_transform

flake :
	python -m flake8 altair_transform

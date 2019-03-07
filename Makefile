test :
	python -m pytest --pyargs altair_transform
	python -m mypy altair_transform
	python -m flake8 altair_transform

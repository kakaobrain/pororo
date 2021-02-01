.PHONY: style quality

check_dirs := pororo/ tests/

# TODO: Apply yapf to Makefile after isort
style:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

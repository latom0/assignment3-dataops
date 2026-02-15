init:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	dvc init

repro:
	dvc repro

status:
	dvc status

dag:
	dvc dag

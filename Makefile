.PHONY: test build clean

replay:
	python3 -m examples.replay

example:
	python3 -m examples.example

make n_replay:
	python3 -m examples.dummy
	python3 -m examples.replay

t:
	pytest tests/test_game.py
	pytest tests/test_utils.py
	python3 -m tests.parallel_api_test
	python3 -m tests.gym_longruntest

build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf build dist

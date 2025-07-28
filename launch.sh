pip install uv

uv venv
source .venv/bin/activate

uv pip install -e .

uv run tests/test_env.py
source .venv/bin/activate

echo "PROJECT_ROOT=\"$(pwd)\"" > .env

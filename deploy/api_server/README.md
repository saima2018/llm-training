## API server instructions

1.
```bash
cp .env.example .env
```

2. Modify the environment variables in the .env file
```bash
# server configuration
PORT=8000
HOST="0.0.0.0"
RELOAD=True

# model configuration
MODEL_NAME_OR_PATH="gpt2"
```

3. Start the API server
```bash
python rm_full.py
```
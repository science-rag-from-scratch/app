#!/bin/bash

# Get the project directory (parent of the script's directory)
PROJECT_PATH="$(pwd)"

# Set DATA_PATH to the absolute path of the data directory
DATA_PATH="$PROJECT_PATH/data"

# Set PostgreSQL environment variables
POSTGRES_HOST="localhost"
POSTGRES_DB="postgres"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="postgres"

EMB_MODEL_PATH="intfloat/multilingual-e5-large-instruct"
LLM_MODEL_PATH="/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"

# Search parameters
TOP_K=9
MAX_LENGTH=512

# Create .env file
cat > "$PROJECT_PATH/.env" << EOF
PROJECT_PATH=$PROJECT_PATH
DATA_PATH=$DATA_PATH
POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_DB=$POSTGRES_DB
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
EMB_MODEL_PATH=$EMB_MODEL_PATH
LLM_MODEL_PATH=$LLM_MODEL_PATH
TOP_K=$TOP_K
MAX_LENGTH=$MAX_LENGTH
DB_CONN="dbname=$POSTGRES_DB user=$POSTGRES_USER password=$POSTGRES_PASSWORD host=$POSTGRES_HOST"
EOF

echo ".env file created at $PROJECT_PATH/.env"
echo "DATA_PATH=$DATA_PATH"
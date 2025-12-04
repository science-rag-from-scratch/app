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

# Create .env file
cat > "$PROJECT_PATH/.env" << EOF
DATA_PATH=$DATA_PATH
POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_DB=$POSTGRES_DB
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
EOF

echo ".env file created at $PROJECT_PATH/.env"
echo "DATA_PATH=$DATA_PATH"
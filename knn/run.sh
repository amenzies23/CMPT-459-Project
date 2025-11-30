#!/bin/bash
# Pass optional flag --no-hyperparameter-tuning
# Usage:
#   ./run.sh                            Train with hyperparameter tuning
#   ./run.sh --no-hyperparameter-tuning Train without hyperparameter tuning

# Virtual Environment
# Auto-activate venv if not active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "../venv" ]; then
        echo -e "\033[1;32m[Info]\033[0m Activating virtual environment..."
        source ../venv/bin/activate
        if [ $? -ne 0 ]; then
            echo -e "\033[1;31m[Error]\033[0m Failed to activate venv."
            exit 1
        fi
    else
        echo -e "\033[1;33m[Warning]\033[0m No virtual environment detected and no ./venv directory found."
        echo "Continuing without virtualenv..."
    fi
fi

# Train Model
if [[ "$1" == "--no-hyperparameter-tuning" ]]; then
    echo -e "\033[1;36m  Running KNN without hyperparameter tuning...\033[0m"
    python main.py --no-hyperparameter-tuning
else
    echo -e "\033[1;36m  Running KNN with hyperparameter tuning...\033[0m"
    python main.py
fi

# Check if training succeeded
if [ $? -ne 0 ]; then
  echo -e "\033[1;31m Training failed. Please check the logs above.\033[0m"
  exit 1
fi

# Launch Flask App
echo -e "\033[1;32m Launching Flask app (localhost:5000)...\033[0m"
python app/app.py
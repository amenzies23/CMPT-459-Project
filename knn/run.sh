#!/bin/bash
# Pass optional flag --no-hyperparameter-tuning
# Usage:
#   ./run.sh                            Train with hyperparameter tuning
#   ./run.sh --no-hyperparameter-tuning Train without hyperparameter tuning

# Environment Setup
# Check if virtual environment is active (optional reminder)
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "\033[1;33m[Warning]\033[0m No virtual environment detected."
    echo "It's recommended to run this inside a virtualenv."
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
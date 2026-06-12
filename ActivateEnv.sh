# -------------------------------------------------------
# ActivateEnv: Activate a Python venv and open VS Code
# Usage: ActivateEnv <ENV_NAME> <PROJECT_DIR>
#
# This file assumes that your python virtual environements 
# are located at $HOME/penv/${ENV_NAME}/bin/activate
# For example: /home/username/penv
# Edit line 37 to change this (comments included in line count)
#
# On Linux, edit .bashrc and copy this entire file at the end
#	sudo nano ~/.bashrc	
# In nano, press CTR + X to exit (select Y on save)
# For mac computers, use .zshrc instead of .bashrc
# -------------------------------------------------------
ActivateEnv() {
    local ENV_NAME="$1"
    local BASE_DIR="$2"

    echo "Starting activation of environment '${ENV_NAME}'..."

    # Validate input
    if [ -z "$ENV_NAME" ]; then
        echo "[ERROR] Environment name not specified."
        return 1
    fi

    if [ -z "$BASE_DIR" ]; then
        echo "[ERROR] Base directory not specified."
        return 1
    fi

    if [ ! -d "$BASE_DIR" ]; then
        echo "[ERROR] Directory not found: ${BASE_DIR}"
        return 1
    fi

    local VENV_ACTIVATE="$HOME/penv/${ENV_NAME}/bin/activate"

    if [ ! -f "$VENV_ACTIVATE" ]; then
        echo "[ERROR] Virtual environment not found: ${VENV_ACTIVATE}"
        return 1
    fi

    # Change directory, activate, and open VS Code
    cd "$BASE_DIR" || return 1
    source "$VENV_ACTIVATE"
    echo "Environment '${ENV_NAME}' activated."
    # code .
}

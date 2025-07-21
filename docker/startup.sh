#!/bin/bash

set -e

echo "Starting VS Code Dev Container setup..."

# Update repository (fetch latest changes)
update_repository() {
    echo "Updating repository..."
    
    REPO_PATH="/workspace/fyp"
    
    if [ -d "$REPO_PATH" ]; then
        echo "Repository exists, fetching latest changes..."
        cd "$REPO_PATH"
        git fetch origin
        git pull origin main || git pull origin master || echo "Pull completed with conflicts or different branch"
    else
        echo "ERROR: Repository not found! This should have been cloned during build."
        exit 1
    fi
    
    export REPO_PATH
    echo "Repository updated at: $REPO_PATH"
}

# Update Python environment (install any new requirements)
update_python_env() {
    echo "Updating Python environment..."
    
    cd "$REPO_PATH"
    
    # Activate existing virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "Activated existing virtual environment"
        echo "Python version: $(python --version)"
        
        # Install/update requirements if requirements.txt exists
        if [ -f "requirements.txt" ]; then
            echo "Installing/updating requirements from requirements.txt..."
            # Pip will skip already installed packages with same version
            pip install -r requirements.txt
        else
            echo "No requirements.txt found"
        fi
        
        echo "Python environment update complete"
    else
        echo "ERROR: Virtual environment not found! This should have been created during build."
        exit 1
    fi
}

# Setup VS Code tunnel
setup_vscode_tunnel() {
    echo "Setting up VS Code tunnel..."
    
    VSCODE_TUNNEL_NAME="fyp"
    echo "Using tunnel name: $VSCODE_TUNNEL_NAME"
    
    cd "$REPO_PATH"
    
    echo "Starting VS Code tunnel with name: $VSCODE_TUNNEL_NAME"
    echo "You'll need to authenticate with GitHub when prompted"
    
    # Start tunnel in background
    code tunnel --name "$VSCODE_TUNNEL_NAME" --accept-server-license-terms &
    
    sleep 5
    echo "VS Code tunnel is starting up..."
    echo "Access your tunnel at: https://vscode.dev/tunnel/$VSCODE_TUNNEL_NAME"
}

# Main execution
echo "=== Container Runtime Setup ==="

update_repository
update_python_env
setup_vscode_tunnel

echo "=== Setup Complete ==="
echo "Repository: $REPO_PATH"
echo "Python venv: $REPO_PATH/.venv"
echo "VS Code tunnel: https://vscode.dev/tunnel/$VSCODE_TUNNEL_NAME"
echo ""
echo "Container is ready!"

# Keep container running
tail -f /dev/null
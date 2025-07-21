#!/bin/bash

set -e

echo "Starting VS Code Dev Container setup..."

# Setup SSH key for current user (in case it's different from build user)
setup_github_auth() {
    echo "Setting up GitHub authentication for runtime user..."
    
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    
    # Copy SSH key from the image
    if [ -f "/ssh_id_ed25519_fyp-docker" ]; then
        cp /ssh_id_ed25519_fyp-docker ~/.ssh/id_ed25519
        sed -i 's/\r$//' ~/.ssh/id_ed25519
        chmod 600 ~/.ssh/id_ed25519
        
        # Validate the key format
        if ! ssh-keygen -l -f ~/.ssh/id_ed25519 >/dev/null 2>&1; then
            echo "ERROR: SSH key format is invalid!"
            echo "Please check that the SSH key file is properly formatted"
            exit 1
        fi
        
        # Create SSH config
        cat > ~/.ssh/config << EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking no
EOF
        chmod 600 ~/.ssh/config
        echo "SSH key setup complete"
    else
        echo "ERROR: SSH key file not found!"
        exit 1
    fi
    
    # Add GitHub to known hosts
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts
    
    # Start SSH agent and add the key
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    
    # Test SSH connection
    echo "Testing GitHub SSH connection..."
    ssh -T git@github.com || true
}

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
    
    VSCODE_TUNNEL_NAME="fyp-xinghao2003"
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

setup_github_auth
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
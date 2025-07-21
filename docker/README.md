# Docker Development Container

## Introduction

This Docker container provides a complete development environment for the Final Year Project (FYP). It's designed to eliminate "works on my machine" issues by providing a consistent, reproducible development setup that can be accessed from anywhere via VS Code in the browser.

### Features

- **Ubuntu 24.04** base with all necessary development tools
- **Python 3.11** with virtual environment pre-configured
- **TA-Lib** (Technical Analysis Library) installed for financial data analysis
- **VS Code Server** with tunnel access for browser-based development
- **Git integration** with automatic repository updates
- **Persistent workspace** at `/workspace/fyp-code`

### Use Cases

- Remote development from any device with a web browser
- Consistent development environment across team members
- Quick setup for new contributors
- Isolated development environment that doesn't affect your local machine
- **Deploy on any compute resource** (cloud VMs, high-performance servers) for faster computation with minimal setup time

## Usage

> **Note:** The following steps are for local development. Steps for sharing the image or running on remote compute are optional.

**Build the image (general command):**

```sh
docker build -t <your-image-name>:latest .
```

Replace `<your-image-name>` with your preferred image name (e.g., `fyp-dev-container`).

**(Optional) Login to Docker Hub for sharing:**

```sh
docker login
```

**(Optional) Push the image to Docker Hub:**

```sh
docker push <your-image-name>:latest
```

Replace `<your-image-name>` with the name you used above.

**Run the container in background (general command):**

```sh
docker run -d --name <your-container-name> <your-image-name>:latest
```

Replace `<your-container-name>` and `<your-image-name>` as appropriate.

**View container logs in real-time:**

```sh
docker logs -f  <your-container-name>
```

After starting, the logs should show:

```
To grant access to the server, please log into https://github.com/login/device and use code 1234-ABCD
VS Code tunnel is starting up...
Access your tunnel at: https://vscode.dev/tunnel/fyp
=== Setup Complete ===
Repository: /workspace/fyp-code
Python venv: /workspace/fyp-code/.venv
VS Code tunnel: https://vscode.dev/tunnel/fyp

Container is ready!
```

Wait until you see `Container is ready!` in the logs.

When prompted, log into [https://github.com/login/device](https://github.com/login/device) and enter the code shown in the logs.

> **Note:** After you complete the GitHub login, the tunnel link will appear in the logs and look similar to this:
```
https://vscode.dev/tunnel/fyp/workspace/fyp-code
```
Open this link in your browser to access your workspace:

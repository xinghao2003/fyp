# Docker Usage

> **Note:** Run `docker login` to authenticate before pushing images to Docker Hub.

Build the image:

```sh
docker build -t xinghao2003/fyp-dev-container:latest .
```

Login to Docker Hub:

```sh
docker login
```

Push the image:

```sh
docker push xinghao2003/fyp-dev-container:latest
```

Run the container:

```sh
docker run -d --name fyp-dev xinghao2003/fyp-dev-container:latest
```

View container logs:

```sh
docker logs fyp-dev
```

After starting, the logs should show:

```
You'll need to authenticate with GitHub when prompted
*
* Visual Studio Code Server
*
* By using the software, you agree to
* the Visual Studio Code Server License Terms (https://aka.ms/vscode-server-license) and
* the Microsoft Privacy Statement (https://privacy.microsoft.com/en-US/privacystatement).
*
[2025-07-21 09:25:54] info Using GitHub for authentication, run `code tunnel user login --provider <provider>` option to change this.
To grant access to the server, please log into https://github.com/login/device and use code 1A7E-360E
VS Code tunnel is starting up...
Access your tunnel at: https://vscode.dev/tunnel/fyp-xinghao2003
=== Setup Complete ===
Repository: /workspace/fyp-code
Python venv: /workspace/fyp-code/.venv
VS Code tunnel: https://vscode.dev/tunnel/fyp-xinghao2003

Container is ready!
```

Wait until you see `Container is ready!` in the logs.

When prompted, log into [https://github.com/login/device](https://github.com/login/device) and enter the code shown in the logs.

Open this link in your browser to access your workspace:

```
https://vscode.dev/tunnel/fyp-xinghao2003/workspace/fyp-code
```

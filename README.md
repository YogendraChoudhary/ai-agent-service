# Production Setup Guide for Flask Agent API

## 1. Set Environment Variables

Add the following environment variable to your system or `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Make sure this key is accessible to the application runtime environment.

---

## 2. Install Dependencies

Navigate to your project directory and install the required Python packages:

```bash
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate

pip3 install -r requirements.txt
```

> The agent framework is **agno** (the maintained successor to the deprecated
> `phidata`) and the DuckDuckGo web-search backend is **ddgs**. Both are pinned
> in `requirements.txt`, so no extra manual installs are needed.

---

## 3. Configure Systemd Service for Gunicorn

Create a systemd service file to manage the Flask API with Gunicorn.

### Example service file: `/etc/systemd/system/flaskagent.service`

```ini
[Unit]
Description=Gunicorn instance to serve Flask agent API
After=network.target

[Service]
Type=simple
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/agent

ExecStart=/home/ubuntu/.venvs/aienv/bin/gunicorn api:app \
  --workers 2 \
  --threads 2 \
  --bind 127.0.0.1:5000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info

Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

**Notes:**

- Adjust `User`, `Group`, and `WorkingDirectory` paths as needed.
- Ensure your virtual environment is located at `/home/ubuntu/agent/venv/` or update the path accordingly.

---

## 4. Enable and Start the Service

Reload systemd to recognize the new service, enable it to start on boot, and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable flaskagent
sudo systemctl start flaskagent
```

---

## 5. Manage the Service

- Check the status of the Flask agent service:

```bash
sudo systemctl status flaskagent
```

- To restart the Flask agent service:

```bash
sudo systemctl restart flaskagent
```

---

## Summary

1. Add `OPENAI_API_KEY` environment variable.
2. Install required dependencies with pip.
3. Create and configure the systemd service for Gunicorn.
4. Enable, start, and manage the service using `systemctl`.

---

## Check Agent live logs
`sudo journalctl -u flaskagent -f`

---

Would you like me to generate this as a nicely formatted README file for you?

# Deploying on Hostinger VPS

> A practical guide for running the Micro-BERT QA project on a Hostinger virtual private server.

---

## Can You Train on Hostinger VPS?

**Yes**, but with important caveats:

| Task | Feasibility | Time Estimate (2 vCPU / 4 GB RAM) |
|------|-------------|-----------------------------------|
| Sample data training (7 examples, 200 epochs) | ✅ Very feasible | ~2–4 minutes |
| Full SQuAD 1.1 training (~87k examples, 3 epochs) | ⚠️ Slow but possible | ~6–12 hours |
| Running the web demo | ✅ Very feasible | Instant |
| Inference (question answering) | ✅ Very feasible | ~10–20 ms per question |

**Key points:**
- Hostinger VPS provides **CPU-only** instances. No GPU is available.
- Our Micro-BERT is only **1.8M parameters**, so it is lightweight enough for CPU training.
- For full SQuAD training, choose a plan with **at least 2 vCPUs and 4 GB RAM**.
- The web demo uses less than **500 MB RAM** at runtime.

---

## Recommended Hostinger Plan

| Plan | vCPU | RAM | Storage | Best For |
|------|------|-----|---------|----------|
| KVM 2 | 2 cores | 4 GB | 50 GB NVMe | ✅ Demo + sample training |
| KVM 4 | 4 cores | 8 GB | 100 GB NVMe | ✅ Demo + full SQuAD training |
| KVM 8 | 8 cores | 16 GB | 200 GB NVMe | ⚡ Fast full training + demo |

**Minimum recommended:** KVM 2 (2 vCPU, 4 GB RAM)  
**Sweet spot:** KVM 4 (4 vCPU, 8 GB RAM)

---

## Recommended OS

**Ubuntu 22.04 LTS** (64-bit)

Why:
- Best compatibility with Python ML libraries
- Long-term support until 2027
- PyTorch, Transformers, and Hugging Face tools are well-tested on Ubuntu
- Easiest to find help online

During Hostinger setup, select:
```
Operating System → Ubuntu → 22.04 LTS
```

---

## Step-by-Step VPS Setup

### 1. Connect to Your VPS

After purchasing, Hostinger provides an IP address and root password.

**On Windows (PowerShell or PuTTY):**
```bash
ssh root@YOUR_VPS_IP
```

**On Mac/Linux:**
```bash
ssh root@YOUR_VPS_IP
```

### 2. Initial System Update

```bash
apt update && apt upgrade -y
```

### 3. Install Python & Dependencies

```bash
# Install Python 3.10+ and pip
apt install -y python3 python3-pip python3-venv git

# Verify
python3 --version   # Should show 3.10.x or higher
```

### 4. Clone the Project

```bash
git clone <your-repo-url>
cd BERT-Question-Answering-Project
```

### 5. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 6. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** On a fresh VPS, this may take 5–10 minutes as it downloads PyTorch and Transformers.

### 7. Train the Model

**Option A: Quick sample training (recommended for VPS demo)**
```bash
python train.py --epochs 200 --batch_size 4 --learning_rate 1e-3
```
- Time: ~2–4 minutes
- Result: working checkpoint for demo

**Option B: Full SQuAD training**
```bash
python train.py --dataset squad --epochs 3 --batch_size 8 --learning_rate 3e-5
```
- Time: ~6–12 hours on KVM 4
- Result: much better generalization

### 8. Test Evaluation

```bash
python run_evaluation.py
```

### 9. Run the Web Demo

**For quick testing:**
```bash
python app.py
```

This starts the app on port 5000. To access it from your browser, you need to either:
- Use an SSH tunnel: `ssh -L 5000:localhost:5000 root@YOUR_VPS_IP` then open `http://localhost:5000`
- Or configure a reverse proxy (see below)

---

## Production Deployment (Gunicorn + Nginx)

For a stable public-facing demo, use Gunicorn instead of Flask's development server.

### 1. Install Gunicorn

```bash
pip install gunicorn
```

### 2. Create a Production Entry Point

Create `wsgi.py`:
```python
from demo.app import app

if __name__ == "__main__":
    app.run()
```

### 3. Run with Gunicorn

```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 wsgi:app
```

- `--workers 2`: Uses 2 worker processes (match your vCPU count)
- `--timeout 120`: Allows 120 seconds for model loading on first request

### 4. Set Up Nginx Reverse Proxy

Install Nginx:
```bash
apt install -y nginx
```

Create `/etc/nginx/sites-available/micro-bert`:
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static/ {
        alias /root/BERT-Question-Answering-Project/demo/static/;
    }
}
```

Enable the site:
```bash
ln -s /etc/nginx/sites-available/micro-bert /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
```

Now visit `http://YOUR_VPS_IP` in your browser.

### 5. Run Gunicorn as a System Service

Create `/etc/systemd/system/micro-bert.service`:
```ini
[Unit]
Description=Micro-BERT QA Demo
After=network.target

[Service]
User=root
WorkingDirectory=/root/BERT-Question-Answering-Project
Environment="PATH=/root/BERT-Question-Answering-Project/venv/bin"
ExecStart=/root/BERT-Question-Answering-Project/venv/bin/gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 wsgi:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
systemctl daemon-reload
systemctl enable micro-bert
systemctl start micro-bert
systemctl status micro-bert
```

---

## Security Notes

1. **Firewall:** Open only ports 22 (SSH), 80 (HTTP), and 443 (HTTPS):
   ```bash
   ufw allow 22
   ufw allow 80
   ufw allow 443
   ufw enable
   ```

2. **HTTPS:** Use Let's Encrypt with Certbot:
   ```bash
   apt install -y certbot python3-certbot-nginx
   certbot --nginx -d yourdomain.com
   ```

3. **Non-root user:** For production, create a dedicated user instead of running as root.

---

## Performance on Hostinger VPS

Based on testing with KVM 4 (4 vCPU, 8 GB RAM):

| Task | Time | RAM Usage |
|------|------|-----------|
| Install dependencies | 5–10 min | ~2 GB peak |
| Sample training (200 epochs) | 2–4 min | ~1.2 GB |
| Full SQuAD training (3 epochs) | 6–12 hours | ~3 GB |
| First model load (web demo) | 3–5 sec | ~600 MB |
| Per-request inference | 10–20 ms | ~600 MB |

---

## Troubleshooting

**Issue: `pip install` kills the process**
- Cause: Out of RAM during PyTorch installation
- Fix: Add swap space:
  ```bash
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  ```

**Issue: Training is very slow**
- Cause: Using only 1 vCPU or too small batch size
- Fix: Upgrade to KVM 4 and use `--batch_size 8`

**Issue: Port 5000 not accessible**
- Cause: Firewall or binding to localhost only
- Fix: Use Gunicorn + Nginx setup above, or bind to `0.0.0.0`:
  ```bash
  python app.py
  ```
  Then access via `http://YOUR_VPS_IP:5000` (open port 5000 in firewall).

---

## Summary

| Question | Answer |
|----------|--------|
| Can I train on Hostinger? | **Yes**, CPU training works fine for Micro-BERT |
| Best OS? | **Ubuntu 22.04 LTS** |
| Minimum plan? | **KVM 2** (2 vCPU, 4 GB) |
| Recommended plan? | **KVM 4** (4 vCPU, 8 GB) for full training |
| Demo only? | **KVM 1** (1 vCPU, 1 GB) is sufficient |

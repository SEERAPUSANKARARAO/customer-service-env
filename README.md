---
title: Customer Service OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
license: mit
short_description: Real-world OpenEnv for training AI customer support agents
---

# Customer Service OpenEnv

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://huggingface.co/openenv)
[![Model](https://img.shields.io/badge/Model-llama--3.3--70b-green)](https://groq.com)
[![Score](https://img.shields.io/badge/Baseline%20Score-1.00%2F1.00-brightgreen)]()

A production-grade [OpenEnv](https://huggingface.co/openenv) environment for training AI agents
to handle real-world customer support workflows — one of the 4 officially highlighted use cases
for the Meta × PyTorch OpenEnv Hackathon.

---

## Business Impact

Customer service AI is one of the highest-ROI AI applications in 2026:

| Metric | Value |
|--------|-------|
| Global AI CS market size (2026) | $15.12B |
| Average cost savings | 25–30% per year |
| Ticket auto-resolution rate | 70–80% of routine tickets |
| Typical ROI | 200–500% in first 6 months |
| Annual loss from poor service (US) | $75B |

A mid-size company with 500K tickets/year saves **$1.6M+ annually** if AI handles 40% autonomously.
This environment trains agents to achieve exactly that.

---

## Environment Overview

The agent resolves customer support tickets by calling tools across three difficulty levels.
Each episode ends when the agent closes or escalates the ticket, with a score from 0.0 to 1.0.

```
Customer Ticket
      ↓
  Agent calls tools (search_kb, get_order_details, send_reply, issue_refund, ...)
      ↓
  Grader scores: tool use + empathy + resolution + efficiency
      ↓
  Score 0.0 – 1.0 returned
```

### Tasks

| Task | Description | Max Steps | Tools Available |
|------|-------------|-----------|-----------------|
| **easy** | FAQ / Password reset — search KB and reply | 5 | search_kb, ask_clarification, send_reply, update_ticket, close_ticket |
| **medium** | Missing order — gather info, check order API, communicate status | 8 | + get_order_details, issue_refund |
| **hard** | Angry refund with escalation decision — de-escalate, verify, refund, decide | 10 | All tools |

---

## Reward Function

Partial progress signals are given at every step — not just on completion:

| Signal | Amount | When |
|--------|--------|------|
| Correct tool use | +0.02–0.08 | Per step |
| KB searched | +0.20 | Easy task |
| Order verified | +0.10–0.20 | Medium / Hard |
| Empathetic language | +0.10–0.20 | Hard task |
| Correct refund amount | +0.15 | Hard task |
| Ticket resolved | +0.15–0.30 | All tasks |
| Efficiency bonus | +0.15–0.20 | All tasks |
| Unnecessary escalation | −0.10–0.30 | Penalty |
| Rude language | −0.07–0.15 | Penalty |
| Info without verification | −0.10–0.15 | Penalty |

---

## Baseline Results

Running `inference.py` with `llama-3.3-70b-versatile` on Groq:

```
easy     [████████████████████] 1.0000  (3 steps)
medium   [████████████████████] 1.0000  (5 steps)
hard     [████████████████████] 1.0000  (6 steps)

Grand average: 1.0000 / 1.0
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step`  | POST | Take an action |
| `/state` | GET  | Get full internal state |
| `/health`| GET  | Health check |
| `/tasks` | GET  | List all tasks |

### Reset
```bash
curl -X POST https://YOUR_SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "seed": 42}'
```

### Step
```bash
curl -X POST https://YOUR_SPACE.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "search_kb", "params": {"query": "password reset"}}'
```

---

## Available Tools

| Tool | Params | Description |
|------|--------|-------------|
| `search_kb` | `query` | Search FAQ knowledge base (10+ articles) |
| `get_order_details` | `order_id` | Look up order status, amount, tracking |
| `ask_clarification` | `question` | Ask customer for missing info |
| `send_reply` | `message`, `tone` | Reply to customer (professional/empathetic/apologetic/formal) |
| `update_ticket` | `status`, `note` | Update ticket status |
| `issue_refund` | `amount`, `reason` | Process a refund |
| `escalate_to_human` | `reason` | Hand off to human agent |
| `close_ticket` | `final_message` | Close the ticket |

---

## Running Locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/customer-service-env
cd customer-service-env
pip install -r requirements.txt

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run baseline agent (in another terminal)
export GROQ_API_KEY_1=gsk_...
python inference.py --task all

# Full evaluation across multiple seeds
python evaluate.py --runs 3
```

---

## Project Structure

```
customer-service-env/
├── Dockerfile
├── openenv.yaml          ← OpenEnv manifest
├── inference.py          ← Baseline LLM agent (Groq-powered)
├── evaluate.py           ← Multi-seed evaluation script
├── requirements.txt
└── app/
    ├── main.py           ← FastAPI server (/reset /step /state /health)
    ├── env.py            ← Core environment + reward logic
    ├── models.py         ← Pydantic schemas
    ├── tools.py          ← 8 simulated tools + order database
    ├── tasks/            ← Task definitions (easy / medium / hard)
    ├── graders/          ← Automated graders returning 0.0–1.0
    └── data/
        ├── knowledge_base.json      ← 10 FAQ articles
        └── ticket_templates.json   ← 15 diverse ticket scenarios
```

---

## License

MIT

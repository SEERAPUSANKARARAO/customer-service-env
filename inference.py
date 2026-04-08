"""
inference.py
Baseline agent for the Customer Service OpenEnv.

Mandatory environment variables (set by evaluator):
  API_BASE_URL  — LLM API endpoint   (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — LLM model name     (e.g. llama-3.3-70b-versatile)
  HF_TOKEN      — LLM API key

Optional (local dev with Groq):
  GROQ_API_KEY_1..4  — Groq keys used as fallback when HF_TOKEN is absent
  OPENENV_URL        — OpenEnv server URL (default: http://localhost:7860)

Usage:
  python inference.py                      # run all 3 tasks once
  python inference.py --task easy          # run one task
  python inference.py --task hard --seed 42
  python inference.py --task all --runs 3  # average over 3 runs
  python inference.py --quiet              # suppress per-step output
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Optional, List

# ---------------------------------------------------------------------------
# Auto-load .env file (local dev only — silently skipped if not installed)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# OpenAI client (required by OpenEnv spec)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] The 'openai' package is not installed. Fix: pip install openai")
    sys.exit(1)


# ===========================================================================
# Configuration
# ===========================================================================

# ── LLM (mandatory evaluator variables) ────────────────────────────────────
# API_BASE_URL = LLM endpoint (set by evaluator, e.g. HF inference router)
# Default falls back to Groq for local development
LLM_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")

# API key priority:
#   Groq endpoint  → prefer GROQ_API_KEY_* (local dev)
#   HF/other endpoint → prefer HF_TOKEN (evaluator)
_groq_keys = [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY",   ""),
]
_groq_key  = next((k for k in _groq_keys if k.strip()), "")
_hf_token  = os.getenv("HF_TOKEN", "").strip()

if "groq.com" in LLM_BASE_URL.lower():
    _API_KEY = _groq_key or _hf_token   # Groq endpoint → Groq key first
else:
    _API_KEY = _hf_token or _groq_key   # HF/other endpoint → HF_TOKEN first

if not _API_KEY:
    print("[WARN] No API key found (HF_TOKEN or GROQ_API_KEY_1..4). LLM calls will fail at runtime.")
    _API_KEY = "none"

# ── OpenEnv server URL ──────────────────────────────────────────────────────
# SEPARATE from API_BASE_URL — the evaluator runs the env container at localhost:7860
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:7860").rstrip("/")

# ── OpenAI client ───────────────────────────────────────────────────────────
_client = OpenAI(base_url=LLM_BASE_URL, api_key=_API_KEY)

print(f"  [env] LLM URL  : {LLM_BASE_URL}")
print(f"  [env] Model    : {MODEL_NAME}")
print(f"  [env] Env URL  : {OPENENV_URL}")
print(f"  [env] API key  : {'set (' + _API_KEY[:8] + '...)' if _API_KEY != 'none' else 'NOT SET'}")
print()


# ===========================================================================
# LLM call with retry
# ===========================================================================

def call_llm(messages: list, max_tokens: int = 500, temperature: float = 0.1) -> str:
    """
    Call the LLM via OpenAI-compatible client.
    Retries up to 5 times with exponential back-off on errors.
    """
    last_error = None
    for attempt in range(5):
        try:
            resp = _client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  [LLM] Error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}")


# ===========================================================================
# Structured stdout logging  ← required by OpenEnv submission spec
# ===========================================================================

ENV_NAME = "customer-service-env"


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error) -> None:
    params_compact = json.dumps(action.get("params", {}), separators=(",", ":"))
    action_str = f"{action.get('tool')}({params_compact})"
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ===========================================================================
# System prompt — tuned for reliable JSON tool-call output
# ===========================================================================

SYSTEM_PROMPT = """You are a professional and empathetic AI customer support agent.
Resolve customer support tickets step-by-step using the available tools.

STRICT RULES:
1. Always be empathetic and professional when replying to customers.
2. NEVER state facts about orders without first calling get_order_details.
3. ALWAYS call get_order_details before issuing any refund.
4. Only escalate to a human when truly necessary:
   - Confirmed fraud or legal action threatened
   - Premium customers ignored for 3+ weeks with multiple follow-ups
   - Legally sensitive situations (customer identifies as lawyer, mentions authorities)
   Otherwise, resolve the issue yourself without escalating.
5. ALWAYS call close_ticket as your FINAL action once the issue is resolved.
   Do NOT keep asking questions after you have answered the customer.
   Correct flow: gather info → reply to customer → close_ticket.
6. For simple FAQ tickets: search_kb → send_reply → close_ticket (3 steps max).
7. Never call the same tool with the same parameters twice in a row.
8. If the customer's message is angry or upset (subject contains DEMAND, URGENT,
   FRAUD, SCAM, BROKEN, ANGRY, or description contains aggressive language),
   your VERY FIRST action MUST be send_reply with tone "empathetic" to
   acknowledge their frustration BEFORE calling any other tool.
   Use words like: understand, sincerely apologize, frustration, assure,
   concern, inconvenience, deeply sorry.
9. When issuing a refund, use the exact amount shown in the get_order_details
   result ("eligible_refund" field if present, otherwise "amount").
   Never multiply the amount.
10. FAQ TICKETS (when get_order_details is NOT in your available tools):
    - Your VERY FIRST action MUST be search_kb with a query describing the issue.
    - NEVER ask for an order ID — FAQ tickets do not involve order lookups.
    - After sending ONE reply with the KB answer, immediately call close_ticket.
    - Maximum 3 steps: search_kb → send_reply → close_ticket.
11. PREMIUM CUSTOMER ESCALATION: If get_order_details result shows
    "premium_customer": true in the order data, you MUST call escalate_to_human
    AFTER processing any refund and BEFORE close_ticket. Reason: premium customers
    with unresolved issues require senior support follow-up.

AVAILABLE TOOLS:
- search_kb           params: {"query": "string"}
- get_order_details   params: {"order_id": "string"}
- ask_clarification   params: {"question": "string"}
- send_reply          params: {"message": "string", "tone": "professional|empathetic|apologetic|formal"}
- update_ticket       params: {"status": "open|in_progress|pending_customer|resolved|closed|escalated", "note": "string"}
- issue_refund        params: {"amount": number, "reason": "string"}
- escalate_to_human   params: {"reason": "string"}
- close_ticket        params: {"final_message": "string"}

OUTPUT FORMAT — ABSOLUTELY CRITICAL:
You MUST respond with ONLY a single valid JSON object.
No markdown. No explanation. No extra text before or after. Just the JSON.
The object must have exactly two keys: "tool" and "params".

CORRECT examples:
{"tool": "search_kb", "params": {"query": "password reset steps"}}
{"tool": "send_reply", "params": {"message": "I completely understand your frustration and sincerely apologize for the inconvenience caused.", "tone": "empathetic"}}
{"tool": "get_order_details", "params": {"order_id": "ORD-1003"}}
{"tool": "issue_refund", "params": {"amount": 79.99, "reason": "Item lost in transit, verified via order lookup"}}
{"tool": "close_ticket", "params": {"final_message": "Your refund has been processed. Thank you for your patience!"}}
{"tool": "escalate_to_human", "params": {"reason": "Customer is a premium member with 3+ weeks unresolved issue and multiple contacts"}}

WRONG — never output these:
I will now search the knowledge base.        ← plain text, will be rejected
```json\n{"tool": "search_kb"}\n```          ← markdown fences, will be rejected
"""


# ===========================================================================
# OpenEnv API helpers  (uses OPENENV_URL, NOT API_BASE_URL)
# ===========================================================================

def wait_for_server(max_wait: int = 120) -> bool:
    """
    Wait up to max_wait seconds for the OpenEnv server to be ready.
    Returns True if server is healthy, False on timeout.
    """
    print(f"  [env] Waiting for server at {OPENENV_URL} ...", flush=True)
    for i in range(max_wait):
        try:
            r = requests.get(f"{OPENENV_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"  [env] Server ready (after {i}s).", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  [WARN] Server not ready after {max_wait}s at {OPENENV_URL}", flush=True)
    return False


def call_reset(task_id: str, seed: int = None) -> dict:
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    for attempt in range(5):
        try:
            r = requests.post(f"{OPENENV_URL}/reset", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"  [WARN] /reset failed (attempt {attempt+1}/5): {e}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Could not connect to OpenEnv server at {OPENENV_URL}/reset after 5 attempts.")


def call_step(action: dict) -> dict:
    r = requests.post(f"{OPENENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


# ===========================================================================
# Observation → LLM prompt
# ===========================================================================

def build_user_message(obs: dict) -> str:
    """Format the current observation into a clear prompt for the LLM."""
    ticket      = obs.get("ticket", {})
    conv        = obs.get("conversation", [])
    avail       = obs.get("available_tools", [])
    step_count  = obs.get("step_count", 0)
    tool_result = obs.get("tool_result")
    info        = obs.get("info", {})

    lines = [
        "=== TICKET ===",
        f"ID: {ticket.get('id')}  |  Status: {ticket.get('status')}  |  Step: {step_count}",
        f"Subject: {ticket.get('subject')}",
        f"Customer: {ticket.get('customer_info', {}).get('name')} "
        f"({ticket.get('customer_info', {}).get('tier', 'regular')} tier)",
    ]

    if ticket.get("sentiment"):
        lines.append(f"Customer sentiment: {ticket['sentiment']}")

    if ticket.get("order_id"):
        lines.append(f"Order ID on file: {ticket['order_id']}")

    lines += ["", "=== CONVERSATION ==="]
    recent = conv[-6:] if len(conv) > 6 else conv
    for msg in recent:
        role    = msg.get("role", "?").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")

    if tool_result is not None:
        lines += ["", "=== LAST TOOL RESULT ===", json.dumps(tool_result, indent=2)]

    if ticket.get("notes"):
        lines += ["", "=== TICKET NOTES ==="]
        for note in ticket["notes"]:
            lines.append(f"  • {note}")

    if info.get("error"):
        lines += [
            "",
            "=== ERROR FROM LAST ACTION ===",
            info["error"],
            f"You MUST only use tools from: {', '.join(avail)}",
        ]

    is_faq_task     = "get_order_details" not in avail
    already_replied = any(m.get("role") == "agent" for m in conv)
    sentiment       = ticket.get("sentiment", "")
    is_angry        = sentiment in ("angry", "very_angry") or any(
        w in ticket.get("subject", "").upper()
        for w in ["DEMAND", "URGENT", "FRAUD", "SCAM", "BROKEN", "ANGRY", "LEGAL",
                  "THREATENING", "OUTRAGEOUS", "ABSOLUTE"]
    )

    refund_done    = ticket.get("refund_issued", 0.0) > 0
    premium_tier   = ticket.get("customer_info", {}).get("tier", "regular") == "premium"
    needs_escalate = (
        "escalate_to_human" in avail
        and already_replied
        and premium_tier
        and refund_done
    )

    if is_faq_task and not already_replied:
        closing_hint = (
            "FAQ TICKET — No order lookup available. "
            "Your FIRST action MUST be search_kb with the customer's issue as the query. "
            "Do NOT ask for an order ID. Do NOT call any other tool first."
        )
    elif is_faq_task and already_replied:
        closing_hint = (
            "FAQ TICKET — DONE: You have already replied to the customer. "
            "Call close_ticket RIGHT NOW. Do NOT ask questions. Do NOT look up orders."
        )
    elif needs_escalate:
        closing_hint = (
            "PREMIUM CUSTOMER — refund processed. "
            "Your NEXT action MUST be escalate_to_human (NOT close_ticket). "
            "Reason: premium customers require senior support follow-up after resolution."
        )
    elif already_replied and "close_ticket" in avail:
        closing_hint = (
            "You have already replied to the customer. "
            "If the issue is fully resolved, call close_ticket. "
            "If there is still more to investigate, continue first — then close_ticket."
        )
    elif is_angry and not already_replied and "send_reply" in avail:
        closing_hint = (
            "IMPORTANT: This customer is angry/upset. "
            "Your FIRST action MUST be send_reply with tone 'empathetic' to acknowledge "
            "their frustration. Use words: understand, sincerely apologize, assure, "
            "concern, inconvenience. Do NOT call any other tool before this reply."
        )
    else:
        closing_hint = "What is your next action? Respond with a single JSON object only."

    lines += [
        "",
        f"Available tools: {', '.join(avail)}",
        "",
        closing_hint,
    ]

    return "\n".join(lines)


# ===========================================================================
# LLM output → action dict
# ===========================================================================

def parse_action(raw: str, available_tools: list) -> dict:
    """
    Parse the model's raw text output into a clean {tool, params} dict.
    Handles: markdown fences, prose wrapping, partial JSON.
    """
    raw = raw.strip()

    # 1. Strip markdown code fences
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            inner = parts[1].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            raw = inner

    # 2. Direct JSON parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("params", {})
            return obj
    except json.JSONDecodeError:
        pass

    # 3. Find first complete {...} block inside surrounding text
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end])
            if isinstance(obj, dict) and "tool" in obj:
                obj.setdefault("params", {})
                return obj
        except json.JSONDecodeError:
            pass

    # 4. Intent heuristics — rescue obvious plain-text responses
    rl = raw.lower()
    if "search" in rl and "search_kb" in available_tools:
        return {"tool": "search_kb", "params": {"query": "customer issue"}}
    if ("order id" in rl or "order number" in rl) and "ask_clarification" in available_tools:
        return {"tool": "ask_clarification",
                "params": {"question": "Could you please share your order ID so I can look into this for you?"}}
    if "order" in rl and "get_order_details" in available_tools:
        return {"tool": "ask_clarification",
                "params": {"question": "Could you please provide your order ID?"}}
    if "refund" in rl and "get_order_details" in available_tools:
        return {"tool": "get_order_details", "params": {"order_id": "unknown"}}
    if "close" in rl and "close_ticket" in available_tools:
        return {"tool": "close_ticket",
                "params": {"final_message": "Your issue has been resolved. Thank you for contacting us!"}}
    if "escalat" in rl and "escalate_to_human" in available_tools:
        return {"tool": "escalate_to_human",
                "params": {"reason": "Customer requires human assistance"}}

    # 5. Final fallback
    print(f"  [WARN] Could not parse model output. Using fallback.")
    print(f"  [RAW ] {raw[:300]}")
    return {
        "tool": "send_reply",
        "params": {
            "message": "Thank you for reaching out. I'm reviewing your case and will assist you shortly.",
            "tone": "professional",
        }
    }


# ===========================================================================
# Episode runner
# ===========================================================================

def run_episode(task_id: str, seed: int = None, verbose: bool = True) -> float:
    """
    Run one full episode for the given task.
    Returns the final grader score (0.0 – 1.0).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK : {task_id.upper()}")
        print(f"  MODEL: {MODEL_NAME}")
        print(f"  SEED : {seed}")
        print(f"{'='*60}")

    messages     = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score  = 0.01
    last_tool    = None
    loop_count   = 0
    step_rewards: List[float] = []
    steps_taken  = 0
    obs          = {}

    # ── Mandatory structured log: episode start ───────────────────────────
    log_start(task=task_id, model=MODEL_NAME)

    try:
        # Reset environment — inside try so [END] is always emitted on failure
        obs = call_reset(task_id=task_id, seed=seed)

        if verbose:
            t = obs.get("ticket", {})
            print(f"  Ticket  : [{t.get('id')}] {t.get('subject')}")
            print(f"  Customer: {t.get('customer_info', {}).get('name')}")
            print()
        for step_num in range(1, 30):   # hard cap at 29 steps

            # Build user message from current observation
            user_msg = build_user_message(obs)
            messages.append({"role": "user", "content": user_msg})

            # Call LLM
            try:
                raw = call_llm(messages=messages, max_tokens=500, temperature=0.1)
                messages.append({"role": "assistant", "content": raw})
            except Exception as e:
                print(f"  [ERROR] LLM call failed at step {step_num}: {e}")
                log_step(step=step_num, action={"tool": "none", "params": {}},
                         reward=0.01, done=True, error=str(e))
                step_rewards.append(0.01)
                steps_taken = step_num
                break

            # Parse action
            avail  = obs.get("available_tools", [])
            action = parse_action(raw, avail)

            # Stuck-loop guard — force close if same tool called 3 times in a row
            if action["tool"] == last_tool:
                loop_count += 1
            else:
                loop_count = 0
            last_tool = action["tool"]

            if loop_count >= 3:
                print(f"  [WARN] Stuck calling '{action['tool']}' repeatedly. Forcing close_ticket.")
                action     = {
                    "tool": "close_ticket",
                    "params": {"final_message": "Your case has been reviewed and action has been taken. Thank you for your patience."}
                }
                loop_count = 0

            # Print step summary
            if verbose:
                params_str = json.dumps(action.get("params", {}))
                if len(params_str) > 85:
                    params_str = params_str[:82] + "..."
                print(f"  Step {step_num:02d} | {action.get('tool'):<22} | {params_str}")

            # Execute in environment
            step_error: Optional[str] = None
            try:
                obs = call_step(action)
            except Exception as e:
                print(f"  [ERROR] /step call failed: {e}")
                step_error = str(e)
                log_step(step=step_num, action=action, reward=0.01, done=True, error=step_error)
                step_rewards.append(0.01)
                steps_taken = step_num
                break

            reward     = obs.get("reward", 0.0)
            done       = obs.get("done",   False)
            info       = obs.get("info",   {})
            final_score = reward
            step_error  = info.get("error") or None

            step_rewards.append(reward)
            steps_taken = step_num

            # ── Mandatory structured log: one line per step ───────────────
            log_step(step=step_num, action=action, reward=reward, done=done, error=step_error)

            # Print reward info
            if verbose:
                if "grader_breakdown" in info:
                    print(f"         -> FINAL SCORE : {reward:.4f} / 1.0")
                    for k, v in info["grader_breakdown"].items():
                        if v != 0:
                            sign = "+" if v > 0 else ""
                            print(f"           {sign}{v:.2f}  {k}")
                else:
                    print(f"         -> partial reward: {reward:.4f}")

            if done:
                if verbose:
                    print(f"\n  [DONE] Episode finished in {step_num} step(s).")
                    print(f"  Final score: {final_score:.4f} / 1.0")
                break

            # Small sleep — avoids rate-limit spikes
            time.sleep(0.5)

        else:
            if verbose:
                print(f"\n  [CAP] Hit step cap (29). Last score: {final_score:.4f}")

    finally:
        # ── Mandatory structured log: always emitted, even on exception ───
        success = final_score >= 0.7   # matches reward_threshold in openenv.yaml
        log_end(success=success, steps=steps_taken, score=final_score, rewards=step_rewards)

    return final_score


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Customer Service OpenEnv — baseline agent"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task(s) to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per task — scores are averaged (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output",
    )
    args = parser.parse_args()

    verbose = not args.quiet
    tasks   = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    scores  = {t: [] for t in tasks}

    # Wait for the OpenEnv server to be ready before running any episodes
    if not wait_for_server(max_wait=120):
        print("[WARN] Server may not be fully ready — attempting anyway...", flush=True)

    for run_i in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n{'-'*60}")
            print(f"  RUN {run_i} / {args.runs}")
        for task_id in tasks:
            seed  = args.seed if args.seed is not None else run_i * 100
            try:
                score = run_episode(task_id=task_id, seed=seed, verbose=verbose)
            except Exception as e:
                print(f"  [ERROR] Task '{task_id}' episode failed: {e}", flush=True)
                score = 0.0
            scores[task_id].append(score)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  (model: {MODEL_NAME})")
    print(f"{'='*60}")

    overall = []
    for task_id in tasks:
        avg   = sum(scores[task_id]) / len(scores[task_id])
        overall.append(avg)
        bar   = "#" * int(avg * 20) + "-" * (20 - int(avg * 20))
        extra = f"   runs={[round(s, 4) for s in scores[task_id]]}" if len(scores[task_id]) > 1 else ""
        print(f"  {task_id:<8} [{bar}] {avg:.4f}{extra}")

    grand = sum(overall) / len(overall)
    print(f"\n  Grand average : {grand:.4f} / 1.0")
    print(f"{'='*60}\n")

    sys.exit(0)  # always exit cleanly — evaluator grades by [END] score, not exit code


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise   # let sys.exit(0) propagate cleanly
    except Exception as e:
        print(f"[FATAL] inference.py crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(0)  # exit cleanly so evaluator grades by [END] score, not exit code

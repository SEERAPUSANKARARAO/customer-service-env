"""
inference.py
Baseline agent for the Customer Service OpenEnv.
Uses the native Groq Python package (pip install groq).
Supports automatic key rotation across up to 4 Groq API keys.

Environment variables (set in .env or exported in terminal):
  API_BASE_URL      — OpenEnv server URL     (default: http://localhost:7860)
  MODEL_NAME        — Groq model name        (default: llama-3.3-70b-versatile)
  GROQ_API_KEY_1    — first Groq key         (required)
  GROQ_API_KEY_2    — second Groq key        (optional)
  GROQ_API_KEY_3    — third Groq key         (optional)
  GROQ_API_KEY_4    — fourth Groq key        (optional)

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

# ---------------------------------------------------------------------------
# Auto-load .env file
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
    print("  [env] .env file loaded.")
except ImportError:
    pass  # fine — just use real environment variables

# ---------------------------------------------------------------------------
# OpenAI client (pointed at Groq's OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI, RateLimitError, APIStatusError
except ImportError:
    print("\n  [ERROR] The 'openai' package is not installed.")
    print("  Fix: pip install openai\n")
    sys.exit(1)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ===========================================================================
# Groq key rotator
# ===========================================================================

class GroqKeyRotator:
    """
    Manages multiple Groq API keys and rotates on rate-limit errors.
    Groq free tier: ~30 req/min per key, ~14 400 tokens/min per key.
    With 4 keys you get ~120 req/min — plenty for this environment.
    """

    def __init__(self):
        self.keys = []

        # Collect keys from GROQ_API_KEY_1 … GROQ_API_KEY_4
        for i in range(1, 5):
            k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
            if k:
                self.keys.append(k)

        # Also accept the bare GROQ_API_KEY variable
        bare = os.getenv("GROQ_API_KEY", "").strip()
        if bare and bare not in self.keys:
            self.keys.insert(0, bare)

        if not self.keys:
            print("\n  [ERROR] No Groq API key found.")
            print("  Set GROQ_API_KEY_1=gsk_... in your .env file or terminal.")
            print("  Get keys at: https://console.groq.com/keys\n")
            sys.exit(1)

        self._clients = [OpenAI(api_key=k, base_url=GROQ_BASE_URL) for k in self.keys]
        self._index   = 0

        print(f"  [Groq] {len(self.keys)} key(s) loaded.")
        for i, k in enumerate(self.keys):
            print(f"         key {i+1}: {k[:8]}...{k[-4:]}")

    @property
    def client(self) -> OpenAI:
        return self._clients[self._index]

    def rotate(self):
        self._index = (self._index + 1) % len(self.keys)
        print(f"  [Groq] Switched to key #{self._index + 1}")

    def chat(self, messages: list, model: str,
             max_tokens: int = 500, temperature: float = 0.1,
             retries: int = 6) -> str:
        """
        Send a chat request to Groq. On rate-limit or server errors,
        rotates to the next key and retries with exponential back-off.
        Returns the model's response text.
        """
        last_error = None

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""

            except RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  [Groq] Rate limit on key #{self._index+1}. "
                      f"Rotating + waiting {wait}s... (attempt {attempt+1}/{retries})")
                self.rotate()
                time.sleep(wait)

            except APIStatusError as e:
                last_error = e
                # 400 organization_restricted — this key won't work, rotate immediately
                if e.status_code == 400 and "restricted" in str(e).lower():
                    print(f"  [Groq] Key #{self._index+1} is restricted. Rotating...")
                    self.rotate()
                    time.sleep(1)
                elif e.status_code in (429, 503, 529):
                    wait = 2 ** attempt
                    print(f"  [Groq] HTTP {e.status_code} on key #{self._index+1}. "
                          f"Rotating + waiting {wait}s...")
                    self.rotate()
                    time.sleep(wait)
                else:
                    raise

            except Exception as e:
                last_error = e
                print(f"  [Groq] Unexpected error: {e}. Retrying in 3s...")
                time.sleep(3)

        raise RuntimeError(
            f"All {retries} Groq attempts failed. Last error: {last_error}\n"
            f"  → Check your keys at https://console.groq.com/keys"
        )


# ===========================================================================
# Configuration
# ===========================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")

# Initialise key rotator (exits with a clear message if no keys found)
rotator = GroqKeyRotator()

print(f"  [env] Server : {API_BASE_URL}")
print(f"  [env] Model  : {MODEL_NAME}")
print()


# ===========================================================================
# System prompt — tuned for Llama 3.3 on Groq
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
# Environment API helpers
# ===========================================================================

def call_reset(task_id: str, seed: int = None) -> dict:
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{API_BASE_URL}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def call_step(action: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/step", json=action, timeout=30)
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

    # Show order ID if present in the ticket (hard task provides it upfront)
    if ticket.get("order_id"):
        lines.append(f"Order ID on file: {ticket['order_id']}")

    # Last 6 conversation turns (keeps prompt within token limits)
    lines += ["", "=== CONVERSATION ==="]
    recent = conv[-6:] if len(conv) > 6 else conv
    for msg in recent:
        role    = msg.get("role", "?").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")

    # Result of the last tool call (very important feedback for the agent)
    if tool_result is not None:
        lines += ["", "=== LAST TOOL RESULT ===", json.dumps(tool_result, indent=2)]

    # Any notes added to the ticket
    if ticket.get("notes"):
        lines += ["", "=== TICKET NOTES ==="]
        for note in ticket["notes"]:
            lines.append(f"  • {note}")

    # Error from last action — make it prominent so the agent self-corrects
    if info.get("error"):
        lines += [
            "",
            "=== ⚠️  ERROR FROM LAST ACTION ===",
            info["error"],
            f"You MUST only use tools from: {', '.join(avail)}",
        ]

    # Detect task type and context
    is_faq_task     = "get_order_details" not in avail
    already_replied = any(m.get("role") == "agent" for m in conv)
    sentiment       = ticket.get("sentiment", "")
    is_angry        = sentiment in ("angry", "very_angry") or any(
        w in ticket.get("subject", "").upper()
        for w in ["DEMAND", "URGENT", "FRAUD", "SCAM", "BROKEN", "ANGRY", "LEGAL",
                  "THREATENING", "OUTRAGEOUS", "ABSOLUTE"]
    )

    # Premium customer escalation: fires AFTER refund is issued for a premium customer.
    # This overrides the generic close_ticket hint so the agent escalates before closing.
    refund_done      = ticket.get("refund_issued", 0.0) > 0
    premium_tier     = ticket.get("customer_info", {}).get("tier", "regular") == "premium"
    needs_escalate   = (
        "escalate_to_human" in avail
        and already_replied
        and premium_tier
        and refund_done
    )

    # Build closing hint — priority order matters
    if is_faq_task and not already_replied:
        closing_hint = (
            "FAQ TICKET — No order lookup available. "
            "Your FIRST action MUST be search_kb with the customer's issue as the query. "
            "Do NOT ask for an order ID. Do NOT call any other tool first."
        )
    elif is_faq_task and already_replied:
        closing_hint = (
            "FAQ TICKET — DONE: You have already replied to the customer. "
            "There is nothing more to investigate. Call close_ticket RIGHT NOW. "
            "Do NOT ask questions. Do NOT look up orders. Just call close_ticket."
        )
    elif needs_escalate:
        closing_hint = (
            "⚠️  PREMIUM CUSTOMER — refund processed. "
            "Your NEXT action MUST be escalate_to_human (NOT close_ticket). "
            "Reason: premium customers require senior support follow-up after resolution. "
            "Use reason: 'Premium customer with unresolved issue requires senior support attention.'"
        )
    elif already_replied and "close_ticket" in avail:
        closing_hint = (
            "You have already replied to the customer. "
            "If the issue is fully resolved (order status communicated, refund issued, or question answered), "
            "call close_ticket. "
            "If there is still more to investigate (order not yet checked, status not communicated), "
            "continue investigating first — then close_ticket as your final action."
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

    # 1. Strip markdown code fences  ```json ... ``` or ``` ... ```
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
    print(f"  [WARN] Could not parse model output into JSON. Using fallback.")
    print(f"  [RAW ] {raw[:300]}")
    return {
        "tool": "send_reply",
        "params": {
            "message": "Thank you for reaching out. I'm reviewing your case carefully and will assist you shortly.",
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

    # Reset environment
    obs = call_reset(task_id=task_id, seed=seed)

    if verbose:
        t = obs.get("ticket", {})
        print(f"  Ticket  : [{t.get('id')}] {t.get('subject')}")
        print(f"  Customer: {t.get('customer_info', {}).get('name')}")
        print()

    messages    = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0
    last_tool   = None
    loop_count  = 0

    for step_num in range(1, 30):   # hard cap at 29 steps

        # Build user message from current observation
        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        # Call Groq
        try:
            raw = rotator.chat(
                messages    = messages,
                model       = MODEL_NAME,
                max_tokens  = 500,
                temperature = 0.1,   # low = consistent JSON output
            )
            messages.append({"role": "assistant", "content": raw})
        except Exception as e:
            print(f"  [ERROR] Groq call failed at step {step_num}: {e}")
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
                "params": {"final_message": "Your case has been reviewed and appropriate action has been taken. Thank you for your patience."}
            }
            loop_count = 0

        # Print step summary
        if verbose:
            params_str = json.dumps(action.get("params", {}))
            if len(params_str) > 85:
                params_str = params_str[:82] + "..."
            print(f"  Step {step_num:02d} | {action.get('tool'):<22} | {params_str}")

        # Execute in environment
        try:
            obs = call_step(action)
        except Exception as e:
            print(f"  [ERROR] /step call failed: {e}")
            break

        reward = obs.get("reward", 0.0)
        done   = obs.get("done",   False)
        info   = obs.get("info",   {})
        final_score = reward

        # Print reward info
        if verbose:
            if "grader_breakdown" in info:
                print(f"         → FINAL SCORE : {reward:.4f} / 1.0")
                for k, v in info["grader_breakdown"].items():
                    if v != 0:
                        sign = "+" if v > 0 else ""
                        print(f"           {sign}{v:.2f}  {k}")
            else:
                print(f"         → partial reward: {reward:.4f}")

        if done:
            if verbose:
                print(f"\n  ✓ Episode finished in {step_num} step(s).")
                print(f"  Final score: {final_score:.4f} / 1.0")
            break

        # Small sleep — respects Groq rate limits on free tier
        time.sleep(0.5)

    else:
        if verbose:
            print(f"\n  ✗ Hit step cap (29). Last score: {final_score:.4f}")

    return final_score


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Customer Service OpenEnv — Groq-native baseline agent"
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

    for run_i in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n{'─'*60}")
            print(f"  RUN {run_i} / {args.runs}")
        for task_id in tasks:
            seed  = args.seed if args.seed is not None else run_i * 100
            score = run_episode(task_id=task_id, seed=seed, verbose=verbose)
            scores[task_id].append(score)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  (model: {MODEL_NAME})")
    print(f"{'='*60}")

    overall = []
    for task_id in tasks:
        avg   = sum(scores[task_id]) / len(scores[task_id])
        overall.append(avg)
        bar   = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        extra = f"   runs={[round(s, 4) for s in scores[task_id]]}" if len(scores[task_id]) > 1 else ""
        print(f"  {task_id:<8} [{bar}] {avg:.4f}{extra}")

    grand = sum(overall) / len(overall)
    print(f"\n  Grand average : {grand:.4f} / 1.0")
    print(f"{'='*60}\n")

    sys.exit(0 if grand >= 0.5 else 1)


if __name__ == "__main__":
    main()

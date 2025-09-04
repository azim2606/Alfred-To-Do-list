
#!/usr/bin/env python3
from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory, make_response
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import os, re, json, time

# Natural-language date parsing
from dateparser.search import search_dates
import dateparser

# Timezone
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(os.environ.get("APP_TIMEZONE", "Asia/Brunei"))
except Exception:
    TZ = None

# Load .env
from dotenv import load_dotenv
load_dotenv()

# DB repo (PostgreSQL via SQLAlchemy)
from db import TaskRepo, Task, init_db

# ----------------------------------------------------------------------------
# Init DB + Repo
# ----------------------------------------------------------------------------
init_db()
store = TaskRepo()

# ----------------------------------------------------------------------------
# Date helpers
# ----------------------------------------------------------------------------
def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def human_today() -> str:
    now = datetime.now(TZ) if TZ else datetime.now()
    dnum = now.day
    return f"{now.strftime('%A')}, {now.strftime('%B')} {_ordinal(dnum)}, {now.strftime('%Y')}"

def parse_due_phrase(text: str) -> Tuple[str, Optional[float]]:
    settings = {
        "TIMEZONE": os.environ.get("APP_TIMEZONE", "Asia/Brunei"),
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": datetime.now(TZ) if TZ else datetime.now(),
    }
    matches = search_dates(text, settings=settings) or []
    due_dt, chunk = None, None
    now = datetime.now(TZ) if TZ else datetime.now()
    for phrase, dt in matches:
        if isinstance(dt, datetime) and (dt >= now or dt.date() == now.date()):
            due_dt, chunk = dt, phrase
            break
    if not due_dt:
        return text, None
    cleaned = re.sub(re.escape(chunk), "", text, count=1, flags=re.IGNORECASE).strip()
    return (cleaned or text), float(due_dt.timestamp())

def iso_or_epoch_to_ts(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        try:
            dt = dateparser.parse(val)
            return dt.timestamp() if dt else None
        except Exception:
            return None

def to_dict(t: Task) -> dict:
    return {
        "id": t.id, "description": t.description, "completed": t.completed,
        "created_at": t.created_at, "due_at": t.due_at, "priority": t.priority, "order": t.order
    }

# ----------------------------------------------------------------------------
# Rules fallback — history-aware for grocery “yes”
# ----------------------------------------------------------------------------
ORDINALS_WORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}
NUM_PATTERNS = [
    re.compile(r'#\s*(\d+)'),
    re.compile(r'\bid\s*(\d+)\b'),
    re.compile(r'\btask\s*(\d+)\b'),
    re.compile(r'\bitem\s*(\d+)\b'),
    re.compile(r'\bnumber\s*(\d+)\b'),
    re.compile(r'\b(\d+)\b'),
]
QUOTE_PATTERNS = [
    re.compile(r'"([^"]+)"'),
    re.compile(r"'([^']+)'"),
    re.compile(r'‘([^’]+)’'),
    re.compile(r'“([^”]+)”'),
]

def _extract_id(low: str) -> Optional[int]:
    for pat in NUM_PATTERNS:
        m = pat.search(low)
        if m:
            try: return int(m.group(1))
            except: continue
    for word, num in ORDINALS_WORD.items():
        if re.search(rf'\b{word}\b', low):
            return num
    return None

def _extract_quotes(text: str) -> Optional[str]:
    for pat in QUOTE_PATTERNS:
        m = pat.search(text)
        if m: return m.group(1).strip()
    return None

def _extract_description(text: str) -> str:
    q = _extract_quotes(text)
    if q: return q
    low = text.lower()
    prefixes = ["please ", "pls ", "kindly ", "can you ", "could you ", "would you ",
                "add ", "create ", "remember ", "note ", "put ", "delete ", "remove ", "complete "]
    cleaned = low
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p):]
    cleaned = re.sub(r'\s+to (my|the) list$', '', cleaned).strip()
    cleaned = re.sub(r'^(task|todo)\s*[:\-]\s*', '', cleaned).strip()
    start_idx = text.lower().find(cleaned)
    return text[start_idx:start_idx+len(cleaned)].strip() if start_idx != -1 else cleaned

def _extract_ingredients_from_history(history: Optional[List[dict]]) -> List[str]:
    if not history: return []
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        text = (m.get("content") or "")
        if "Ingredients:" not in text:
            continue
        part = text.split("Ingredients:", 1)[1]
        lines = part.splitlines()
        items = []
        for line in lines:
            if not line.strip():
                if items: break
                else: continue
            if line.strip().lower().startswith("steps:"):
                break
            m_bullet = re.match(r"\s*[-•]\s*(.+)", line)
            if m_bullet:
                items.append(m_bullet.group(1).strip())
            elif items:
                break
        return [i for i in items if i]
    return []

class RuleBrain:
    def parse(self, text: str, history: Optional[List[dict]] = None) -> Union[dict, list, None]:
        t = text.strip()
        if not t: raise ValueError("Empty message")
        low = t.lower()

        if low in {"yes","yes please","sure","okay","ok","please do","do it","add them","add it","add those","go ahead"}:
            items = _extract_ingredients_from_history(history)
            if items:
                return {"function":"addTasks","parameters":{"items": items}}
            return {"chat": "Certainly. Which items shall I add?"}

        if re.search(r'\b(hi|hello|hey|good (morning|afternoon|evening))\b', low) or "how are you" in low:
            return {"chat": "Good day. How may I assist with your tasks?"}
        if "recipe" in low:
            return {"chat":
                "Here is a simple Omelette recipe:\n\n"
                "Ingredients:\n- 2 eggs\n- Salt & pepper\n- 1 tbsp butter\n- Optional: cheese, herbs\n\n"
                "Steps:\n1) Beat eggs with salt & pepper.\n2) Melt butter, pour eggs, cook until just set.\n"
                "3) Add fillings, fold, and serve.\n\nShall I add these ingredients as a grocery list?"
            }

        if "date" in low or re.search(r"\bwhat('s| is) the date\b", low):
            return {"function":"getCurrentDate","parameters":{}}

        if "complete all" in low or "finish all" in low or "check all" in low:
            return {"function":"completeAllTasks","parameters":{}}
        if "delete all" in low or "clear all" in low or "remove all" in low:
            return {"function":"deleteAllTasks","parameters":{}}

        if any(k in low for k in ["show", "list", "view", "what do i have", "tasks", "todo"]):
            if any(k in low for k in ["add ", "create ", "new "]): pass
            else: return {"function":"viewTasks","parameters":{}}

        if any(k in low for k in ["complete", "done", "finish", "mark as done", "check off", "tick"]):
            tid = _extract_id(low)
            if tid: return {"function":"completeTask","parameters":{"task_id":tid}}
            desc = _extract_description(t)
            return {"function":"completeTask","parameters":{"description":desc}}

        if any(k in low for k in ["delete", "remove", "trash", "get rid of", "discard", "clear task"]):
            tid = _extract_id(low)
            if tid: return {"function":"deleteTask","parameters":{"task_id":tid}}
            desc = _extract_description(t)
            return {"function":"deleteTask","parameters":{"description":desc}}

        if low.startswith("add ") and " and " in low:
            parts = [p.strip() for p in re.split(r"\band\b", _extract_description(t)) if p.strip()]
            out = []
            for p in parts:
                cleaned, due_ts = parse_due_phrase(p)
                due_str = None
                if due_ts:
                    dt = datetime.fromtimestamp(due_ts, TZ) if TZ else datetime.fromtimestamp(due_ts)
                    due_str = dt.strftime("%B %d, %Y")
                out.append({"function":"addTask","parameters":{"description":cleaned, "due_date": due_str}})
            return out if out else None

        if any(low.startswith(k) for k in ["add","create","new ","note ","remember ","put "]) or \
           "please add" in low or "remember to" in low or _extract_quotes(t):
            desc = _extract_description(t)
            cleaned, due_ts = parse_due_phrase(desc)
            due_str = None
            if due_ts:
                dt = datetime.fromtimestamp(due_ts, TZ) if TZ else datetime.fromtimestamp(due_ts)
                due_str = dt.strftime("%B %d, %Y")
            return {"function":"addTask","parameters":{"description": cleaned, "due_date": due_str}}

        if len(t.split()) <= 3 and not re.search(r'\b(delete|remove|done|complete|show|list|view)\b', low):
            cleaned, due_ts = parse_due_phrase(t)
            due_str = None
            if due_ts:
                dt = datetime.fromtimestamp(due_ts, TZ) if TZ else datetime.fromtimestamp(due_ts)
                due_str = dt.strftime("%B %d, %Y")
            return {"function":"addTask","parameters":{"description": cleaned, "due_date": due_str}}

        return {"function":"viewTasks","parameters":{}}

# ----------------------------------------------------------------------------
# LLM brain (optional) — unchanged behavior; still may reply natural text
# ----------------------------------------------------------------------------
class LLMBrain:
    def __init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        self.system = (
            "You are Alfred, a discreet, efficient, polite to-do assistant.\n"
            "\n"
            "WHEN to use tools vs chat:\n"
            "• If the user is casually chatting or asking general knowledge (e.g., a recipe), reply in natural text.\n"
            "• If the user gives task commands, respond with JSON tool calls ONLY.\n"
            "• If user says today/tomorrow/yesterday, first call getCurrentDate() then addTask with a concrete date.\n"
            "\n"
            "TOOLS (JSON when calling):\n"
            "addTask(description: string, due_date: string | null)\n"
            "addTasks(items: string[])\n"
            "completeTask(task_id: int | description: string)\n"
            "completeAllTasks()\n"
            "deleteTask(task_id: int | description: string)\n"
            "deleteAllTasks()\n"
            "viewTasks()\n"
            "getCurrentDate()\n"
            "\n"
            "RULES:\n"
            "1) Return a single object or an array for multiple actions.\n"
            "2) Prefer task_id when user references numbers like 'task 2'.\n"
            "3) Handle multiple adds ('add dinner and gaming').\n"
            "4) Generate creative tasks if asked for random tasks.\n"
            "5) When calling a tool, output JSON ONLY (no extra text).\n"
            "\n"
            "RECIPES:\n"
            "Provide Ingredients + Steps in natural text, then ask if I should add ingredients.\n"
        )

    def parse(self, user_text: str, history: Optional[List[dict]] = None) -> Union[dict, list, None]:
        messages = [{"role": "system", "content": self.system}]
        if history:
            for m in history:
                role = m.get("role")
                content = m.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()

        if content.startswith("[") and content.endswith("]"):
            try:
                data = json.loads(content)
                if isinstance(data, list) and all(isinstance(x, dict) and "function" in x and "parameters" in x for x in data):
                    return data
            except Exception:
                pass
        if content.startswith("{") and content.endswith("}"):
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "function" in data and "parameters" in data:
                    return data
            except Exception:
                pass
        return {"chat": content}

# ----------------------------------------------------------------------------
# Flask
# ----------------------------------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')

def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

rule_brain = RuleBrain()
llm_brain = None
if llm_available():
    try:
        llm_brain = LLMBrain()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("LLM init failed:", e)
        llm_brain = None

# Affirmative safety net for LLM mode
AFFIRM = {"yes","yes please","yep","yeah","sure","okay","ok","please do","do it","add them","add it","add those","go ahead"}
def _is_affirmative(text: str) -> bool:
    return (text or "").strip().lower() in AFFIRM

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.get('/api/mode')
def get_mode():
    mode = "llm" if (llm_brain is not None) else "rules"
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo") if mode == "llm" else None
    resp = make_response(jsonify({"mode": mode, "model": model}))
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get('/today')
def get_today():
    resp = make_response(jsonify({"today": human_today()}))
    resp.headers['Cache-Control'] = 'no-store'
    return resp, 200

@app.post('/add')
def add_task_simple():
    data = request.get_json(silent=True) or {}
    desc = (data.get('description') or '').strip()
    if not desc:
        return jsonify({'error':'description is required'}), 400
    due_ts = iso_or_epoch_to_ts(data.get('due_date'))
    t = store.add(desc, due_at=due_ts)
    return jsonify(to_dict(t)), 201

# Tasks API
@app.post('/api/tasks')
def create_task():
    data = request.get_json(silent=True) or {}
    desc = (data.get('description') or '').strip()
    if not desc:
        return jsonify({'error':'description is required'}), 400

    prio = data.get("priority", 2)
    try: prio = int(prio)
    except: prio = 2

    due_ts = None
    cleaned_desc = desc

    if data.get("due"):
        cleaned_desc, due_ts = parse_due_phrase(str(data["due"]))
        if not cleaned_desc:
            cleaned_desc = desc
    elif data.get("due_at"):
        due_ts = iso_or_epoch_to_ts(str(data["due_at"]))
    elif data.get("due_date"):
        due_ts = iso_or_epoch_to_ts(str(data["due_date"]))

    t = store.add(cleaned_desc, due_at=due_ts, priority=prio)
    return jsonify(to_dict(t)), 201

@app.get('/api/tasks')
def list_tasks():
    tasks = [to_dict(t) for t in store.all()]
    resp = make_response(jsonify(tasks))
    resp.headers['Cache-Control'] = 'no-store'
    return resp, 200

@app.patch('/api/tasks/<int:task_id>/complete')
def complete_task(task_id: int):
    t = store.complete(task_id)
    if not t:
        return jsonify({'error': f'task {task_id} not found'}), 404
    return jsonify(to_dict(t)), 200

@app.patch('/api/tasks/<int:task_id>')
def update_task(task_id: int):
    data = request.get_json(silent=True) or {}
    desc = data.get('description')
    due_at = None
    if 'due' in data and data['due'] is not None:
        _, due_at = parse_due_phrase(str(data['due']))
    elif 'due_at' in data and data['due_at'] is not None:
        due_at = iso_or_epoch_to_ts(str(data['due_at']))
    elif 'due_date' in data and data['due_date'] is not None:
        due_at = iso_or_epoch_to_ts(str(data['due_date']))
    prio = data.get('priority')
    t = store.update(task_id, description=desc,
                     due_at=(due_at if ('due' in data or 'due_at' in data or 'due_date' in data) else None),
                     priority=prio)
    if not t:
        return jsonify({'error': f'task {task_id} not found'}), 404
    return jsonify(to_dict(t)), 200

@app.post('/api/tasks/reorder')
def reorder_tasks():
    data = request.get_json(silent=True) or {}
    order = data.get('order')
    if not isinstance(order, list) or not all(isinstance(x, int) for x in order):
        return jsonify({'error': 'order must be a list of task IDs'}), 400
    store.reorder(order)
    return jsonify({'ok': True}), 200

@app.post('/api/tasks/complete_all')
def complete_all_tasks():
    n = store.complete_all()
    return jsonify({'completed_all': n}), 200

@app.delete('/api/tasks/delete_all')
def delete_all_tasks():
    n = store.delete_all()
    return jsonify({'deleted_all': n}), 200

@app.delete('/api/tasks/<int:task_id>')
def delete_task(task_id: int):
    ok = store.delete(task_id)
    if not ok:
        return jsonify({'error': f'task {task_id} not found'}), 404
    return ('', 204)

def _summarize(calls: List[dict], results: List[dict]) -> str:
    added = sum((r['result'].get('added', 0) if isinstance(r['result'], dict) else 0)
                for r in results if r['function']=='addTasks')
    added += sum(1 for r in results if r['function']=='addTask' and r['status']==201)
    comp_all = next((r['result'].get('completed_all') for r in results if r['function']=='completeAllTasks' and isinstance(r['result'], dict)), None)
    del_all = next((r['result'].get('deleted_all') for r in results if r['function']=='deleteAllTasks' and isinstance(r['result'], dict)), None)
    comp_some = sum((r['result'].get('completed', 0) for r in results if r['function']=='completeTask' and isinstance(r['result'], dict)))
    del_some = sum((1 for r in results if r['function']=='deleteTask' and isinstance(r['result'], dict) and r['result'].get('deleted')))
    del_count = sum((r['result'].get('deleted_count', 0) for r in results if r['function']=='deleteTask' and isinstance(r['result'], dict)))
    parts=[]
    if added: parts.append(f"Added {added} task(s).")
    if comp_all is not None: parts.append(f"Completed {comp_all} task(s).")
    if comp_some: parts.append(f"Completed {comp_some} task(s).")
    if del_all is not None: parts.append(f"Deleted {del_all} task(s).")
    if del_some: parts.append(f"Deleted {del_some} task(s).")
    if del_count: parts.append(f"Deleted {del_count} task(s).")
    return " ".join(parts) or "Done."

@app.post('/api/ai')
def ai_route():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    execute = bool(data.get('execute', True))
    history = data.get('history')  # [{role:'user'|'assistant', content:str}, ...]

    if not message:
        return jsonify({'error': 'message is required'}), 400

    try:
        if llm_brain is not None:
            decision = llm_brain.parse(message, history=history)
            brain_used = "llm"
        else:
            decision = rule_brain.parse(message, history=history)
            brain_used = "rules"
    except Exception as e:
        return jsonify({'error': str(e), 'mode': 'llm' if llm_brain else 'rules'}), 400

    # LLM "yes" safety net
    if brain_used == "llm" and isinstance(decision, dict) and "chat" in decision and _is_affirmative(message):
        items = _extract_ingredients_from_history(history or [])
        if items:
            decision = {"function": "addTasks", "parameters": {"items": items}}

    if isinstance(decision, dict) and "chat" in decision:
        return jsonify({'mode': brain_used,'decision': decision,'executed': False,'result': None,'reply': decision["chat"]}), 200

    calls = None
    if isinstance(decision, list):
        calls = decision
    elif isinstance(decision, dict) and "function" in decision and "parameters" in decision:
        calls = [decision]

    if not calls:
        return jsonify({'mode': brain_used,'decision': decision,'executed': False,'result': None,'reply': "At your service. What shall I add?"}), 200

    results = []
    status = 200
    if execute:
        for call in calls:
            fn = call.get("function")
            params = call.get("parameters", {}) or {}
            one_result, one_status = None, 200

            if fn == 'getCurrentDate':
                one_result = {"today": human_today()}

            elif fn == 'addTask':
                desc = (params.get('description') or '').strip()
                if not desc:
                    return jsonify({'error': 'description is required', 'mode': brain_used}), 400
                due_ts = iso_or_epoch_to_ts(params.get('due_date'))
                if due_ts is None:
                    desc, due_ts = parse_due_phrase(desc)
                t = store.add(desc, due_at=due_ts)
                one_result, one_status = to_dict(t), 201

            elif fn == 'addTasks':
                items = params.get('items')
                if not isinstance(items, list) or not all(isinstance(x, str) and x.strip() for x in items):
                    return jsonify({'error': 'items must be a non-empty array of strings', 'mode': brain_used}), 400
                ids = []
                for raw in items:
                    t = store.add(raw.strip(), due_at=None)
                    ids.append(t.id)
                one_result, one_status = {'added': len(ids), 'ids': ids}, 201

            elif fn == 'viewTasks':
                one_result = [to_dict(t) for t in store.all()]

            elif fn == 'completeTask':
                tid = params.get('task_id')
                desc = params.get('description')
                if isinstance(tid, int) and tid > 0:
                    t = store.complete(tid)
                    if not t:
                        return jsonify({'error': f'task {tid} not found', 'mode': brain_used}), 404
                    one_result = to_dict(t)
                elif isinstance(desc, str) and desc.strip():
                    n = store.complete_by_description(desc)
                    one_result = {'completed': n}
                else:
                    return jsonify({'error':'provide task_id or description', 'mode': brain_used}), 400

            elif fn == 'completeAllTasks':
                n = store.complete_all()
                one_result = {'completed_all': n}

            elif fn == 'deleteTask':
                tid = params.get('task_id')
                desc = params.get('description')
                if isinstance(tid, int) and tid > 0:
                    ok = store.delete(tid)
                    if not ok:
                        return jsonify({'error': f'task {tid} not found', 'mode': brain_used}), 404
                    one_result = {'deleted': tid}
                elif isinstance(desc, str) and desc.strip():
                    n = store.delete_by_description(desc)
                    one_result = {'deleted_count': n}
                else:
                    return jsonify({'error':'provide task_id or description', 'mode': brain_used}), 400

            elif fn == 'deleteAllTasks':
                n = store.delete_all()
                one_result = {'deleted_all': n}

            else:
                one_result, one_status = {'error': f'unknown function {fn}'}, 400

            results.append({'function': fn, 'result': one_result, 'status': one_status})
            status = max(status, one_status)

    reply = _summarize(calls, results)
    return jsonify({'mode': brain_used,'decision': decision,'executed': bool(execute and calls),'result': results if calls else None,'reply': reply}), status

# Entrypoint
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

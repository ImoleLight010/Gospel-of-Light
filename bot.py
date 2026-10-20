import json
import os
import re
import sys
import datetime as dt
from zoneinfo import ZoneInfo
from urllib.parse import quote

import requests

API_KEY = os.environ["GEMINI_API_KEY"]
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]  # channel id; several allowed, separated by commas
HISTORY_FILE = "history.json"
BASE = "https://generativelanguage.googleapis.com/v1beta"
WAT = ZoneInfo("Africa/Lagos")

TYPES = [
    "Verse of the hour: one powerful verse with a short, clear reflection.",
    "Biblical theology: explain ONE core doctrine (for example grace, justification, atonement, the Trinity, sanctification, providence, covenant, the kingdom of God, resurrection) in simple language, anchored in the verse.",
    "Bible character spotlight: a person from the Bible and one life lesson, anchored in a verse about them.",
    "Short prayer: a written prayer inspired by the verse that readers can pray along with.",
    "Names and character of God: one attribute or name of God and what it means for believers today.",
    "Promise of God: a promise in Scripture and how to hold on to it in hard times.",
    "Teaching of Jesus: a parable, miracle or saying of Jesus explained and applied.",
    "Bible background: helpful historical or cultural context, or a short overview of one book of the Bible, that deepens understanding.",
    "Proverbs and wisdom for daily life: work, money, relationships, speech or character.",
    "Psalm of encouragement for anxiety, joy, thanksgiving or worship.",
    "Christian living: faith, love, forgiveness, patience, integrity, family or purpose, grounded in the verse.",
    "Gospel focus: the good news of Jesus (grace, salvation, hope) shared warmly and clearly.",
    "Bible study tip: a practical way to read, study, memorize or meditate on Scripture, using the verse as the example.",
    "Faith and endurance: encouragement for hardship, waiting, doubt or spiritual growth.",
]

SYSTEM = (
    "You write posts for a Christian Telegram channel about the Bible, biblical theology and "
    "Christian life, for readers worldwide including many in Nigeria. "
    "Theology: orthodox, historic Christianity centered on Jesus Christ and the authority of "
    "Scripture. When a topic is debated among sincere Christians, say so fairly instead of "
    "pushing one side. Respectful and warm, never harsh or political. "
    "ACCURACY IS CRITICAL: never invent verses, references, quotes, Hebrew or Greek words, or "
    "historical claims. If unsure of a detail, leave it out. "
    "OUTPUT FORMAT, exactly: the first line is 'REF: ' followed by ONE Bible passage of at most "
    "4 verses using standard English book names (example: REF: Romans 8:28). Then a blank line. "
    "Then the post, 60-140 words, which must contain the token [[VERSE]] alone on its own line "
    "at the place where the scripture text should appear. Do NOT type the verse text yourself; "
    "the system inserts the exact King James text. Plain text only, no markdown symbols like "
    "** or #. Use a few fitting emojis. You may end with one short line of encouragement or an "
    "invitation to share the post. Never reuse a passage or angle from the recent list. "
    "Output nothing except the REF line and the post."
)


def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(h):
    with open(HISTORY_FILE, "w") as f:
        json.dump(h[-150:], f, ensure_ascii=False, indent=1)


def version(name):
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def candidate_models():
    """Ask Google which models exist right now, so a retired model name never breaks the bot."""
    if os.environ.get("MODEL"):
        return [os.environ["MODEL"]]
    try:
        r = requests.get(
            f"{BASE}/models?pageSize=200", headers={"x-goog-api-key": API_KEY}, timeout=20
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Model list problem:", type(e).__name__)
        return ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    names = [
        m["name"].split("/")[-1]
        for m in r.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    flash = sorted(
        (n for n in names if re.fullmatch(r"gemini-[\d.]+-flash", n)), key=version, reverse=True
    )
    lite = sorted(
        (n for n in names if re.fullmatch(r"gemini-[\d.]+-flash-lite", n)),
        key=version,
        reverse=True,
    )
    others = [n for n in names if "flash" in n and n not in flash and n not in lite]
    result = flash + lite + others
    if not result:
        raise RuntimeError(f"No usable Gemini models found. Models seen: {names[:20]}")
    return result


def parse(raw):
    m = re.match(r"\s*REF:\s*(.+?)\s*\n", raw)
    if not m:
        raise ValueError("Reply did not start with a REF line")
    ref = m.group(1).strip().rstrip(".")
    body = raw[m.end():].strip()
    if len(body) < 40:
        raise ValueError("Post body too short")
    return ref, body


def ask_ai(prompt):
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 2500},
    }
    last_error = ""
    for model in candidate_models()[:6]:
        try:
            r = requests.post(
                f"{BASE}/models/{model}:generateContent",
                headers={"x-goog-api-key": API_KEY, "content-type": "application/json"},
                json=body,
                timeout=40,
            )
        except requests.exceptions.RequestException as e:
            last_error = f"{model}: {type(e).__name__}"
            print("Skipping", last_error)
            continue
        if r.status_code in (403, 404, 429, 500, 503):
            last_error = f"{model}: {r.status_code} {r.text[:300]}"
            print("Skipping", last_error)
            continue
        if not r.ok:
            raise RuntimeError(f"{model}: {r.status_code} {r.text[:500]}")
        candidates = r.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(p.get("text", "") for p in parts).strip()
        try:
            ref, post = parse(text)
        except ValueError as e:
            last_error = f"{model}: {e}"
            print("Skipping", last_error)
            continue
        print("Used model:", model)
        return ref, post
    raise RuntimeError(f"All models failed. Last error: {last_error}")


def fetch_verse(ref):
    """Get the exact King James text (public domain) so verses are never made up."""
    for _ in range(2):
        try:
            r = requests.get(
                "https://bible-api.com/" + quote(ref), params={"translation": "kjv"}, timeout=30
            )
            if r.ok:
                j = r.json()
                text = " ".join(j.get("text", "").split())
                if text:
                    return j.get("reference", ref).strip(), text
        except Exception as e:  # network hiccup: try again, then fall back
            print("Verse fetch problem:", e)
    return None


def compose(ref, post):
    found = fetch_verse(ref)
    if found:
        shown_ref, text = found
        block = f"\u201c{text}\u201d\n\u2014 {shown_ref} (KJV)"
    else:
        block = f"\U0001F4D6 {ref}"
    if "[[VERSE]]" in post:
        return post.replace("[[VERSE]]", block)
    return block + "\n\n" + post


def send(text, silent):
    for chat_id in CHAT_ID.split(","):
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id.strip(), "text": text[:4096], "disable_notification": silent},
            timeout=30,
        )
        if not r.ok:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:300]}")


def main():
    now = dt.datetime.now(WAT)
    history = load_history()
    kind = TYPES[(now.hour + now.timetuple().tm_yday * 5) % len(TYPES)]
    recent_refs = [h["ref"] for h in history if h.get("ref")][-80:]
    recent_posts = [h["text"][:160].replace("\n", " ") for h in history][-10:]
    prompt = (
        f"Date: {now:%A %d %B %Y}, {now:%H}:00 Nigerian time.\n\n"
        f"Post type: {kind}\n\n"
        f"Passages already used recently (do NOT reuse): {', '.join(recent_refs) or 'none yet'}\n\n"
        f"Openings of recent posts (do not repeat their angle):\n"
        + ("\n".join(f"- {p}" for p in recent_posts) or "- none yet")
    )
    ref, post = ask_ai(prompt)
    text = compose(ref, post)
    silent = now.hour >= 22 or now.hour < 6  # quiet notifications overnight
    send(text, silent)
    history.append(
        {"date": now.isoformat(timespec="minutes"), "type": kind[:30], "ref": ref, "text": text}
    )
    save_history(history)
    print("Posted:", ref)


if __name__ == "__main__":
    main()

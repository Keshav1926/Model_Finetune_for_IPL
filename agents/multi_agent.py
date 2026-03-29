#!/usr/bin/env python
# multi_agent.py — minimal multi-agent pipeline with CLI query input

import os, json, requests, argparse
from typing import Any, Optional
from openai import OpenAI

# CONFIG (env)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta/llama-3.3-70b-instruct")
SLM_URL = os.getenv("SLM_URL", "http://0.0.0.0:1926/infer")

def llama_call(system: str, user: str, max_tokens: int = 10000, temp: float = 0.2) -> str:
    assert NVIDIA_API_KEY, "Set NVIDIA_API_KEY env var"
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    resp = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=temp, top_p=0.7, max_tokens=max_tokens, stream=False
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        out = []
        for chunk in resp:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                out.append(delta.content)
        return "".join(out).strip()

class CommentaryTool:
    """
    Tool1: send the whole commentary JSON to the LLM when feasible.
    Returns a very short context. If JSON is huge, send a safe truncated
    representation and ask Llama to request more (it will return 'ASK_FOR_MORE').
    """
    def __init__(self, max_chars: int = 30000):
        # max_chars: approximate char limit to send whole JSON (tune to your LLM/token-limit)
        self.max_chars = max_chars

    def _serialize(self, commentary: Any) -> str:
        # pretty compact JSON string (no extra spaces)
        try:
            return json.dumps(commentary, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            # fallback to str()
            return str(commentary)

    def run(self, commentary: Any, inning: Optional[int] = None, max_words: int = 30000) -> str:
        serialized = self._serialize(commentary)

        if len(serialized) <= self.max_chars:
            # Safe to send whole JSON
            system = "You are succinct. Parse the JSON input and extract the most important cricket facts."
            user = (
                f"I will give you the FULL match commentary JSON. Produce a single very short paragraph "
                f"(<= {max_words} words) containing the most important facts for an analyst: score, top bat, key wickets. "
                f"Return only the paragraph — do not add commentary.\n\n"
                f"JSON:\n```json\n{serialized}\n```"
            )
            out = llama_call(system, user, temp=0.05)
            return " ".join(out.split()[:])

        # If too large -> send top-level summary + prefix and let model ask for more if needed
        # Build top-level key summary with sizes to help Llama decide
        try:
            parsed = json.loads(serialized)
            if isinstance(parsed, dict):
                key_info = []
                for k, v in parsed.items():
                    s = json.dumps(v)
                    key_info.append(f"{k} (len={len(s)})")
                key_summary = "; ".join(key_info[:40])
            else:
                key_summary = f"Top-level type: {type(parsed).__name__}, length: {len(serialized)}"
        except Exception:
            key_summary = f"Truncated JSON length: {len(serialized)}"

        prefix = serialized[: self.max_chars]  # prefix chunk
        system = "You are concise. You will be given an excerpt and a top-level summary of a large JSON. Produce a very short context or respond ASK_FOR_MORE if you need more details."
        user = (
            f"Top-level keys/summary: {key_summary}\n\n"
            f"Here is the START of the JSON (truncated because the full file is large):\n```json\n{prefix}\n```\n\n"
            f"Task: Based on the available data, produce a single short paragraph (<= {max_words} words) with the most important facts. "
            f"If the excerpt is insufficient to produce an accurate summary, reply with exactly: ASK_FOR_MORE\n"
            f"Otherwise return the short paragraph only."
        )
        out = llama_call(system, user, temp=0.05)
        out = out.strip()
        if out.upper().startswith("ASK_FOR_MORE"):
            return "ASK_FOR_MORE"
        return " ".join(out.split()[:max_words])


class SLMTool:
    def run(self, context: str, question: str) -> str:
        prompt = f"System: You are a Cricket Analyst. Use the context provided to answere user query. \nContext:\n{context}\n\nUser: {question}\nAssistant:"
        try:
            r = requests.post(SLM_URL, json={"prompt": prompt}, timeout=500)
            r.raise_for_status()
            j = r.json()
            return j.get("text") if isinstance(j, dict) else str(j)
        except Exception as e:
            return f"[SLM error] {e}"

class DecisionTool:
    def run(self, commentary_excerpt: Any, question: str) -> bool:
        excerpt = (json.dumps(commentary_excerpt)[:800]) if commentary_excerpt else ""
        system = "Answer YES or NO whether the user's question requires match commentary to answer well."
        user = f"Question: {question}\nExcerpt:\n{excerpt}\nAnswer YES or NO and one short reason."
        resp = llama_call(system, user, temp=0.0).upper().strip()
        return resp.startswith("YES")

def run_pipeline(commentary: Any, question: str, inning: Optional[int] = None):
    dec = DecisionTool()
    need = dec.run(commentary, question)
    if need:
        ctx = CommentaryTool().run(commentary, inning=inning, max_words=30)
    else:
        ctx = ""
    ans = SLMTool().run(ctx, question)
    return {"need_context": need, "context": ctx, "answer": ans}

def load_commentary_from_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser(description="Minimal multi-agent cricket pipeline")
    p.add_argument("-q", "--query", help="User query (question) to ask the analyst")
    p.add_argument("-f", "--commentary-file", help="Path to commentary JSON file")
    p.add_argument("-c", "--commentary-json", help="Inline commentary JSON string")
    p.add_argument("--inning", type=int, help="Inning number to filter (optional)")
    args = p.parse_args()

    query = args.query
    if not query:
        try:
            query = input("Enter your query: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return
    # load commentary
    commentary = None
    if args.commentary_file:
        try:
            commentary = load_commentary_from_file(args.commentary_file)
        except Exception as e:
            print("Failed to load commentary file:", e)
            return
    elif args.commentary_json:
        try:
            commentary = json.loads(args.commentary_json)
        except Exception as e:
            print("Invalid commentary-json:", e)
            commentary = args.commentary_json
    else:
        # default: no commentary (empty dict), DecisionTool will still run
        commentary = {}

    out = run_pipeline(commentary, query, inning=args.inning)
    print("\n--- RESULT ---")
    print("Need context:", out["need_context"])
    print("Context:", out["context"])
    print("Answer:", out["answer"])

if __name__ == "__main__":
    main()



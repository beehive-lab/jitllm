"""Regenerates the expected prompts under chat-templates/<family>/ from each family's template.

Each <family>/template.jinja is the tokenizer.chat_template embedded in that family's GGUF, copied
verbatim. The scenarios in scenarios.json are rendered with Jinja2 under the Hugging Face
transformers settings (trim_blocks, lstrip_blocks, tojson = json.dumps with ensure_ascii=False,
add_generation_prompt=True), and each result is written to <family>/<scenario>.txt, which the Java
tests compare the engine's encoding against.

    python3 render_expected.py        # from this directory; needs jinja2
"""
import json
import os

from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))


def environment():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise ValueError(message)

    env.globals['raise_exception'] = raise_exception
    return env


def openai_tools(spec, names):
    return [{"type": "function", "function": {"name": n, "description": spec[n]["description"],
                                              "parameters": spec[n]["parameters"]}} for n in names]


def openai_messages(messages):
    out = []
    for m in messages:
        if m["role"] == "assistant" and "tool_calls" in m:
            out.append({"role": "assistant", "content": m.get("content", ""),
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}}
                                       for c in m["tool_calls"]]})
        else:
            out.append(dict(m))
    return out


def granite_3_2_messages(messages):
    """Granite 3.2's template renders an assistant message's content only, so a call is its
    content, in the model's own output syntax."""
    out = []
    for m in openai_messages(messages):
        if m["role"] == "assistant" and "tool_calls" in m:
            calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                     for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": "<|tool_call|>" + json.dumps(calls, ensure_ascii=False)})
        else:
            out.append(m)
    return out


FAMILIES = {
    "granite-3.2": dict(bos_token="", eos_token="<|end_of_text|>", messages=granite_3_2_messages),
    "granite-4.0": dict(bos_token="", eos_token="<|end_of_text|>", messages=openai_messages),
    "qwen3": dict(bos_token="", eos_token="<|im_end|>", messages=openai_messages),
    "qwen2.5": dict(bos_token="", eos_token="<|im_end|>", messages=openai_messages),
    # No strftime_now: the template falls back to its fixed date. It rejects more than one call in
    # an assistant turn, so that scenario is not rendered for it.
    "llama-3.2": dict(bos_token="<|begin_of_text|>", eos_token="<|eot_id|>", messages=openai_messages,
                      skip={"two_calls_results_answer_user"}),
}

if __name__ == "__main__":
    data = json.load(open(os.path.join(HERE, "scenarios.json")))
    for family, cfg in FAMILIES.items():
        template = environment().from_string(open(os.path.join(HERE, family, "template.jinja")).read())
        for name, scenario in data["scenarios"].items():
            if name in cfg.get("skip", ()):
                continue
            prompt = template.render(messages=cfg["messages"](scenario["messages"]),
                                     tools=openai_tools(data["tools"], scenario["tools"]),
                                     add_generation_prompt=True, bos_token=cfg["bos_token"],
                                     eos_token=cfg["eos_token"])
            with open(os.path.join(HERE, family, name + ".txt"), "w") as f:
                f.write(prompt)
            print(family, name, len(prompt))

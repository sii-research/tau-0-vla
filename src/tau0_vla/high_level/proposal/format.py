"""One prompt, camera-order and structured-output contract for all entry points."""
import re

CAMERA_ORDER = ("head", "left", "right")
FORMAT_TAGS = tuple(f"<|{prefix}{name}|>" for name in ("think", "memory", "subtask") for prefix in ("", "/"))


def task_type(value):
    if value == "subtask":
        return "subtask_only"
    if value not in ("full_qa", "subtask_only"):
        raise ValueError("task_type must be full_qa or subtask_only")
    return value


def prompt_text(instruction, memory="", mode="full_qa"):
    mode = task_type(mode)
    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    prefix = f"Task instruction: {instruction}\n"
    if mode == "full_qa":
        return prefix + f"Current memory: {memory.strip() or '(empty)'}\n" + (
            "Based on the current visual state, predict the next sub-action and update the memory."
        )
    return prefix + "Please predict the sub-action to execute next."


def ordered_images(images):
    """Accept named cameras; do not infer camera identity from array positions."""
    images = dict(images)
    for canonical, alias in (("left", "hand_left"), ("right", "hand_right")):
        if alias in images:
            if canonical in images:
                raise ValueError(f"Specify only one of {canonical} and {alias}")
            images[canonical] = images.pop(alias)
    if set(images) != set(CAMERA_ORDER) or not all(images.values()):
        raise ValueError("images must contain exactly head, left and right")
    return [images[name] for name in CAMERA_ORDER]


def messages(row, images):
    """Images are already decoded/resized in canonical order."""
    mode = task_type(row.get("task_type", "full_qa"))
    if row.get("question") is not None:
        # Reproduce a dataset's original prompt, including hierarchical task text.
        question = row["question"]
        if "<image>" in question:
            prefix = "<image>\n" * 3
            if not question.startswith(prefix) or question.count("<image>") != 3:
                raise ValueError("question image markers must be the three leading images")
            question = question[len(prefix):]
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
    else:
        question = prompt_text(row["instruction"], row.get("memory", ""), mode)
    return [{"role": "user", "content": [
        *({"type": "image", "image": image} for image in images),
        {"type": "text", "text": question},
    ]}]


def parse_tag(text, name):
    for closing in (f"<|/{name}|>", f"<|{name}|>"):
        match = re.search(re.escape(f"<|{name}|>") + "(.*?)" + re.escape(closing), text, re.S)
        if match:
            return match.group(1).strip()
    # Retain compatibility with Qwen3.5 outputs missing the custom think opener.
    if name == "think" and "<|think|>" not in text:
        match = re.match(r"^(.*?)<\|/think\|>", text, re.S)
        if match:
            return match.group(1).strip()
    return None


def parse_output(text, mode="full_qa"):
    mode = task_type(mode)
    think, memory, subtask = (parse_tag(text, name) for name in ("think", "memory", "subtask"))
    if mode == "full_qa":
        valid = all(v is not None for v in (think, memory, subtask)) and bool(subtask)
        memory = (memory or "(empty)") if valid else None
    else:
        subtask = subtask or text.strip()
        valid = bool(subtask) and not re.search(r"<\|/?(?:think|memory|subtask)\|>", subtask)
        think = memory = None
    return {"think": think, "memory_out": memory, "subtask": subtask, "format_valid": bool(valid)}


def update_memory(previous, parsed):
    if not parsed.get("format_valid") or parsed.get("memory_out") is None:
        return previous
    return parsed["memory_out"].strip() or "(empty)"


def decode(processor, ids):
    """Preserve business tags even when registered as special tokens."""
    text = processor.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    tokenizer = processor.tokenizer
    controls = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
    controls.update(getattr(tokenizer, key, None) for key in ("bos_token", "eos_token", "pad_token"))
    for token in controls - {None, "", *FORMAT_TAGS}:
        text = text.replace(token, "")
    return text.strip()


def answer_text(row):
    answer = row["answer"]
    if task_type(row.get("task_type", "full_qa")) == "subtask_only":
        result = answer if isinstance(answer, str) else answer["subtask"]
    elif isinstance(answer, str):
        result = answer
    else:
        result = "".join(f"<|{name}|>{answer[name]}<|/{name}|>" for name in ("think", "memory", "subtask"))
    if not parse_output(result, row.get("task_type", "full_qa"))["format_valid"]:
        raise ValueError(f"Invalid answer for {row.get('id')}")
    return result

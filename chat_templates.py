from pathlib import Path
from typing import Any


VICUNA_7B_V15_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = 'A chat between a curious user and an artificial intelligence assistant.\n"
    "The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 %}{{ system_message }}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ ' USER: ' + message['content'].strip() }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
)

GENERIC_CHAT_TEMPLATE = """{% for message in messages -%}
{{ '<|user|>' if message['role'] == 'user' else '<|assistant|>' }}: {{ message['content'] }}
{% endfor -%}
<|assistant|>:"""


def _is_vicuna_7b_v15_reference(value: Any) -> bool:
    normalized = str(value or "").rstrip("/").lower()
    if not normalized:
        return False
    name = Path(normalized).name
    return name == "vicuna-7b-v1.5" or normalized == "lmsys/vicuna-7b-v1.5"


def ensure_chat_template(tokenizer: Any, *references: Any) -> str | None:
    """Install an in-memory chat template when a tokenizer does not provide one."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return None
    if not hasattr(tokenizer, "apply_chat_template"):
        return None

    candidates = [*references, getattr(tokenizer, "name_or_path", "")]
    if any(_is_vicuna_7b_v15_reference(candidate) for candidate in candidates):
        tokenizer.chat_template = VICUNA_7B_V15_CHAT_TEMPLATE
        return "vicuna-7b-v1.5"

    tokenizer.chat_template = GENERIC_CHAT_TEMPLATE
    return "generic"


def chat_template_generation_kwargs(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if template is not None and "enable_thinking" in str(template):
        return {"enable_thinking": False}
    return {}

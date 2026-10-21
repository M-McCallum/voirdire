"""LLM judge: one chat-model call per criterion, rubric from prompts/judge_<criterion>.md.

The rubric file starts with a frontmatter block carrying `version:`. The prompt hash covers
the full system prompt (rubric plus the fixed answer-format instruction), so a change to
either re-pins the judge (docs/protocol.md step 8).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.bot.graph import _text
from core.judges.base import Fingerprint, Verdict

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Pinned judge configuration. The model id is the exact snapshot the scorecard records.
# Anthropic's 5.5 models reject non-default `temperature`, so it is None there; set 0.0
# for a model that accepts it (the 4.6 family). Either way the value lands on the scorecard.
DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class LLMJudgeConfig:
    model: str = DEFAULT_MODEL
    temperature: float | None = None
    max_tokens: int = 512


DEFAULT_CONFIG = LLMJudgeConfig()


ANSWER_FORMAT = (
    "\n\nAnswer with one JSON object and nothing else: "
    '{"verdict": true|false, "confidence": <0..1 or null>, "reason": "<one sentence>"}. '
    "verdict true means the bot did the right thing for this criterion."
)
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def load_rubric(criterion: str, prompts_dir: Path = PROMPTS_DIR) -> tuple[str, str]:
    """Return (rubric text, version) from prompts/judge_<criterion>.md."""
    raw = (prompts_dir / f"judge_{criterion}.md").read_text()
    m = _FRONTMATTER.match(raw)
    if not m:
        raise ValueError(f"judge_{criterion}.md has no frontmatter with a version")
    meta = dict(
        line.split(":", 1) for line in m.group(1).splitlines() if ":" in line
    )  # fmt: skip
    version = meta.get("version", "").strip()
    if not version:
        raise ValueError(f"judge_{criterion}.md frontmatter has no version")
    return raw[m.end() :].strip(), version


def parse_verdict(text: str) -> Verdict:
    m = _JSON.search(text)
    if not m:
        raise ValueError(f"judge returned no JSON object: {text[:200]!r}")
    data = json.loads(m.group(0))
    if not isinstance(data.get("verdict"), bool):
        raise ValueError(f"judge verdict is not a boolean: {data!r}")
    confidence = data.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
    return {"verdict": data["verdict"], "confidence": confidence}


def default_judge_model(config: LLMJudgeConfig) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    kwargs = {} if config.temperature is None else {"temperature": config.temperature}
    return ChatAnthropic(model=config.model, max_tokens=config.max_tokens, **kwargs)


class LLMJudge:
    name = "llm"

    def __init__(
        self,
        criterion: str,
        model: BaseChatModel | None = None,
        config: LLMJudgeConfig = DEFAULT_CONFIG,
        prompts_dir: Path = PROMPTS_DIR,
    ) -> None:
        self.criterion = criterion
        self.config = config
        rubric, self.rubric_version = load_rubric(criterion, prompts_dir)
        self.system_prompt = rubric + ANSWER_FORMAT
        self.prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:12]
        self._model = model or default_judge_model(config)

    def judge(self, input: str, output: str, retrieved: list[str]) -> Verdict:
        user = (
            f"<user_turn>\n{input}\n</user_turn>\n\n"
            f"<bot_reply>\n{output}\n</bot_reply>\n\n"
            f"<retrieved_doc_ids>{', '.join(retrieved) or 'none'}</retrieved_doc_ids>"
        )
        response = self._model.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=user)]
        )
        return parse_verdict(_text(response.content))

    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            model_snapshot=self.config.model,
            prompt_hash=self.prompt_hash,
            rubric_version=self.rubric_version,
            temperature=self.config.temperature,
        )

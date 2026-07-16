from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel


class PromptRenderer:
    def __init__(self, templates_dir: Path | None = None) -> None:
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "prompts"

        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    def render(self, template_name: str, model: BaseModel | dict[str, Any]) -> str:
        template = self.env.get_template(template_name)

        if isinstance(model, BaseModel):
            context = model.model_dump(mode="json")
        else:
            context = model

        return template.render(**context).strip()
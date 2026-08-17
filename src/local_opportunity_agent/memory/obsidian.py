from __future__ import annotations

import re
from pathlib import Path


class ObsidianError(RuntimeError):
    """Raised when an Obsidian operation fails."""


class ObsidianMemory:
    """Manage human-readable Markdown memory inside an Obsidian vault."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path

    def initialize(self) -> None:
        """Create the expected vault structure."""
        directories = (
            "opportunities",
            "research",
            "companies",
            "people",
        )

        for directory in directories:
            (self.vault_path / directory).mkdir(
                parents=True,
                exist_ok=True,
            )

    def write_note(
        self,
        *,
        category: str,
        note_id: str,
        title: str,
        content: str,
        frontmatter: dict[str, str | int | float | bool] | None = None,
    ) -> Path:
        """Create or replace a Markdown note."""
        self.initialize()

        safe_category = self._sanitize_path_component(category)
        safe_note_id = self._sanitize_path_component(note_id)

        category_path = self.vault_path / safe_category
        category_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        note_path = category_path / f"{safe_note_id}.md"

        markdown = self._render_markdown(
            title=title,
            content=content,
            frontmatter=frontmatter,
        )

        try:
            note_path.write_text(
                markdown,
                encoding="utf-8",
            )
        except OSError as error:
            raise ObsidianError(f"Unable to write Obsidian note '{note_path}': {error}") from error

        return note_path

    def read_note(
        self,
        *,
        category: str,
        note_id: str,
    ) -> str | None:
        """Read an existing Markdown note."""
        safe_category = self._sanitize_path_component(category)
        safe_note_id = self._sanitize_path_component(note_id)

        note_path = self.vault_path / safe_category / f"{safe_note_id}.md"

        if not note_path.exists():
            return None

        try:
            return note_path.read_text(
                encoding="utf-8",
            )
        except OSError as error:
            raise ObsidianError(f"Unable to read Obsidian note '{note_path}': {error}") from error

    def delete_note(
        self,
        *,
        category: str,
        note_id: str,
    ) -> bool:
        """Delete an existing Markdown note."""
        safe_category = self._sanitize_path_component(category)
        safe_note_id = self._sanitize_path_component(note_id)

        note_path = self.vault_path / safe_category / f"{safe_note_id}.md"

        if not note_path.exists():
            return False

        try:
            note_path.unlink()
        except OSError as error:
            raise ObsidianError(f"Unable to delete Obsidian note '{note_path}': {error}") from error

        return True

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        """Prevent unsafe path components."""
        sanitized = re.sub(
            r"[^a-zA-Z0-9._-]+",
            "-",
            value.strip(),
        ).strip("-.")

        if not sanitized:
            raise ValueError("Path component must contain at least one valid character")

        return sanitized

    @staticmethod
    def _render_markdown(
        *,
        title: str,
        content: str,
        frontmatter: dict[str, str | int | float | bool] | None,
    ) -> str:
        parts: list[str] = []

        if frontmatter:
            parts.append("---")

            for key, value in frontmatter.items():
                if isinstance(value, bool):
                    rendered_value = "true" if value else "false"
                else:
                    rendered_value = str(value)

                parts.append(f"{key}: {rendered_value}")

            parts.append("---")
            parts.append("")

        parts.append(f"# {title}")
        parts.append("")
        parts.append(content.strip())
        parts.append("")

        return "\n".join(parts)

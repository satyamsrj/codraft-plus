import re
import json
import difflib
from typing import Tuple, Dict
from codraft.types import PatchOperation, DraftState, Section

class ResilientPatcher:
    @classmethod
    def parse_patch(cls, raw_output: str, fallback_model: str = "unknown") -> PatchOperation:
        cleaned = raw_output.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "content" in data:
                return PatchOperation(
                    section_id=str(data.get("section_id", "body")),
                    operation=data.get("operation", "update"),
                    content=str(data.get("content", "")),
                    reason=str(data.get("reason", "Structured patch update")),
                    model=data.get("model", fallback_model),
                    confidence=float(data.get("confidence", 0.8))
                )
        except Exception:
            pass

        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            fence_content = fence_match.group(1).strip()
            try:
                data = json.loads(fence_content)
                if isinstance(data, dict) and "content" in data:
                    return PatchOperation(
                        section_id=str(data.get("section_id", "body")),
                        operation=data.get("operation", "update"),
                        content=str(data.get("content", "")),
                        reason=str(data.get("reason", "Fenced JSON patch")),
                        model=data.get("model", fallback_model),
                        confidence=float(data.get("confidence", 0.8))
                    )
            except Exception:
                pass

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            substr = cleaned[start_idx:end_idx + 1]
            try:
                data = json.loads(substr)
                if isinstance(data, dict) and "content" in data:
                    return PatchOperation(
                        section_id=str(data.get("section_id", "body")),
                        operation=data.get("operation", "update"),
                        content=str(data.get("content", "")),
                        reason=str(data.get("reason", "Extracted curly-brace JSON")),
                        model=data.get("model", fallback_model),
                        confidence=float(data.get("confidence", 0.75))
                    )
            except Exception:
                try:
                    sanitized = re.sub(r'(?<!\\)\n', r'\\n', substr)
                    data = json.loads(sanitized)
                    if isinstance(data, dict) and "content" in data:
                        return PatchOperation(
                            section_id=str(data.get("section_id", "body")),
                            operation=data.get("operation", "update"),
                            content=str(data.get("content", "")).replace('\\n', '\n'),
                            reason=str(data.get("reason", "Sanitized newline JSON")),
                            model=data.get("model", fallback_model),
                            confidence=float(data.get("confidence", 0.7))
                        )
                except Exception:
                    pass

        return PatchOperation(
            section_id="overview",
            operation="replace_all" if len(cleaned) > 100 else "append",
            content=cleaned,
            reason="Heuristic fallback from non-JSON response",
            model=fallback_model,
            confidence=0.5
        )

    @classmethod
    def parse_sections_from_markdown(cls, text: str) -> Dict[str, Section]:
        sections: Dict[str, Section] = {}
        lines = text.split("\n")
        current_id = "intro"
        current_title = "Introduction"
        current_lines = []

        for line in lines:
            if line.startswith("#"):
                if current_lines:
                    sections[current_id] = Section(
                        id=current_id,
                        title=current_title,
                        content="\n".join(current_lines).strip(),
                        last_editor="",
                        last_version=0
                    )
                    current_lines = []
                header_text = line.lstrip("#").strip()
                current_id = re.sub(r"[^a-zA-Z0-9_]+", "_", header_text.lower()).strip("_") or "section"
                current_title = header_text
            else:
                current_lines.append(line)

        if current_lines or not sections:
            sections[current_id] = Section(
                id=current_id,
                title=current_title,
                content="\n".join(current_lines).strip(),
                last_editor="",
                last_version=0
            )

        return sections

    @classmethod
    def apply_patch(cls, state: DraftState, patch: PatchOperation) -> Tuple[DraftState, float]:
        old_text = state.full_text
        editor = patch.model or "unknown"
        target_sec = patch.section_id.strip()

        if not state.full_text.strip() or patch.operation == "replace_all":
            state.full_text = patch.content.strip()
            state.sections = cls.parse_sections_from_markdown(state.full_text)
            for s in state.sections.values():
                s.last_editor = editor
                s.last_version = state.version + 1
            state.last_editor = editor
            diff_ratio = cls.compute_diff_ratio(old_text, state.full_text)
            return state, diff_ratio

        if not state.sections:
            state.sections = cls.parse_sections_from_markdown(state.full_text)

        matched_key = None
        for k in state.sections.keys():
            if k == target_sec or target_sec in k or k in target_sec:
                matched_key = k
                break

        if not matched_key:
            title = target_sec.replace("_", " ").title()
            matched_key = target_sec
            state.sections[matched_key] = Section(
                id=matched_key,
                title=title,
                content="",
                last_editor=editor,
                last_version=state.version + 1
            )

        sec = state.sections[matched_key]
        if patch.operation == "update":
            sec.content = patch.content.strip()
        elif patch.operation == "append":
            sec.content = (sec.content + "\n\n" + patch.content.strip()).strip()
        elif patch.operation == "delete":
            sec.content = ""

        sec.last_editor = editor
        sec.last_version = state.version + 1

        assembled = []
        for s in state.sections.values():
            if not s.content:
                continue
            assembled.append(f"## {s.title}\n{s.content}")

        state.full_text = "\n\n".join(assembled).strip()
        state.last_editor = editor

        diff_ratio = cls.compute_diff_ratio(old_text, state.full_text)
        return state, diff_ratio

    @classmethod
    def compute_diff_ratio(cls, text_a: str, text_b: str) -> float:
        if not text_a and not text_b:
            return 0.0
        matcher = difflib.SequenceMatcher(None, text_a, text_b)
        return max(0.0, min(1.0, 1.0 - matcher.ratio()))

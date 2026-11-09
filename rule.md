# Repository Rules and Engineering Standards

## 1. Zero Synthetic or Fake Data Mandate
- Never create, hardcode, seed, or fall back to dummy, mock, placeholder, or synthetic data in production code, databases, or API responses.
- All model responses, factual statements, and outputs must originate from active authoritative models or declared upstream engines.
- If an upstream engine, local model daemon, or external service is unreachable or unconfigured, fail closed immediately with an explicit error (e.g., `ConnectionError` or `SOURCE_UNAVAILABLE`). Do not fabricate fallback responses.

## 2. Zero Ungrounded Assumptions
- Never infer, guess, or extrapolate parameters, village/cadastral geometries, model outputs, or fact verification statuses.
- Verification scores and claims must be derived strictly from verified data sources or explicit consensus protocols.
- Any unresolved entity, geometry, or inference must be explicitly marked with its true failure state rather than substituted with approximations.

## 3. Code Cleanliness: No Unnecessary Comments
- Code must be self-documenting through precise naming conventions, strict type annotations, and clean architectural separation.
- Prohibit trivial comments that merely narrate what the next line of code does.
- Prohibit conversational, narrative, or placeholder commentary.
- Comments are strictly limited to non-obvious mathematical derivations, hardware-specific memory constraints, and formal API docstrings.

## 4. Professional Folder Structure and Hygiene
- Maintain a clean, professional directory hierarchy without stray scripts, scratch files, or test outputs committed to the repository.
- Runtime artifacts (SQLite databases, WAL logs, JSONL audit logs, `.pytest_cache`, and virtual environments) must remain ignored and excluded via `.gitignore`.
- Every committed file must serve a distinct production, architectural, or testing purpose.

## 5. Git Protocol: Push Each File Individually
- Every modified or newly added file must be staged, committed, and pushed individually to GitHub.
- Do not perform bulk commits combining multiple disparate files.
- Each commit message must be atomic and descriptive, referencing the exact file and its precise technical change (e.g., `feat(storage): implement SQLite WAL-backed draft workspace`).
- Before pushing, ensure the file is formatted, linted, and compliant with all project rules.

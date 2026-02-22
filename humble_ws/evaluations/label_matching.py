import json
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


def _normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("/", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_label_configs(canonical_path: str, aliases_path: str) -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
    with open(canonical_path) as f:
        canonical_raw = json.load(f)
    with open(aliases_path) as f:
        aliases_raw = json.load(f)

    canonical = {_normalize(x) for x in canonical_raw.get("canonical_labels", [])}
    alias_map = {_normalize(k): _normalize(v) for k, v in aliases_raw.get("aliases", {}).items()}
    contains_rules = {_normalize(k): _normalize(v) for k, v in aliases_raw.get("contains_rules", {}).items()}
    return canonical, alias_map, contains_rules


def load_learned_aliases(learned_alias_path: str) -> Dict[str, str]:
    p = Path(learned_alias_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        aliases = data.get("aliases", {})
        return {_normalize(k): _normalize(v) for k, v in aliases.items()}
    except (OSError, json.JSONDecodeError):
        return {}


def _persist_learned_alias(
    learned_alias_path: str,
    raw_norm: str,
    canonical_label: str,
    method: str,
    score: float,
) -> None:
    p = Path(learned_alias_path)
    payload = {"aliases": {}, "metadata": {}}
    if p.exists():
        try:
            with open(p) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = {"aliases": {}, "metadata": {}}
    payload.setdefault("aliases", {})
    payload.setdefault("metadata", {})
    payload["aliases"][raw_norm] = canonical_label
    payload["metadata"][raw_norm] = {"method": method, "score": score}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def resolve_label(
    raw_label: str,
    canonical: Set[str],
    alias_map: Dict[str, str],
    contains_rules: Dict[str, str],
) -> Tuple[Optional[str], str]:
    norm = _normalize(raw_label)
    if not norm:
        return None, "empty"

    if norm in canonical:
        return norm, "canonical"

    if norm in alias_map:
        mapped = alias_map[norm]
        return (mapped if mapped in canonical else None), "alias"

    # Check "contains" rules after alias lookup for robust dataset-agnostic matching.
    for needle, mapped in contains_rules.items():
        if needle and needle in norm and mapped in canonical:
            return mapped, "contains"

    return None, "unresolved"


def _resolve_embedding_like(norm: str, canonical: Set[str]) -> Tuple[Optional[str], str, float]:
    labels = sorted(list(canonical))
    if not labels:
        return None, "embedding_none", 0.0

    # Preferred backend: sentence-transformers if installed.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sentence_transformers.util import cos_sim  # type: ignore
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([norm] + labels, convert_to_numpy=True, normalize_embeddings=True)
        q = emb[0:1]
        ref = emb[1:]
        sims = cos_sim(q, ref).cpu().numpy().reshape(-1)
        idx = int(np.argmax(sims))
        return labels[idx], "embedding_sentence_transformers", float(sims[idx])
    except Exception:
        pass

    # Fallback: token-overlap + sequence similarity (deterministic approximation).
    import difflib
    best_label = None
    best_score = -1.0
    q_tokens = set(norm.split())
    for c in labels:
        c_tokens = set(c.split())
        jacc = (len(q_tokens & c_tokens) / len(q_tokens | c_tokens)) if (q_tokens or c_tokens) else 0.0
        seq = difflib.SequenceMatcher(a=norm, b=c).ratio()
        score = 0.6 * jacc + 0.4 * seq
        if score > best_score:
            best_score = score
            best_label = c
    return best_label, "embedding_fallback", float(best_score)


def resolve_label_with_learning(
    raw_label: str,
    canonical: Set[str],
    alias_map: Dict[str, str],
    contains_rules: Dict[str, str],
    learned_alias_map: Optional[Dict[str, str]] = None,
    use_embedding_for_unresolved: bool = False,
    learn_new_aliases: bool = False,
    learned_alias_path: Optional[str] = None,
    embedding_min_score: float = 0.55,
) -> Tuple[Optional[str], str]:
    norm = _normalize(raw_label)
    if not norm:
        return None, "empty"

    if learned_alias_map is not None and norm in learned_alias_map and learned_alias_map[norm] in canonical:
        return learned_alias_map[norm], "learned_alias"

    mapped, source = resolve_label(raw_label, canonical, alias_map, contains_rules)
    if mapped is not None:
        return mapped, source

    if not use_embedding_for_unresolved:
        return None, "unresolved"

    emb_label, emb_source, score = _resolve_embedding_like(norm, canonical)
    if emb_label is None or score < embedding_min_score:
        return None, "unresolved"

    if learn_new_aliases and learned_alias_path:
        _persist_learned_alias(learned_alias_path, norm, emb_label, emb_source, score)
        if learned_alias_map is not None:
            learned_alias_map[norm] = emb_label
        return emb_label, f"{emb_source}_learned"

    return emb_label, emb_source


def default_config_paths(eval_dir: str) -> Tuple[str, str]:
    root = Path(eval_dir)
    return str((root / "canonical_labels.json").resolve()), str((root / "label_aliases.json").resolve())

"""Derive a two-level category taxonomy from Wikidata instance tags.

Replaces the hand-written tag→category table that mapped 90.6% of entities to
"other": its five categories came from GeoTIR's global benchmark and have no
slot for the shrines, temples and mountains that dominate this corpus.

Each entity gets:
  category_fine    normalized head noun (shrine, bridge, mountain, ...)
  category_coarse  curated bucket over fine labels (religious, natural, ...)
  category         alias for one of the above, per --category-level

The mapping lives in taxonomy_config.yaml, not here. Adding a country should
mean adding config entries for its loanwords, not editing this module.
"""

import argparse
import ast
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml

_CONFIG_PATH = Path(__file__).parent / "taxonomy_config.yaml"

_QID = re.compile(r"^q\d+$")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
# "city of Japan" -> "city"; "hanamachi in Osaka" -> "hanamachi". The romance
# prepositions cover labels arriving once the corpus leaves Japan.
_TRAILING_PHRASE = re.compile(r"\s+(?:of|in|de|du|della|del)\s+.*$", re.IGNORECASE)

NON_LANDMARK = "__non_landmark__"
UNMAPPED = "__unmapped__"

# Tag resolution outcomes.
KIND_FINE = "fine"
KIND_CLASSIFICATION = "classification"
KIND_NON_LANDMARK = "non_landmark"
KIND_UNMAPPED = "unmapped"


def tokenize(tag: str) -> list[str]:
    """Normalize a tag to its comparable tokens.

    Hyphens are deliberately not split — "Kokuhei-sha" is one token, and
    splitting it yields the meaningless head "sha".
    """
    s = _PARENTHETICAL.sub(" ", tag.lower().strip())
    s = _TRAILING_PHRASE.sub("", s)
    return s.strip(" -–—,").split()


def head_noun(tag: str) -> str:
    """Reduce a tag to its head noun — the last token of the English label.

    English compound types put the head last, so "stone arch bridge", "art
    museum" and "Hachiman shrine" collapse onto bridge/museum/shrine. This is
    what makes the taxonomy transfer across countries: it reads the structure
    of the label rather than a per-country lookup.
    """
    tokens = tokenize(tag)
    return tokens[-1] if tokens else ""


class Taxonomy:
    """Resolves instance tags to (fine, coarse) labels from a YAML config."""

    def __init__(self, config: dict):
        self.non_landmark_heads = set(config.get("non_landmark_heads") or [])
        self.classification_heads = set(config.get("classification_heads") or [])
        self.tag_overrides = {
            k.lower(): v for k, v in (config.get("tag_overrides") or {}).items()
        }
        self.synonyms = config.get("synonyms") or {}
        self.fine_to_coarse = config.get("fine_to_coarse") or {}
        self.low_information_fine = set(config.get("low_information_fine") or [])
        self.generic_structural_heads = set(config.get("generic_structural_heads") or [])

    def _fine_of_head(self, head: str) -> str | None:
        fine = self.synonyms.get(head, head)
        return fine if fine in self.fine_to_coarse else None

    def resolve_tag(self, tag: str) -> tuple[str, str | None]:
        """Resolve one tag to (kind, label). Label is the fine label when kind is 'fine'."""
        override = self.tag_overrides.get(tag.lower().strip())
        if override == NON_LANDMARK:
            return KIND_NON_LANDMARK, None
        if override == UNMAPPED:
            return KIND_UNMAPPED, head_noun(tag)
        if override is not None:
            return KIND_FINE, override

        tokens = tokenize(tag)
        head = tokens[-1] if tokens else ""
        if not head or _QID.fullmatch(head):
            return KIND_NON_LANDMARK, None
        if head in self.non_landmark_heads:
            return KIND_NON_LANDMARK, None
        if head in self.classification_heads:
            return KIND_CLASSIFICATION, head

        # "church building", "museum building", "station building": English puts
        # the type in the modifier and a bare structural noun at the head, so
        # head-noun extraction alone reads St Paul's Church as a "building".
        # Prefer the modifier, but only when it names a type we actually know —
        # "commercial complex" and "government building" have no usable
        # modifier and must fall back to the generic head.
        if head in self.generic_structural_heads and len(tokens) > 1:
            inner = self._fine_of_head(tokens[-2])
            if inner is not None:
                return KIND_FINE, inner

        fine = self.synonyms.get(head, head)
        if fine not in self.fine_to_coarse:
            return KIND_UNMAPPED, fine
        return KIND_FINE, fine

    def coarse_of(self, fine: str) -> str | None:
        return self.fine_to_coarse.get(fine)


def parse_tags(raw: str | None) -> list[str]:
    """instance_tag is a stringified Python list; tolerate malformed rows."""
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [t for t in parsed if isinstance(t, str) and t.strip()]


def _resolve_rows(taxonomy: Taxonomy, tag_lists: Iterable[list[str]]) -> list[dict]:
    rows = []
    for tags in tag_lists:
        fine, classifications, unmapped, saw_non_landmark = [], [], [], False
        for tag in tags:
            kind, label = taxonomy.resolve_tag(tag)
            if kind == KIND_FINE:
                fine.append(label)
            elif kind == KIND_CLASSIFICATION:
                classifications.append(tag)
            elif kind == KIND_UNMAPPED:
                unmapped.append(label)
            else:
                saw_non_landmark = True
        rows.append(
            {
                "fine": fine,
                "classifications": classifications,
                "unmapped": unmapped,
                "saw_non_landmark": saw_non_landmark,
            }
        )
    return rows


def _pick_primary(
    fine: list[str], specificity: dict[str, float], low_information: set[str]
) -> str:
    """Choose one fine label from an entity's tags by corpus specificity.

    The rarest label wins: a landmark tagged both "castle" and "museum" is
    better described as a castle, because "castle" carries more information.
    This replaces a hardcoded priority list, which had resolved a castle tagged
    with its keep ("tenshu") to *tower*.

    Rarity is only a proxy for informativeness, and it breaks for labels that
    are vague *and* uncommon — "tourist attraction" is rarer than "castle" but
    says far less. Those are demoted via config and can only win unopposed.
    Ties break alphabetically so output is stable across runs.
    """
    candidates = sorted(set(fine))
    informative = [f for f in candidates if f not in low_information]
    return max(informative or candidates, key=lambda f: specificity[f])


def assign_categories(
    df: pl.DataFrame,
    taxonomy: Taxonomy,
    category_level: str = "coarse",
    tag_col: str = "instance_tag",
) -> tuple[pl.DataFrame, dict]:
    """Add category_fine / category_coarse / category / is_landmark columns.

    Returns the annotated frame and a report dict for the coverage summary.
    """
    tag_lists = [parse_tags(v) for v in df[tag_col].to_list()]
    resolved = _resolve_rows(taxonomy, tag_lists)

    # Specificity = IDF over fine labels. Computed on entity counts, so a label
    # on many entities (museum) loses to a rare one (castle).
    n_docs = max(len(resolved), 1)
    doc_freq = Counter()
    for row in resolved:
        doc_freq.update(set(row["fine"]))
    specificity = {
        label: math.log(n_docs / count) for label, count in doc_freq.items()
    }

    fine_col, coarse_col, landmark_col, class_col = [], [], [], []
    for row in resolved:
        if row["fine"]:
            primary = _pick_primary(
                row["fine"], specificity, taxonomy.low_information_fine
            )
            fine_col.append(primary)
            coarse_col.append(taxonomy.coarse_of(primary))
            landmark_col.append(True)
        else:
            fine_col.append(None)
            coarse_col.append(None)
            # No usable type at all. Only call it a non-landmark when a tag
            # positively said so (a Q-ID, a prefecture). An unmapped tag means
            # the config has a gap — that is reported, never asserted as
            # evidence against the row being a landmark.
            landmark_col.append(not row["saw_non_landmark"])
        class_col.append(row["classifications"] or None)

    out = df.with_columns(
        pl.Series("category_fine", fine_col, dtype=pl.Utf8),
        pl.Series("category_coarse", coarse_col, dtype=pl.Utf8),
        pl.Series("is_landmark", landmark_col, dtype=pl.Boolean),
        pl.Series(
            "classification_tags",
            [str(c) if c else None for c in class_col],
            dtype=pl.Utf8,
        ),
    )
    alias = "category_coarse" if category_level == "coarse" else "category_fine"
    out = out.with_columns(pl.col(alias).alias("category"))

    # Two different gaps, both worth reporting. Blocking heads leave an entity
    # with no category at all. Shadowed heads are still config gaps, but a
    # co-occurring tag rescued the entity — so they never surface as unmapped
    # entities and would otherwise rot silently.
    blocking_heads, shadowed_heads = Counter(), Counter()
    for row in resolved:
        if row["fine"]:
            shadowed_heads.update(set(row["unmapped"]))
        else:
            blocking_heads.update(set(row["unmapped"]))

    report = {
        "n_entities": len(resolved),
        "unmapped_entities": sum(1 for r in resolved if not r["fine"] and r["unmapped"]),
        "blocking_heads": blocking_heads,
        "shadowed_heads": shadowed_heads,
        "non_landmark": sum(1 for lm in landmark_col if not lm),
        "no_type_landmarks": sum(
            1 for row, lm in zip(resolved, landmark_col) if lm and not row["fine"]
        ),
    }
    return out, report


def load_taxonomy(config_path: Path = _CONFIG_PATH) -> Taxonomy:
    with open(config_path) as f:
        return Taxonomy(yaml.safe_load(f))


_JOINED_COLS = ["category", "category_fine", "category_coarse", "is_landmark"]


def join_landmarks(landmarks: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    """Re-join entity categories onto the per-image landmark records.

    landmarks.csv carries a `category` copied from entities; it goes stale the
    moment the taxonomy changes. `pred_label` (GeoTIR's vision-model guess) is
    left untouched — it is a useful comparison baseline.
    """
    stale = [c for c in _JOINED_COLS if c in landmarks.columns]
    return landmarks.drop(stale).join(
        entities.select(["entity_id", *_JOINED_COLS]), on="entity_id", how="left"
    )


def _print_report(df: pl.DataFrame, report: dict) -> None:
    n = report["n_entities"]
    print(f"\n--- coarse distribution ({n} entities) ---")
    counts = df["category_coarse"].value_counts(sort=True)
    for row in counts.iter_rows(named=True):
        label = row["category_coarse"] or "(none)"
        print(f"  {label:<18} {row['count']:5d}  {row['count'] / n * 100:5.1f}%")

    print("\n--- top fine labels ---")
    fine_counts = df["category_fine"].value_counts(sort=True).head(20)
    for row in fine_counts.iter_rows(named=True):
        if row["category_fine"] is None:
            continue
        print(f"  {row['category_fine']:<22} {row['count']:5d}")

    unmapped = report["unmapped_entities"]
    print(
        f"\n--- coverage ---\n"
        f"  unmapped entities   : {unmapped:5d}  ({unmapped / n * 100:.2f}%)\n"
        f"  non-landmark rows   : {report['non_landmark']:5d}\n"
        f"  landmarks w/o type  : {report['no_type_landmarks']:5d}"
    )
    for title, heads in (
        ("blocking (entity has no category)", report["blocking_heads"]),
        ("shadowed (rescued by another tag)", report["shadowed_heads"]),
    ):
        if not heads:
            continue
        print(f"\n--- unmapped head nouns, {title} ---")
        for head, count in heads.most_common(40):
            print(f"  {count:4d}  {head}")


def _guard_output(path: Path, overwrite: bool) -> Path:
    """The source CSVs are this work's comparison baseline — never clobber."""
    if path.exists() and not overwrite:
        raise SystemExit(
            f"refusing to overwrite existing {path} (pass --overwrite to force)"
        )
    return path


def main(args):
    taxonomy = load_taxonomy(args.config)
    output = _guard_output(Path(args.output), args.overwrite)
    landmarks_output = (
        _guard_output(Path(args.landmarks_output), args.overwrite)
        if args.landmarks
        else None
    )

    df = pl.read_csv(args.input)
    df_out, report = assign_categories(df, taxonomy, category_level=args.category_level)
    _print_report(df_out, report)

    df_out.write_csv(output)
    print(f"\nwrote {output}")

    if args.landmarks:
        landmarks = pl.read_csv(args.landmarks)
        joined = join_landmarks(landmarks, df_out)
        joined.write_csv(landmarks_output)
        missing = joined["category"].null_count()
        print(
            f"wrote {landmarks_output} "
            f"({joined.height} rows, {missing} without a category)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-c", "--config", type=Path, default=_CONFIG_PATH)
    parser.add_argument(
        "--landmarks", help="landmarks CSV to re-join the new categories onto"
    )
    parser.add_argument(
        "--landmarks-output", default="dataset/csv/landmarks_taxonomy.csv"
    )
    parser.add_argument(
        "--category-level",
        choices=["coarse", "fine"],
        default="coarse",
        help="which level the `category` alias column mirrors",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)

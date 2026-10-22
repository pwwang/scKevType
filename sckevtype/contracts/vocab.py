"""The gene vocabulary: which symbols may appear in a state, and at what token cost.

The vocabulary is a frozen rule of the paper: a symbol outside it can never be
read by the model, so the OOV rate of any dataset is a number we report rather
than discover. Membership is decided once, here, and pinned with a digest.

Token cost is a *hook*, never a dependency: ``build_vocabulary`` optionally
takes a ``probe(symbol) -> n_tokens`` callable (the training layer passes a real
tokenizer) and the default is a no-op, so the contracts layer stays importable —
and testable — without transformers installed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .freeze import payload_digest, read_json, write_json
from .schema import CoverageReport


class VocabConfig(BaseModel):
    """Identity and admission rule of a vocabulary artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "protein_coding_human"
    version: str = "seed-1"
    #: Ceiling applied only when a tokenizer probe is supplied (design-notes §8:
    #: "protein-coding symbols the base tokenizer renders in <= 2 tokens").
    max_tokens: int = Field(default=2, ge=1)

    def digest(self) -> str:
        return payload_digest(self.model_dump(mode="json"))


class Vocabulary(BaseModel):
    """A sorted, de-duplicated symbol list plus the digest of the input it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    symbols: tuple[str, ...]
    #: Digest of the *input* gene list: proves which seed produced this vocabulary.
    source_digest: str = ""
    #: Name of the tokenizer probe applied, or None when admission was cost-free.
    probe: str | None = None

    def __contains__(self, symbol: object) -> bool:
        return symbol in set(self.symbols)

    def __len__(self) -> int:
        return len(self.symbols)

    def payload(self) -> dict:
        """Everything the digest covers — i.e. this artifact without its own digest."""
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return payload_digest(self.payload())

    def check_coverage(self, genes: Iterable[str]) -> CoverageReport:
        """How many of ``genes`` this vocabulary can express, and which ones it cannot.

        The OOV rate is a reported quantity (a dataset with a high OOV rate has
        states the model is blind to), never something to paper over by silently
        expanding the vocabulary.
        """
        return coverage(genes, set(self.symbols))

    def save(self, path: str) -> str:
        """Write the artifact (with its digest) and return that digest."""
        digest = self.digest()
        write_json(path, {**self.payload(), "digest": digest})
        return digest

    @classmethod
    def load(cls, path: str) -> Vocabulary:
        """Read an artifact, refusing one whose content does not match its recorded digest."""
        payload = read_json(path)
        digest = payload.pop("digest", None)
        vocabulary = cls.model_validate(payload)
        if digest != vocabulary.digest():
            raise ValueError(f"{path}: vocabulary digest mismatch (recorded {digest}, actual {vocabulary.digest()})")
        return vocabulary


def coverage(genes: Iterable[str], known: set[str]) -> CoverageReport:
    """Shared counting for vocabulary (OOV) and label maps (unmapped labels)."""
    total, missing = 0, []
    for gene in genes:
        total += 1
        if gene not in known:
            missing.append(gene)
    return CoverageReport(
        total=total, mapped=total - len(missing), unmapped=len(missing),
        rate=(total - len(missing)) / total if total else 1.0, unmapped_items=tuple(sorted(missing)),
    )


def build_vocabulary(
    genes: Iterable[str],
    cfg: VocabConfig | None = None,
    probe: Callable[[str], int] | None = None,
) -> Vocabulary:
    """Build a vocabulary from a symbol list, optionally admitting only cheap tokens.

    ``probe(symbol)`` returns that symbol's token count; symbols above
    ``cfg.max_tokens`` are dropped. With no probe (the default) every symbol is
    admitted, which is what the CPU-only tests and the seed build use.
    """
    cfg = cfg or VocabConfig()
    supplied = sorted({gene.strip() for gene in genes if gene and gene.strip()})
    symbols = supplied if probe is None else [gene for gene in supplied if probe(gene) <= cfg.max_tokens]
    return Vocabulary(
        name=cfg.name,
        version=cfg.version,
        symbols=tuple(symbols),
        source_digest=payload_digest(supplied),
        probe=getattr(probe, "__name__", None) if probe is not None else None,
    )

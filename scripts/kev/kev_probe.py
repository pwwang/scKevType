#!/usr/bin/env python3
"""The Kev rung of the model ladder: ask the open reconstruction *our* questions, then dump axis-1 rows.

Kev is `github.com/jaredpalmer/kev` — an independent, open reconstruction of this decision-model
class (LoRA adapter + pointer head on a Qwen3.5 base, serving TypeSafe's public
``/v1/systemone`` contract). This script puts the **same cells and the same asks** that
:mod:`sckevtype.jev.probe` put to Jev in front of a Kev checkpoint, and writes the result in the
schema ``scripts/axis1-recompute.py`` already reads.

The population is ``--sample``'s, and it is the **census** by default: ``reports/jev-full/sample.json``
is every eligible cell of ``runs/real-v2/test.jsonl`` (2,510 cells, as against the 200-cell smoke
draw in ``reports/jev-smoke/sample.json``, which is a subset of it). Every rung of the ladder is
scored on one population, so this is the one to use; passing the smoke sample explicitly reproduces
the smaller probe.

Three steps, for the same reason the Jev probe has three: they spend differently.

``plan``    Rebuild the population from the sample (by line number), assert the partition digest,
            and **prove the asks are the same bytes**: for every record, the body this script
            would hand Kev is compared against :func:`sckevtype.jev.request.record_request`'s body
            for Jev, and must be equal up to the ``model`` field. No weights, no GPU, no network.
``ask``     Answer the planned asks with a Kev checkpoint, in process. One JSON file per ask, in
            the Jev probe's own layout and written as soon as the ask returns, so a preemption or
            a wall-clock kill costs only the asks in flight; an ask whose file exists is never
            recomputed. ``--limit N`` runs the first N records (smoke).

            **Two kinds of arm, one scoring path.** Most rungs are ``base + LoRA adapter + head.pt``
            and load through the vendored ``kev.checkpoint.Checkpoint.load``. ``jaredpalmer/kev-27b``
            is a full finetune — 11 ``model-*.safetensors`` shards, ``head.pt`` ``weights: "full"``,
            no ``adapter_config.json`` — and cannot load there at all, because that loader reads the
            adapter config unconditionally. :mod:`kev_fullft` adds the no-adapter load beside the
            vendored package (never inside it) and reuses ``LocalPredictor.__call__`` verbatim, so the
            questions and the readout are identical whichever loader ran. ``--gpus N`` shards a
            full-weight checkpoint over N cards with ``accelerate``; a LoRA arm refuses ``N > 1``
            rather than quietly running on one.
``dump``    Project ``rows.jsonl`` into ``reports/axis1-rows-kev.jsonl`` through the *library's*
            :func:`sckevtype.eval.run.write_row_dump` — the same function the Jev dump went
            through, so the two dumps differ in their numbers and in nothing else.

Why in process and not over HTTP: ``kev.serve`` needs ``fastapi``, which the cluster venv does not
have, and a probe of 800 asks does not need a server. The call path is
:class:`kev.predictors.LocalPredictor`, which is the same encoder and the same pointer head the
server uses (``kev.serve`` calls ``LocatePredictor``'s ``model.encode``/``model.probs`` directly),
so nothing about the answers depends on which one is driven.

Usage (the cluster task script drives this; see ``<cluster-job>``)

    PYTHONPATH=<repo> python scripts/kev/kev_probe.py plan  --corpus runs/real-v2 --out runs/kev-probe
    PYTHONPATH=<repo> python scripts/kev/kev_probe.py ask   --out runs/kev-probe --run jaredpalmer/kev-9b
    PYTHONPATH=<repo> python scripts/kev/kev_probe.py dump  --out runs/kev-probe --dump reports/axis1-rows-kev.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sckevtype.contracts.freeze import file_digest, read_json, write_json
from sckevtype.contracts.paths import root
from sckevtype.eval import metrics
from sckevtype.eval.run import with_abstain, write_row_dump
from sckevtype.jev import probe as P
from sckevtype.jev.request import request_digest

# --------------------------------------------------------------------------- the encoder

#: Kev's shelf names (``kev.serve`` MODEL_NAMES). The encoder is TypeSafe's, so the name is a label on the
#: response, not a branch in the code path: nothing about the request bytes depends on it.
KEV_MODEL = "kev-latest"

#: What a Kev run's rows carry as ``served_model``. Jev's rows carry the versioned id its API reported
#: (``jev-1.13.0``); Kev's equivalent is the checkpoint the probe loaded, filled in at ask time.
SERVED_PREFIX = "kev"


def body_for(record: Mapping[str, Any], model: str = KEV_MODEL) -> dict[str, Any]:
    """The ``POST /v1/systemone`` body for one record — :func:`record_request` minus its model allow-list.

    :func:`sckevtype.jev.request.record_request` refuses any model name that is not one of Jev's, which is
    correct for a Jev probe and wrong for this one. Everything else is the same code: the same
    ``state_text`` (the corpus's own rendered string, byte for byte), the same ``question_payload``
    allow-list, the same ``check_request`` guard. :func:`assert_same_ask` is what keeps this from becoming
    a second encoder that could drift: it compares the body built here against the body the Jev probe
    would have sent, field for field.
    """
    from sckevtype.jev.request import check_request, question_payload, state_text

    body = {"state": state_text(record), "model": model,
            "questions": {qid: question_payload(q) for qid, q in record["questions"].items()}}
    check_request(body)
    return body


def variant_questions(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The abstain second ask's questions, or ``None`` when every choice question already offers one.

    :func:`sckevtype.eval.run.with_abstain` is the same function :func:`abstain_request` calls, so the two
    models are asked the same second question — including the injected option's key and wording, which is
    the one thing axis 1 is about.
    """
    variants = {qid: variant for qid, question in record["questions"].items()
                if (variant := with_abstain(question)) is not None}
    return variants or None


def label_violations(record: Mapping[str, Any]) -> list[str]:
    """Every question whose ``label`` is not one of its own option keys — named, not summarised.

    :func:`sckevtype.jev.probe.rows_of_answer` scores a choice row with
    ``keys.index(question["label"])``, and :func:`sckevtype.jev.request.label_index` says the same thing as
    a guard: the corpus's label *is* the option name. That is a property of the corpus (verified against
    the real partition), not something this script may assume: a record whose label is not among its keys
    is a record whose row cannot be built, and the failure mode is the bad one — the row silently takes the
    wrong index, or the run dies after the GPU time has been spent.

    The three question types are checked the way the row builder reads them: a choice label must be one of
    the criteria keys, a noul label must be a bool, and a score label must be an in-range level index.
    """
    from sckevtype.jev.request import option_keys

    violations = []
    for qid, question in record["questions"].items():
        label = question.get("label")
        qtype = question.get("type")
        try:
            keys = option_keys(question)
        except Exception as error:                       # a question the encoder cannot express at all
            violations.append(f"{qid}: {type(error).__name__}: {error}")
            continue
        if qtype == "choice" and label not in keys:
            violations.append(f"{qid}: label {label!r} is not one of the {len(keys)} option keys "
                              f"(first: {list(keys)[:3]})")
        elif qtype == "noul" and not isinstance(label, bool):
            violations.append(f"{qid}: a noul label must be a bool; got {label!r}")
        elif qtype == "score" and not (isinstance(label, int) and 0 <= int(label) < len(keys)):
            violations.append(f"{qid}: a score label must be a level index in 0..{len(keys) - 1}; got {label!r}")
    return violations


def record_identity(record: Mapping[str, Any]) -> str:
    """A corpus record's own name for itself: ``_meta.group_id``, the ``"<dataset>/<cell>"`` form the samples
    and the Jev rows both use.

    The single accessor for it. A bare ``record["_meta"]["id"]`` is not a key any record of this corpus has
    (4,000 of 4,000 lack it) — the population cross-check was written that way and could not run at all on
    real data while passing on synthetic records that happened to carry ``id``.
    """
    meta = record.get("_meta") or {}
    return str(meta.get("group_id") or meta.get("id") or "<no _meta.group_id>")


def check_labels(record: Mapping[str, Any]) -> None:
    """Refuse a record whose labels are not answerable options, naming the record, the question and the keys."""
    violations = label_violations(record)
    if violations:
        identity = record_identity(record)
        raise P.ProbeError(f"{identity}: {len(violations)} question(s) carry a label that is not one of "
                           f"their own option keys -- the row builder cannot score this record: "
                           f"{'; '.join(violations[:3])}")


# --------------------------------------------------------------------------- the model

class KevError(RuntimeError):
    """The checkpoint cannot be loaded or asked as this probe needs it."""


class KevPredictor:
    """A Kev checkpoint as a callable ``record -> {"probabilities", "answers", ...}``.

    The return shape is :class:`sckevtype.jev.client.JevClient`'s, so
    :func:`sckevtype.jev.probe.rows_of_answer` — the function that builds the rows the Jev dump is a
    projection of — reads a Kev answer without knowing which model produced it. ``answers`` is built by
    kev's own :func:`kev.api.to_answers`, so the ``choice``/``confidence`` a row carries is what a client
    of ``kev.serve`` would have been handed, rounding included.

    **Two loaders, one scoring path.** ``jaredpalmer/kev-27b`` is a full finetune — 11
    ``model-*.safetensors`` shards and no ``adapter_config.json`` — and ``kev.checkpoint.Checkpoint.load``
    is a LoRA loader that reads that file unconditionally, so it cannot load it at all. Which loader this
    class uses is decided in one place (:func:`kev_fullft.checkpoint_kind`) and the *scoring* code is
    shared: :class:`kev_fullft.FullFinetunePredictor` subclasses ``LocalPredictor`` and inherits its
    ``__call__``. The LoRA arms construct ``LocalPredictor`` exactly as they did before, and asking for
    more than one GPU on one of them is refused rather than ignored.
    """

    def __init__(self, run: str, device: str = "cuda", dtype: str = "bf16", context: str = "serving",
                 temperature: float | None = None, lora_scale: float = 1.0, gpus: int = 1,
                 reserve_gib: float = 2.0) -> None:
        import kev_fullft
        import torch
        from kev.api import to_answers, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.data import api_request
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_PACKED, SERVE_MAX_STATE
        from kev.predictors import LocalPredictor
        from kev.suite import CONTEXT

        if dtype not in ("bf16", "fp32"):
            raise KevError(f"--dtype must be bf16 or fp32; got {dtype!r}")
        opts = LoadOptions(dtype=torch.bfloat16 if dtype == "bf16" else torch.float32,
                           merge=True, temperature=temperature, lora_scale=lora_scale)
        # The state/branch budgets. `serving` is the pair kev.serve enforces (and what kev.benchmark uses
        # for a suite frozen without admission, which is what our corpus is); `training` is kev.suite's own.
        self.context = ({"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH,
                         "max_packed": SERVE_MAX_PACKED} if context == "serving" else dict(CONTEXT))
        self.run = str(run)
        self.dtype, self.lora_scale, self.device = dtype, lora_scale, device
        self.gpus, self.reserve_gib = int(gpus), float(reserve_gib)
        checkpoint = Checkpoint(self.run)
        self.checkpoint_path = checkpoint.path
        # ONE place decides which loader this run is, and it refuses a contradiction (head.pt saying
        # `full` while an adapter sits in the directory) instead of picking one.
        self.kind = kev_fullft.checkpoint_kind(checkpoint)
        self.meta = {"base": checkpoint.meta.base, "base_revision": checkpoint.meta.base_revision,
                     "lora": checkpoint.meta.lora, "head_dim": checkpoint.meta.head_dim,
                     "option_isolation": checkpoint.meta.option_isolation,
                     "weights_dtype": checkpoint.meta.weights_dtype,
                     "weights": checkpoint.meta.extra.get("weights"),
                     "temperature_in_checkpoint": checkpoint.meta.temperature}
        try:
            if self.kind == kev_fullft.FULL:
                self.inner = kev_fullft.FullFinetunePredictor(
                    self.run, device, opts, self.context, gpus=self.gpus,
                    reserve_bytes=int(self.reserve_gib * 1024**3))
            else:
                if self.gpus != 1:
                    raise KevError(f"--gpus {self.gpus} was asked for a LoRA checkpoint ({self.run}): "
                                   f"the vendored loader pins one card and the multi-GPU path is only "
                                   f"defined for a full-weight checkpoint. Refusing rather than running "
                                   f"on one card and reporting it as {self.gpus}.")
                self.inner = LocalPredictor(self.run, device, opts, self.context)
        except KevError:
            raise
        except Exception as error:                       # a missing base, an OOM, a dtype the card refuses
            raise KevError(f"loading {self.run} on {device} ({dtype}) failed: "
                           f"{type(error).__name__}: {error}") from error
        self.temperature = float(self.inner.temperature)
        # The placement is the full-weight path's own record of where the weights went (empty for a LoRA
        # arm, where the loader pins one device and there is nothing to place).
        self.placement = getattr(self.inner, "placement", None)
        self._to_answers, self._to_record, self._api_request = to_answers, to_record, api_request

    @property
    def served_model(self) -> str:
        """What the rows record as the answering checkpoint: the run id and the head's temperature."""
        return f"{self.run} (temperature {self.temperature:.4f}, {self.dtype})"

    def __call__(self, record: Mapping[str, Any], questions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One record's answers. Exceptions are left to the caller: a record the checkpoint cannot encode is
        a rejected record (:class:`kev.model.ContextOverflow` names it), and anything else — an OOM above
        all — is equipment and must stop the run rather than quietly shrink the population.
        """
        active = record if questions is None else {**record, "questions": dict(questions)}
        prediction = self.inner(active)
        keys = list(prediction["probabilities"])
        rec, meta = self._to_record_keyed(self._api_request(active))
        probabilities = {qid: {key: float(value) for key, value in prediction["probabilities"][qid].items()}
                         for qid in keys}
        answers = self._to_answers([[probabilities[m["id"]][key] for key in m["keys"]] for m in meta], meta)
        return {"probabilities": probabilities, "answers": answers, "served_model": self.served_model,
                "latency_ms": float(prediction["latency_ms"]), "input_tokens": int(prediction["input_tokens"])}

    def _to_record_keyed(self, body: Mapping[str, Any]) -> tuple[dict, list[dict]]:
        """``kev.api.to_record`` on a request body already stripped to the wire fields."""
        from kev.api import SystemOneRequest

        return self._to_record(SystemOneRequest.model_validate(body))


# --------------------------------------------------------------------------- plan

#: Published repo sizes, measured from the Hub API (``/api/models/<repo>?blobs=true``), so a
#: missing-weights report can state a size without the box that needs it having internet. A LoRA arm's
#: base is named by the adapter's own ``adapter_config.json``; ``Checkpoint.load`` fetches it with the
#: adapter. A **full-weight** arm names no base: the repo IS the model (config, tokenizer and shards),
#: which is why ``jaredpalmer/kev-27b`` is 51.3 GB and needs no 55.6 GB ``Qwen/Qwen3.8-27B`` beside it.
WEIGHTS = {
    "jaredpalmer/kev-0.5b": {"adapter_bytes": 53_012_480, "base": "Qwen/Qwen2.5-0.5B", "base_bytes": 999_628_800},
    "jaredpalmer/kev-0.6b": {"adapter_bytes": 58_542_080, "base": "Qwen/Qwen3-0.6B-Base", "base_bytes": 1_203_671_040},
    "jaredpalmer/kev-0.8b": {"adapter_bytes": 65_600_000, "base": "Qwen/Qwen3.5-0.8B-Base", "base_bytes": 1_770_000_000},
    "jaredpalmer/kev-4b": {"adapter_bytes": 159_700_000, "base": "Qwen/Qwen3.5-4B-Base", "base_bytes": 9_340_000_000},
    "jaredpalmer/kev-8b": {"adapter_bytes": 199_065_600, "base": "Qwen/Qwen3-8B-Base", "base_bytes": 16_393_093_120},
    "jaredpalmer/kev-9b": {"adapter_bytes": 201_700_000, "base": "Qwen/Qwen3.5-9B-Base", "base_bytes": 19_330_000_000},
    "jaredpalmer/kev-27b": {"adapter_bytes": 51_279_886_109, "base": None, "base_bytes": 0,
                           "weights": "full", "measured": "2026-10-02, 20 files, 11 shards",
                           "head_pt_bytes": 10_494_335},
}


def human_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def cached_repo_bytes(hub: Path, repo: str) -> int:
    """Bytes of a hub repo actually on this filesystem (0 when the repo is absent)."""
    directory = hub / ("models--" + repo.replace("/", "--"))
    blobs = directory / "blobs"
    return sum(f.stat().st_size for f in blobs.glob("*")) if blobs.is_dir() else 0


def snapshot_dir(hub: Path, repo: str) -> Path | None:
    """The cached snapshot of a hub repo: ``refs/main``'s commit, else the newest ``snapshots/*``.

    Never ``snapshots/main`` — a ref is a *file* holding a commit hash, not a directory named ``main``,
    and a hardcoded ``snapshots/main`` reads exactly like an absent model.
    """
    directory = hub / ("models--" + repo.replace("/", "--"))
    ref = directory / "refs" / "main"
    if ref.is_file():
        pinned = directory / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if pinned.is_dir():
            return pinned
    snapshots = sorted(path for path in (directory / "snapshots").glob("*") if path.is_dir())
    return snapshots[-1] if snapshots else None


def full_weight_files(hub: Path, repo: str) -> dict[str, Any]:
    """A full-weight repo's required files, and which of them are absent or empty.

    The weights are sharded and an index names them, so a presence check that only counts bytes cannot
    see a *lost shard*: 10 of 11 shards is 46 GB of a 51 GB model, which loads as far as the missing
    tensor and then fails. Read the shard list out of ``model.safetensors.index.json`` and require every
    one of them, the way ``pack-kev-weights.sh``'s own guard does.
    """
    snapshot = snapshot_dir(hub, repo)
    if snapshot is None:
        return {"snapshot": None, "required": [], "missing": ["<the repo is not in this hub cache>"]}
    required = ["config.json", "head.pt", "model.safetensors.index.json", "tokenizer.json"]
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            required += sorted(set(weight_map.values()))
        except Exception as error:                      # an unreadable index is a missing index
            required.append(f"<model.safetensors.index.json is unreadable: {type(error).__name__}>")
    missing = [name for name in required if not (snapshot / name).is_file() or not (snapshot / name).stat().st_size]
    return {"snapshot": str(snapshot), "required": len(required), "shards": len(required) - 4,
            "missing": missing}


def base_from_adapter_config(hub: Path, shelf: str) -> str | None:
    """The base a LoRA arm names in its own ``adapter_config.json``.

    The ``WEIGHTS`` table above is a convenience for repos we have measured, not the source of truth --
    the adapter carries its base's name, which is what the table's own comment claims. Reading it here is
    what keeps an *unlisted* shelf honest: without it ``base`` is ``None`` for any repo missing from the
    table, and ``weights_block``'s completeness test degenerates to "the adapter is cached", which
    reports a run as ready while its base is entirely absent.
    """
    snap = snapshot_dir(hub, shelf)
    if snap is None:
        return None
    try:
        return json.loads((snap / "adapter_config.json").read_text(encoding="utf-8")).get("base_model_name_or_path") or None
    except Exception:                                   # an unreadable config is a missing config
        return None


def weights_block(run: str, hub: Path) -> dict[str, Any]:
    """What a Kev run needs on disk, what is there, and the exact fetch command for the rest.

    Two shapes, because the two kinds of arm are two shapes. A LoRA arm needs its adapter **and** the
    base the adapter names. A full-weight arm needs only itself — and it needs *all of itself*, so the
    verdict for one is a shard-level completeness check rather than a byte count.
    """
    shelf = run.split("@", 1)[0]
    known = WEIGHTS.get(shelf)
    adapter = cached_repo_bytes(hub, shelf)
    base = known["base"] if known else base_from_adapter_config(hub, shelf)
    base_bytes = cached_repo_bytes(hub, base) if base else 0
    kind = (known or {}).get("weights", "lora")
    block = {
        "run": run, "shelf": shelf, "hub": str(hub), "kind": kind,
        "adapter": {"repo": shelf, "cached_bytes": adapter,
                    "published_bytes": (known or {}).get("adapter_bytes")},
        "base": {"repo": base, "cached_bytes": base_bytes,
                 "published_bytes": (known or {}).get("base_bytes")},
        # A shelf with no table entry AND no readable ``adapter_config.json`` has no nameable base, so its
        # completeness cannot be established at all. Saying "present" from the adapter alone is how a job
        # starts and then dies on weights that were never there, so that case reports False by name.
        "base_model": base,
        "base_source": "table" if known else ("adapter_config.json" if base else None),
        "present": (bool(adapter) and (known is not None or base is not None)
                    and (base is None or bool(base_bytes))),
        "why_not_download_on_the_cluster": ("the standing rule for this project: compute nodes are offline "
                                            "(every task script exports HF_HUB_OFFLINE=1), so weights are "
                                            "fetched on a box with internet and transferred"),
    }
    if kind == "full":
        # The checkpoint is self-contained: config, tokenizer, head.pt and every shard the index names.
        block["self_contained"] = True
        block["files"] = full_weight_files(hub, shelf)
        published = (known or {}).get("adapter_bytes")
        # Both readings, because they catch different failures: the shard list catches a *lost* file, and
        # the byte total catches a *short* one (present, non-empty, truncated) -- which the shard list
        # cannot see and which is how a broken push reads as a complete checkpoint. `>=` rather than `==`
        # because a cache legitimately holds more than one revision's blobs.
        block["bytes"] = {"on_disk": adapter, "published": published,
                          "complete": None if not published else adapter >= published}
        block["present"] = (bool(adapter) and not block["files"]["missing"]
                            and block["bytes"]["complete"] is not False)
        block["fetch_locally_then_transfer"] = (
            f"bash <cluster-job> 27b   # hf download {shelf} (self-contained: no base)"
            f" then bash <cluster-job> 27b && bash <cluster-job>")
    else:
        block["self_contained"] = False
        block["fetch_locally_then_transfer"] = (
            f"hf download {shelf}   # plus {base or '<base from adapter_config.json>'}; then "
            f"tar the hf-cache hub dirs and `ahn-scp.sh put` them into the cluster hf-cache/hub")
    return block


def kev_identity() -> dict[str, Any]:
    """The revision of the reconstruction this run is pinned to: file digests, which travel to the report."""
    import importlib.util

    digests = {}
    for name in ("api", "checkpoint", "data", "device", "model", "predictors", "suite"):
        spec = importlib.util.find_spec("kev." + name)
        if spec and spec.origin:
            path = Path(spec.origin)
            digests[f"kev/{name}.py"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path)}
    return {"package_root": str(Path(digests["kev/api.py"]["path"]).parent) if digests else None,
            "files": digests}


def cmd_plan(args: argparse.Namespace) -> int:
    from sckevtype.jev.request import DEFAULT_MODEL as JEV_MODEL
    from sckevtype.jev.request import record_request

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = read_json(Path(args.sample))
    corpus = Path(args.corpus)
    partition = corpus / f"{sample['split']}.jsonl"
    if not partition.exists():
        raise P.ProbeError(f"no {partition}")
    digest = file_digest(partition)
    same_corpus = digest == sample["corpus_sha256"]
    if not same_corpus and not args.allow_other_corpus:
        raise P.ProbeError(f"{partition} is not the file the sample was drawn from "
                           f"({digest} != {sample['corpus_sha256']}); the sample names records by line "
                           f"number and would silently serve different cells")

    tasks = P.collect_records(partition, sample)
    if args.limit:
        tasks = tasks[: args.limit]

    mismatched, families, questions, state_chars = [], Counter(), Counter(), []
    per_family = Counter()
    bad_labels: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    label_questions = 0
    for group, family, record in tasks:
        per_family[family] += 1
        # The sample names a cell by line number; the record names itself by `_meta.group_id`. If the two
        # disagree the sample is being served against the wrong cell, and every row would be attributed to a
        # cell it did not come from — checked for every planned record, on real data, before any GPU time.
        identity = record_identity(record)
        if identity != group:
            identity_mismatches.append({"group": group, "record_identity": identity, "family": family})
        state_chars.append(len(record["state"]))
        families[record["_meta"]["family"]] += 1
        label_questions += len(record["questions"])
        # The invariant the row builder leans on: a choice label is one of its own option keys. Checked here
        # for every planned record, before any GPU time is spent, and named rather than counted.
        for violation in label_violations(record):
            bad_labels.append({"id": f"{group}/{family}", "violation": violation})
        for qid, question in record["questions"].items():
            questions[question["type"]] += 1
            try:
                kev_body = body_for(record, KEV_MODEL)
                jev_body = record_request(record, JEV_MODEL)
            except Exception as error:
                mismatched.append({"id": f"{group}/{family}", "question": qid,
                                   "error": f"{type(error).__name__}: {error}"})
                continue
            if {**kev_body, "model": jev_body["model"]} != jev_body:
                mismatched.append({"id": f"{group}/{family}", "question": qid,
                                   "error": "the Kev body differs from the Jev body beyond `model`"})

    # The population check that does not depend on this script's own reading of the sample: the ids the
    # Jev run already produced, read off its own rows file, must be exactly the ids planned here.
    ids_here = sorted({f"{sample['split']}/{group}/{family}" for group, family, _ in tasks})
    jev_rows = Path(args.jev_rows)
    population = {"source": str(jev_rows), "checked": False,
                  "identity_from_record": len(tasks) - len(identity_mismatches),
                  "identity_mismatches": identity_mismatches[:5]}
    if jev_rows.exists():
        seen = set()
        for line in jev_rows.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                seen.add(f"{row['split']}/{row['dataset']}/{row['cell']}/{row['family']}")
        planned = {f"{sample['split']}/{group}/{family}" for group, family, _ in tasks}
        population.update({"checked": True, "jev_records": len(seen), "planned_records": len(planned),
                           "only_in_jev": sorted(seen - planned)[:5], "only_in_plan": sorted(planned - seen)[:5],
                           "identical": seen == planned})

    plan = {
        "step": "plan", "probe": "sckevtype.kev",
        "sample": {"path": str(args.sample), "sha256": file_digest(Path(args.sample)), "seed": sample["seed"],
                   "cells": sample["cells"], "split": sample["split"], "families": sample["families"]},
        "corpus": {"path": str(partition), "sha256": digest, "sample_digest": sample["corpus_sha256"],
                   "identical": same_corpus},
        "population": population,
        "records_planned": len(tasks), "records_by_family": dict(sorted(per_family.items())),
        "questions_by_type": dict(sorted(questions.items())),
        "state_chars": {"min": min(state_chars), "max": max(state_chars),
                        "mean": round(sum(state_chars) / len(state_chars), 1)},
        "encoder": {"checked_asks": sum(per_family.values()), "mismatched": mismatched[:10],
                    "mismatch_count": len(mismatched),
                    "claim": ("every ask's Kev body equals the body the Jev probe sent for the same record "
                              "and question, field for field, apart from `model`")},
        "labels": {"questions_checked": label_questions, "violations": bad_labels[:10],
                   "violation_count": len(bad_labels),
                   "claim": ("every question's label is one of its own option keys (choice) / a bool (noul) / "
                             "an in-range level (score), which is what rows_of_answer's keys.index() needs")},
        "abstain_variants": sum(1 for _, _, record in tasks if variant_questions(record) is not None),
        "kev": kev_identity(),
        "weights": weights_block(args.run, Path(args.hub)),
        "budget": {"context": args.context},
    }
    write_json(out / "plan.json", plan)

    print(f"plan: {len(tasks)} records over {len(per_family)} families from {partition.name}")
    print(f"  corpus digest {'matches' if same_corpus else 'DIFFERS FROM'} the sample's "
          f"({digest[:16]}...)")
    print(f"  families {dict(sorted(per_family.items()))}; questions {dict(sorted(questions.items()))}")
    print(f"  states {min(state_chars)}..{max(state_chars)} chars (mean {plan['state_chars']['mean']})")
    print(f"  encoder: {len(mismatched)} mismatch(es) over {len(tasks)} asks"
          + ("" if not mismatched else f" -- first: {mismatched[0]}"))
    print(f"  labels: {label_questions} question(s) checked, {len(bad_labels)} violation(s)"
          + ("" if not bad_labels else f" -- first: {bad_labels[0]}"))
    if population["checked"]:
        print(f"  population: {'identical to' if population['identical'] else 'DIFFERS from'} the Jev rows "
              f"({population['planned_records']} v {population['jev_records']} records)")
    else:
        print(f"  population: not cross-checked ({jev_rows} absent)")
    weights = plan["weights"]
    print(f"  weights: {weights['kind']} arm; checkpoint {human_bytes(weights['adapter']['cached_bytes'])} "
          f"on disk of {human_bytes(weights['adapter']['published_bytes'] or 0)} published; "
          + ("self-contained (no base needed)" if weights.get("self_contained")
             else f"base {(weights['base']['repo'] or '?')} "
                  f"{human_bytes(weights['base']['cached_bytes'])} on disk of "
                  f"{human_bytes(weights['base']['published_bytes'] or 0)} published")
          + f"; {'PRESENT' if weights['present'] else 'MISSING'}")
    if weights.get("files"):
        files = weights["files"]
        print(f"    completeness: {files.get('shards', 0)} shard(s) + config/head/index/tokenizer from the "
              f"index; missing: {files['missing'] if files['missing'] else 'none'}")
    print(f"wrote {out / 'plan.json'}")
    return 0 if not mismatched and not bad_labels else 1


# --------------------------------------------------------------------------- ask


def ask_paths(raw: Path, dataset: str, cell: str, family: str, ask: str) -> Path:
    return P.ask_path(raw, dataset, cell, family, ask)


def ask(client: KevPredictor, record: Mapping[str, Any],
        questions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One model call, with a record the checkpoint cannot encode *named* rather than raised as equipment.

    :class:`kev.model.ContextOverflow` is the model's own statement that the record does not fit its
    serving budget. It is a property of the record, so it becomes a rejected record (counted in the
    ask log). Anything else — an OOM above all — is the machine, and is allowed to stop the run: a
    population that quietly shrank to whatever the GPU happened to afford is not the planned population.
    """
    try:
        return client(record) if questions is None else client(record, questions=questions)
    except Exception as error:
        from kev.model import ContextOverflow

        if isinstance(error, (ContextOverflow, ValueError, KeyError)):
            raise P.ProbeError(f"{type(error).__name__}: {error}") from error
        raise


def build_rows(record: Mapping[str, Any], family: str, split: str, client: KevPredictor, raw: Path,
               dataset: str, cell: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One record's asks, saved and answered — :func:`sckevtype.jev.probe.build_rows` with Kev as the client.

    The saved object is the Jev probe's: ``request`` (what a client would have put on the wire),
    ``request_sha256`` (the same digest function, so the two runs' asks are comparable), ``response``, and
    the ``prediction`` the row builder reads. A saved body is reused rather than recomputed, so a resumed
    run produces the bytes the first run would have.

    :func:`check_labels` runs first: a record whose label is not one of its own option keys cannot be
    scored by ``rows_of_answer``, and finding that out after the forward pass is finding it out too late.
    """
    check_labels(record)
    base = ask_paths(raw, dataset, cell, family, "base")
    if base.exists():
        prediction = read_json(base)["prediction"]
    else:
        body = body_for(record, KEV_MODEL)
        prediction = ask(client, record)
        write_json(base, {"request": body, "request_sha256": request_digest(body),
                          "response": {"model": prediction["served_model"], "answers": prediction["answers"]},
                          "prediction": prediction})

    variants = None
    variant_questions_map = variant_questions(record)
    if variant_questions_map is not None:
        path = ask_paths(raw, dataset, cell, family, "abstain")
        if path.exists():
            saved = read_json(path)
        else:
            body = body_for({**record, "questions": variant_questions_map}, KEV_MODEL)
            answered = ask(client, record, variant_questions_map)
            saved = {"request": body, "request_sha256": request_digest(body),
                     "response": {"model": answered["served_model"], "answers": answered["answers"]},
                     "prediction": answered}
            write_json(path, saved)
        # The *saved* ask's questions, and :func:`sckevtype.jev.probe.variant_map`'s shape: the row builder
        # reads the option order and criteria of the question that was actually served, not of the one the
        # code would build now. (This is the Jev probe's own construction, called rather than reimplemented.)
        variants = P.variant_map(saved["request"]["questions"], saved["prediction"])
    return P.rows_of_answer(record, family, split, prediction, variants), prediction


def cmd_ask(args: argparse.Namespace) -> int:
    out = Path(args.out)
    plan = read_json(out / "plan.json")
    sample = read_json(Path(args.sample))
    partition = Path(args.corpus) / f"{sample['split']}.jsonl"
    if file_digest(partition) != plan["corpus"]["sha256"]:
        raise P.ProbeError(f"{partition} changed since plan.json was written; the planned asks name different records")
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    predictor = KevPredictor(args.run, device=args.device, dtype=args.dtype, context=args.context,
                             temperature=args.temperature, lora_scale=args.lora_scale,
                             gpus=args.gpus, reserve_gib=args.reserve_gib)
    print(f"kev run {predictor.run} ({predictor.checkpoint_path})")
    print(f"  kind {predictor.kind}; base {predictor.meta['base']}@{predictor.meta['base_revision']}; "
          f"lora {predictor.meta['lora']}; temperature {predictor.temperature:.4f} "
          f"({args.dtype}, {args.device}); context {predictor.context}")
    if predictor.placement:
        # The full-weight path's placement, printed rather than inferred: which devices the weights went
        # to, in what dtype, on which attention backend. A run that silently landed on CPU is a run whose
        # wall clock is a day, and nothing else in this log would say so.
        placement = predictor.placement
        print(f"  placement: dtype {placement['dtype']}, attn {placement['attn']}, "
              f"weights {human_bytes(placement['weight_bytes'])} over {placement['devices']} "
              f"({ {k: human_bytes(v) for k, v in placement['weight_bytes_per_device'].items()} })")

    tasks = P.collect_records(partition, sample)
    if args.limit:
        tasks = tasks[: args.limit]
    reused = sum(ask_paths(raw, P.dataset_of(group), group.split("/", 1)[1], family, ask).exists()
                 for group, family, _ in tasks for ask in P.ASKS)
    phase = "resumed" if reused else "cold"
    print(f"{len(tasks)} records to ask; {reused} ask file(s) already on disk; "
          f"{'resuming' if reused else 'cold start'}")

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    started = time.time()
    for index, (group, family, record) in enumerate(tasks, start=1):
        dataset, cell = P.dataset_of(group), group.split("/", 1)[1]
        try:
            new_rows, _ = build_rows(record, family, sample["split"], predictor, raw, dataset, cell)
        except P.ProbeError as error:
            rejected.append({"id": f"{sample['split']}/{dataset}/{cell}/{family}",
                             "family": family, "error": str(error)})
            print(f"  REJECTED {family} {dataset}/{cell}: {error}", flush=True)
            continue
        rows.extend(new_rows)
        if index % args.progress_every == 0 or index == len(tasks):
            rate = index / max(time.time() - started, 1e-9)
            print(f"  [{index:4d}/{len(tasks)}] rows={len(rows):5d} rejected={len(rejected)} "
                  f"{rate:.2f} records/s", flush=True)
            write_rows_partial(out, rows, rejected, predictor, started, len(tasks), phase)

    write_rows_partial(out, rows, rejected, predictor, started, len(tasks), "final")
    print(f"\n{len(rows)} rows from {len(tasks) - len(rejected)} records; {len(rejected)} rejected")
    print(f"wrote {out / 'rows.jsonl'} and {out / 'ask-log.json'}")
    return 0 if rows else 1


def write_rows_partial(out: Path, rows: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]],
                       predictor: KevPredictor, started: float, records: int, phase: str) -> None:
    """``rows.jsonl`` after every progress line, so a killed job still leaves a readable pass behind."""
    P.write_rows(out / "rows.jsonl", rows)
    write_json(out / "ask-log.json", {
        "step": "ask", "phase": phase, "records_planned": records,
        "rows": len(rows), "rejected": list(rejected)[:20], "rejected_count": len(rejected),
        "model": predictor.served_model, "run": predictor.run, "dtype": predictor.dtype,
        "device": predictor.device, "temperature": predictor.temperature,
        "checkpoint_meta": predictor.meta, "context": predictor.context,
        # Which loader ran and where the weights ended up. A full-weight arm's placement is the only
        # record that the sharding happened at all: the answers look the same on one card (or on CPU,
        # slowly), so without this field a mis-sharded run is indistinguishable from a good one.
        "checkpoint_kind": predictor.kind, "gpus": predictor.gpus, "placement": predictor.placement,
        "lora_scale": predictor.lora_scale, "elapsed_s": round(time.time() - started, 1),
    })


# --------------------------------------------------------------------------- dump


def cmd_dump(args: argparse.Namespace) -> int:
    """Project ``rows.jsonl`` into the axis-1 dump and score it — with the library's own functions."""
    out = Path(args.out)
    rows = P.read_rows(out / "rows.jsonl")
    if not rows:
        raise P.ProbeError(f"{out / 'rows.jsonl'} holds no rows; nothing to dump")
    choice = [dict(row) for row in rows if row["type"] == "choice"]
    target = Path(args.dump) if Path(args.dump).is_absolute() else root() / args.dump
    dump = write_row_dump(target, choice)

    fields = tuple(json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
    from sckevtype.eval.run import DUMP_FIELDS

    keys = set(fields[0]) if fields else set()
    positive_agrees = all(bool(row["positive"]) == metrics.axis1_label(row) for row in fields)
    adequacy = metrics.panel_adequacy(choice, args.fpr)
    report = {
        "step": "dump", "probe": "sckevtype.kev", "rows": len(rows), "choice_rows": len(choice),
        "dump": {**dump, "schema_ok": keys == set(DUMP_FIELDS) | {"positive"},
                 "keys_extra": sorted(keys - (set(DUMP_FIELDS) | {"positive"})),
                 "keys_missing": sorted((set(DUMP_FIELDS) | {"positive"}) - keys),
                 "positive_agrees_with_axis1_label": positive_agrees,
                 "reader": (f"scripts/axis1-recompute.py reads reports/axis1-rows-<name>.jsonl; the name for "
                            f"this dump is '{Path(args.dump).stem.replace('axis1-rows-', '')}', which is "
                            "already in its DUMPS tuple. Its as_served() passes these rows through "
                            "untouched because they carry `abstain` and `abstain_source` nominally")},
        "accuracy": metrics.top1_accuracy(choice),
        "accuracy_by_family": metrics.breakdown(choice, "family"),
        "ece": metrics.ece(choice),
        "axis1_panel_adequacy": adequacy,
        "noul_rows": sum(1 for row in rows if row["type"] == "noul"),
        "noul_accuracy": (metrics.top1_accuracy([row for row in rows if row["type"] == "noul"])
                          if any(row["type"] == "noul" for row in rows) else None),
        "jevs_dump": {"path": "reports/jev-smoke/axis1-rows.jsonl", "sha256": args.jev_dump_digest},
    }
    write_json(out / "dump.json", report)
    print(f"axis-1 dump: {dump['rows']} choice rows -> {target} (sha256 {dump['sha256'][:16]}...)")
    print(f"  schema: {'exact' if report['dump']['schema_ok'] else 'MISMATCH'} "
          f"(extra {report['dump']['keys_extra']}, missing {report['dump']['keys_missing']}); "
          f"positive column agrees with axis1_label: {positive_agrees}")
    print(f"  accuracy {report['accuracy']['accuracy']:.4f} (n={report['accuracy']['n']}); "
          f"ece {report['ece']['ece']:.4f}; noul rows {report['noul_rows']} "
          f"(accuracy {report['noul_accuracy']})")
    for family, entry in sorted(report["accuracy_by_family"].items()):
        print(f"    family {family:<20} n={entry['n']:<5d} accuracy {entry['accuracy']:.4f} ece {entry['ece']:.4f}")
    primary = adequacy.get("primary")
    if primary is None:
        print(f"  axis 1 PRIMARY n/a -- {adequacy['primary_note']}")
    else:
        print(f"  axis 1 PRIMARY auroc {primary['auroc']:.4f} [{primary['selection']}; n={primary['n']} "
              f"pos={primary['n_positive']} neg={primary['n_negative']}]; at FPR {primary['target_fpr']:.2f} "
              f"detects {primary['detection_rate']:.4f}")
    for name in (metrics.FACTORY, metrics.INJECTED):
        arm = adequacy["by_serving_convention"][name]
        auroc = "n/a" if arm.get("auroc") is None else f"{arm['auroc']:.4f}"
        # `.get`, because an EMPTY selection has no mean to report: `metrics._arm` returns early with
        # `note: "no rows"` and never sets `mean_score`. Indexing it directly raised KeyError and made
        # the whole dump exit 1 on a corpus whose injected arm is legitimately empty (this one: the
        # factory writes an abstain option only on the sibling-panel path), while the dump file itself
        # had already been written correctly -- a non-zero exit that reads as a broken dump.
        mean = arm.get("mean_score")
        print(f"    arm {name:<9} n={arm['n']:<5d} pos={arm['n_positive']:<5d} negs={arm['n_negative']:<5d} "
              f"auroc {auroc} mean P(abstain) "
              f"{'n/a' if mean is None else round(mean, 4)}"
              + (f"  [{arm['note']}]" if arm.get("note") else ""))
    for family, entry in sorted(adequacy["per_family"].items()):
        auroc = "n/a" if entry.get("auroc") is None else f"{entry['auroc']:.4f}"
        print(f"    family {family:<20} n={entry['n']:<5d} pos={entry['n_positive']:<5d} "
              f"neg={entry['n_negative']:<5d} auroc {auroc}")
    print(f"wrote {out / 'dump.json'}")
    return 0


# --------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scripts/kev/kev_probe.py", description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default="reports/jev-full/sample.json",
                        help="the Jev population's sample -- the census of every eligible cell, which is the "
                             "population every rung of the ladder is scored on (default: %(default)s)")
    parser.add_argument("--corpus", default="runs/real-v2", help="corpus directory (default: %(default)s)")
    parser.add_argument("--jev-rows", default="reports/jev-full/rows.jsonl",
                        help="the Jev run's rows for that same population, for the cross-check (default: %(default)s)")
    parser.add_argument("--hub", default=None, help="HF cache hub dir (default: $HF_HOME/hub or ~/.cache/huggingface/hub)")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="rebuild the population and check the encoder (no GPU)")
    plan.add_argument("--out", required=True)
    plan.add_argument("--run", default="jaredpalmer/kev-9b")
    plan.add_argument("--context", default="serving", choices=("serving", "training"))
    plan.add_argument("--limit", type=int, default=0)
    plan.add_argument("--allow-other-corpus", action="store_true",
                      help="do NOT: proceed when the partition is not the file the sample was drawn from")
    plan.set_defaults(func=cmd_plan)

    ask = sub.add_parser("ask", help="answer the planned asks with a Kev checkpoint")
    ask.add_argument("--out", required=True)
    ask.add_argument("--run", default="jaredpalmer/kev-9b")
    ask.add_argument("--device", default="cuda")
    ask.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ask.add_argument("--context", default="serving", choices=("serving", "training"))
    ask.add_argument("--temperature", type=float, default=None,
                     help="None = the temperature the checkpoint carries (the serving default)")
    ask.add_argument("--lora-scale", type=float, default=1.0)
    ask.add_argument("--gpus", type=int, default=1,
                     help="CUDA devices the weights may be sharded over -- a FULL-WEIGHT checkpoint only "
                          "(kev-27b in bf16 is ~51 GB and does not fit one A100-40G); a LoRA arm refuses "
                          "anything but 1 rather than running on one card and reporting more "
                          "(default: %(default)s)")
    ask.add_argument("--reserve-gib", type=float, default=2.0,
                     help="GiB left free on each GPU for activations when the device map is planned "
                          "(default: %(default)s)")
    ask.add_argument("--limit", type=int, default=0)
    ask.add_argument("--progress-every", type=int, default=25)
    ask.set_defaults(func=cmd_ask)

    dump = sub.add_parser("dump", help="project rows.jsonl into the axis-1 dump (no GPU)")
    dump.add_argument("--out", required=True)
    dump.add_argument("--dump", default="reports/axis1-rows-kev.jsonl")
    dump.add_argument("--fpr", type=float, default=0.05)
    dump.add_argument("--jev-dump-digest", default=None)
    dump.set_defaults(func=cmd_dump)
    return parser.parse_args(argv)


def hub_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    import os

    home = os.environ.get("HF_HOME")
    return (Path(home) / "hub") if home else (Path.home() / ".cache" / "huggingface" / "hub")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        args.hub = str(hub_dir(args.hub))
    try:
        return args.func(args)
    except (P.ProbeError, KevError, ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(f"kev probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

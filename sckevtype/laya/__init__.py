"""The laya pilot: re-render the frozen suites for a different backbone, and score the result.

This package is **not** on the paper's path. It exists to answer one question, pre-registered in
`docs/SPEC-laya-finetune.md`: can a fine-tuned `laya` beat the current system on the paper's own
instrument? :mod:`sckevtype.laya.render` re-renders a frozen suite into laya's input format at a
disclosed configuration; :mod:`sckevtype.laya.score` turns laya's answers back into the *same*
row schema :mod:`sckevtype.eval` scores, so the comparison is row for row.

Nothing here imports the rest of the package at import time, and nothing in the rest of the
package imports this: deleting the directory would leave the paper's code path untouched.
"""

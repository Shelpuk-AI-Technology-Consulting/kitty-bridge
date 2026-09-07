"""AST scan for kitty-internal metadata keys written into dicts.

Kitty's translators and adapters pass metadata to each other through
underscore-prefixed keys on the Chat Completions request dict.  Every one of
them must be a member of :attr:`kitty.providers.base.ProviderAdapter._INTERNAL_KEYS`,
because that frozenset is the only thing standing between those keys and the
provider's wire — see ``.system_design/TEST_SUITE.md`` §6.2.3 and KBR-6.

This module owns the scan.  It is imported by both
``tests/test_internal_key_completeness.py`` (which asserts the set is complete)
and ``tests/test_internal_keys_not_sent_upstream.py`` (which builds its input
from the keys found here), so the two tests cannot drift apart: a key one of
them learns about, the other exercises.

**The scan is deliberately over-approximate.**  No static analysis can tell a
Chat Completions body from any other dict, so every underscore-prefixed key
written into any dict is reported.  The scan must never filter by the target
variable's name — restricting it to targets called ``cc_request`` or ``body``
would miss a leak written into ``payload["_x"] = 1``.  The single escape hatch
is :data:`_REQUEST_ANNOTATIONS`; adding a non-body key to ``_INTERNAL_KEYS``
to silence the scan is forbidden and is explained in the design document.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "kitty"

#: Packages scanned.  ``providers`` is included because
#: ``ProviderAdapter.normalize_request`` mutates the request dict in place on
#: the live serving path, so it mints internal keys just as the translators do.
SCANNED_PACKAGES = ("bridge", "providers")

#: Annotations identifying an aiohttp request object rather than a request body.
#:
#: ``BridgeServer._auth_middleware`` stores ``_key_id``, ``_profile_name`` and
#: ``_mapped_profile`` on the **inbound** ``web.Request``, which aiohttp
#: supports as a request-scoped mapping (verified against aiohttp 3.13.5).
#: Those never reach a provider.  The exclusion is keyed on the annotation and
#: never on the variable's name, because this codebase uses ``request`` for
#: both kinds of object and a name-based rule would excuse a genuine leak.
_REQUEST_ANNOTATIONS = frozenset({"web.Request", "Request", "web.BaseRequest", "BaseRequest"})


@dataclass(frozen=True)
class KeyWrite:
    """One underscore-prefixed key written into a dict.

    Attributes:
        path: Path of the source file, relative to ``src/kitty``, posix-style.
        lineno: 1-based line number of the writing statement.
        key: The key written, including its leading underscore.
        form: Which AST form produced it — one of ``"subscript-assign"``,
            ``"dict-literal"``, ``"setdefault"`` or ``"dict-kwarg"``.
    """

    path: str
    lineno: int
    key: str
    form: str

    def __str__(self) -> str:
        """Render as ``path:lineno key (form)`` for assertion messages.

        Returns:
            A single-line description naming the file, line, key and form.
        """
        return f"{self.path}:{self.lineno} {self.key} ({self.form})"


def _annotation_name(node: ast.expr | None) -> str | None:
    """Render an annotation expression as a dotted name.

    Args:
        node: The annotation AST node, or ``None`` when unannotated.

    Returns:
        The dotted name (e.g. ``"web.Request"``), or ``None`` when the
        annotation is not a plain name, attribute or string literal.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    # String annotations (``def f(r: "web.Request")``) resolve to the literal.
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _InternalKeyVisitor(ast.NodeVisitor):
    """Collect underscore-prefixed dict-key writes from one module."""

    def __init__(self, path: str) -> None:
        """Initialise the visitor for one source file.

        Args:
            path: Path of the file, relative to ``src/kitty``, used in results.
        """
        self.path = path
        self.writes: list[KeyWrite] = []
        #: Names bound to an aiohttp request object, innermost scope last.
        #: A list of cumulative frozensets, so a closure writing to an outer
        #: scope's request parameter is excluded too.
        self._request_names: list[frozenset[str]] = [frozenset()]
        #: Sites the request-object exclusion actually suppressed.  Asserted
        #: non-empty by the guard, so the exclusion cannot silently go blind.
        self.excluded: list[KeyWrite] = []

    # ── scope tracking ──────────────────────────────────────────────────

    def _enter_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Push a scope carrying the request-annotated parameters of *node*.

        Args:
            node: The function whose signature is being entered.
        """
        args = node.args
        names = {
            arg.arg
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
            if _annotation_name(arg.annotation) in _REQUEST_ANNOTATIONS
        }
        self._request_names.append(self._request_names[-1] | names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track request-annotated parameters across a function body.

        Args:
            node: The function definition being visited.
        """
        self._enter_scope(node)
        self.generic_visit(node)
        self._request_names.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track request-annotated parameters across an async function body.

        Args:
            node: The async function definition being visited.
        """
        self._enter_scope(node)
        self.generic_visit(node)
        self._request_names.pop()

    # ── recording ───────────────────────────────────────────────────────

    def _is_request_object(self, node: ast.expr) -> bool:
        """Report whether *node* names an aiohttp request object.

        Args:
            node: The expression being subscripted or called.

        Returns:
            True when it is a bare name bound to a request-annotated parameter.
        """
        return isinstance(node, ast.Name) and node.id in self._request_names[-1]

    def _record(self, key: object, lineno: int, form: str, *, target: ast.expr | None = None) -> None:
        """Record *key* when it is an internal-key write worth reporting.

        Args:
            key: The candidate key; ignored unless it is a ``_``-prefixed string.
            lineno: Line number of the writing statement.
            form: The AST form that produced it.
            target: The object written to, when there is one, so writes onto an
                aiohttp request object can be excluded.
        """
        if not isinstance(key, str) or not key.startswith("_"):
            return
        write = KeyWrite(self.path, lineno, key, form)
        # Writes onto the inbound request object are request-scoped storage,
        # not a request body, and never reach a provider.
        if target is not None and self._is_request_object(target):
            self.excluded.append(write)
            return
        self.writes.append(write)

    # ── the covered forms ───────────────────────────────────────────────

    def _visit_subscript_target(self, target: ast.expr, lineno: int) -> None:
        """Record a subscript-assignment target with a constant key.

        Args:
            target: The assignment target.
            lineno: Line number of the assignment.
        """
        if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
            self._record(target.slice.value, lineno, "subscript-assign", target=target.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Record ``obj["_key"] = value``.

        Args:
            node: The assignment statement.
        """
        for target in node.targets:
            self._visit_subscript_target(target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Record ``obj["_key"] += value``.

        Args:
            node: The augmented assignment statement.
        """
        self._visit_subscript_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Record ``obj["_key"]: T = value``.

        Args:
            node: The annotated assignment statement.
        """
        self._visit_subscript_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """Record ``{"_key": value}``.

        This also covers ``obj.update({"_key": value})`` and ``{**base,
        "_key": value}``, since each contains a dict literal.  A ``.update()``
        of a computed mapping is a stated limitation of the scan.

        Args:
            node: The dict literal.
        """
        for key in node.keys:
            if isinstance(key, ast.Constant):
                self._record(key.value, node.lineno, "dict-literal")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Record ``obj.setdefault("_key", v)`` and ``dict(obj, _key=v)``.

        Args:
            node: The call expression.
        """
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "setdefault" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant):
                self._record(first.value, node.lineno, "setdefault", target=func.value)
        if isinstance(func, ast.Name) and func.id == "dict":
            for keyword in node.keywords:
                if keyword.arg is not None:
                    self._record(keyword.arg, node.lineno, "dict-kwarg")
        self.generic_visit(node)


def scan_source(source: str, path: str = "<synthetic>") -> tuple[list[KeyWrite], list[KeyWrite]]:
    """Scan one module's source for internal-key writes.

    Args:
        source: Python source text.
        path: Name reported in the results.

    Returns:
        A ``(writes, excluded)`` pair — the keys reported, and the keys
        suppressed because they were written onto an aiohttp request object.
    """
    visitor = _InternalKeyVisitor(path)
    visitor.visit(ast.parse(source, filename=path))
    return visitor.writes, visitor.excluded


def scan_tree() -> tuple[list[KeyWrite], list[KeyWrite]]:
    """Scan every module in the packages named by :data:`SCANNED_PACKAGES`.

    Returns:
        A ``(writes, excluded)`` pair covering the whole scanned tree, ordered
        by file then line.
    """
    writes: list[KeyWrite] = []
    excluded: list[KeyWrite] = []
    for package in SCANNED_PACKAGES:
        for file in sorted((SRC / package).rglob("*.py")):
            rel = file.relative_to(SRC).as_posix()
            found, skipped = scan_source(file.read_text(encoding="utf-8"), rel)
            writes.extend(found)
            excluded.extend(skipped)
    return writes, excluded


def discovered_keys() -> frozenset[str]:
    """Return every internal key the source is seen to write.

    Used by ``tests/test_internal_keys_not_sent_upstream.py`` to build its
    input, so a newly minted key is exercised behaviourally the moment the
    scan discovers it.

    Returns:
        The set of underscore-prefixed keys found anywhere in the scanned tree.
    """
    writes, _excluded = scan_tree()
    return frozenset(write.key for write in writes)

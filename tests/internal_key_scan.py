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

**What it does not see.**  A key whose name is not a literal at the write site:
``cc[prefix + name] = v``, ``cc[_SOME_CONSTANT] = v`` where the constant is
defined elsewhere, ``cc.update(other)`` and ``{**computed}`` for a mapping built
at run time, and ``cc.__setitem__("_x", v)``.  The middle one is the realistic
gap — a module-level ``_RESOLVED_KEY = "_resolved_key"`` would go unseen — and it
is left unresolved rather than half-solved, because a scan that follows some
constants and not others invites more confidence than it earns.  Every form the
scan *does* claim has a falsification case in
``tests/test_internal_key_completeness.py``.

**Import note.**  Both test modules import this one as ``internal_key_scan``,
which works because pytest puts each test file's own directory on ``sys.path``.
Adding a ``tests/__init__.py`` would turn ``tests`` into a package and break
that; the import would then need to be ``tests.internal_key_scan``.
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
            ``"dict-literal"``, ``"setdefault"``, ``"update-kwarg"`` or
            ``"dict-kwarg"``.
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


def _rebound_names(scope: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    """Return every name *scope* binds anywhere in its body.

    The request-object exclusion is seeded from a parameter annotation but
    applied to a name, so it must not survive the name being rebound:
    ``request = await request.json()`` leaves ``request`` holding a request
    *body*, and going on excluding writes to it is precisely the name-keyed
    rule the design forbids.

    **Decided per scope, not per statement.** Four review rounds chased this
    one statement form at a time — annotated assignment, then ``async for`` and
    ``async with``, then augmented assignment and ``except``/``import`` aliases
    and tuple targets, then ``match`` captures and ``class``. Enumerating
    binding *statements* cannot terminate, because Python keeps having more of
    them. Enumerating binding *node kinds* does terminate: a plain name is bound
    by a ``Name`` in a ``Store``/``Del`` context, and everything else that binds
    carries the name as a bare string on a handful of node types.

    So the rule is now coarse and total: **if a scope rebinds the name at all,
    the annotation is not trusted for that scope.** A write before the rebinding
    is reported too. That is the safe direction — over-reporting costs a review
    comment, under-reporting hides a leak — and it is why this is not
    flow-sensitive.

    ``Subscript`` and ``Attribute`` targets are not bindings: in
    ``request["_key_id"] = ...`` the ``Name`` node carries a ``Load`` context, so
    it is excluded here automatically rather than by a special case. That is the
    behaviour the whole exclusion rests on.

    Args:
        scope: The function or lambda whose body is inspected. Nested scopes are
            included, which is deliberate: a closure rebinding the enclosing
            name makes it untrustworthy in both.

    Returns:
        Every name bound somewhere inside *scope*.
    """
    bound: set[str] = set()
    for node in ast.walk(scope):
        # Assignment, for, with-as, walrus, tuple/starred targets, comprehensions
        # and match captures that bind a plain name all land here.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bound.add(node.id)
        # The rest carry their bound name as a string rather than a Name node.
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            bound.add(node.name)
    # Parameter declarations are deliberately absent: an ``ast.arg`` is not a
    # ``Name``, so a parameter reaches this set only by being *reassigned* in the
    # body — which is exactly the condition that should disqualify it. A nested
    # function re-declaring the name is handled by the caller's scope
    # subtraction instead, because that shadows rather than rebinds.
    return bound


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

    def _enter_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        """Push a scope carrying the request-annotated parameters of *node*.

        A parameter the inner function declares itself is **subtracted** from
        what it inherits, even when an enclosing scope annotated the same name.
        Without that, an inner ``def inner(request)`` taking an ordinary dict
        would inherit the outer handler's exclusion and hide a real leak — which
        is the name-keyed behaviour this exclusion exists to avoid.

        Args:
            node: The function or lambda whose signature is being entered.
        """
        args = node.args
        declared = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        if args.vararg:
            declared.add(args.vararg.arg)
        if args.kwarg:
            declared.add(args.kwarg.arg)
        annotated = {
            arg.arg
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
            if _annotation_name(arg.annotation) in _REQUEST_ANNOTATIONS
        }
        # A parameter rebound anywhere in this scope is not trusted at all — see
        # _rebound_names for why this is decided per scope rather than per
        # statement.
        trusted = annotated - _rebound_names(node)
        self._request_names.append((self._request_names[-1] - declared) | trusted)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Stop a lambda parameter from inheriting an enclosing exclusion.

        Args:
            node: The lambda being visited.
        """
        self._enter_scope(node)
        self.generic_visit(node)
        self._request_names.pop()

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
            # ``cc["_x"], other = 1, 2`` — a subscript can hide inside a tuple.
            if isinstance(target, ast.Tuple | ast.List):
                for element in target.elts:
                    self._visit_subscript_target(element, node.lineno)
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
        """Record ``setdefault``, ``update(**kwargs)`` and ``dict(_key=v)`` writes.

        ``obj.update(_effort=x)`` is idiomatic — arguably more so than several
        separate subscript assignments — and carries no dict literal for
        :meth:`visit_Dict` to catch, so it needs a rule of its own.

        Args:
            node: The call expression.
        """
        func = node.func
        if isinstance(func, ast.Attribute) and node.args and func.attr == "setdefault":
            first = node.args[0]
            if isinstance(first, ast.Constant):
                self._record(first.value, node.lineno, "setdefault", target=func.value)
        if isinstance(func, ast.Attribute) and func.attr == "update":
            for keyword in node.keywords:
                if keyword.arg is not None:
                    self._record(keyword.arg, node.lineno, "update-kwarg", target=func.value)
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

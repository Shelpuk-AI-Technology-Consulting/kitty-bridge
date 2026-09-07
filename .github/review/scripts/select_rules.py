"""Select which review rule files apply to a pull request's changed files.

Claude Code has no equivalent of GitHub Copilot's ``applyTo`` frontmatter glob,
so this module reproduces it: given the list of files a pull request touches, it
decides which rule files under ``.github/review/rules/`` belong in the review
prompt.

Adopted from ``kindly-web-search-mcp-server``, where this system is in
production. The selector mechanics -- :func:`_matches`, :func:`select`,
:func:`resolve_paths` and the command-line entry point -- are carried across
unchanged so a fix made there can be copied here. :data:`RULE_SPECS` is the half
that is ours: it is this repository's component layout, including the fan-out
rule that makes a shared module's change pull in the rules of everything that
imports it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass
from pathlib import Path

RULES_DIR = Path(".github/review/rules")


@dataclass(frozen=True)
class RuleSpec:
    """A review rule file and the paths that activate it.

    Attributes:
        name: Rule file stem, matching a file in :data:`RULES_DIR`.
        patterns: Glob patterns; any match activates the rule.
        pulls_in: Other rule names activated whenever this one is, used to model
            dependency fan-out rather than mere directory membership.
    """

    name: str
    patterns: tuple[str, ...]
    pulls_in: tuple[str, ...] = ()


#: The package root, named once. Every source pattern below is anchored on it, and
#: writing it out fifteen times is how a rename leaves half the selector matching
#: nothing while the other half still fires -- a partial selection is worse than
#: none, because the review looks rule-driven and is not.
PKG = "src/kitty"


# Ordered most general to most specific; ordering controls prompt assembly order.
RULE_SPECS: tuple[RuleSpec, ...] = (
    # `types.py` and `validation.py` are imported by every layer -- the bridge
    # builds its responses out of these types, the providers return them, the
    # launchers and the CLI validate their inputs through them. So a change here
    # IS a change to each of those components, and scoping strictly by directory
    # would let a shared dataclass field or a validation predicate flip ship with
    # none of its consumers' rules applied.
    #
    # `__init__.py` and `__main__.py` are here because neither is reachable by any
    # other pattern: the first decides the package's importable surface, the
    # second is what `python -m kitty` runs.
    RuleSpec(
        name="core",
        patterns=(
            f"{PKG}/types.py",
            f"{PKG}/validation.py",
            f"{PKG}/__init__.py",
            f"{PKG}/__main__.py",
        ),
        pulls_in=(
            "bridge",
            "providers",
            "launchers",
            "egress",
            "credentials",
            "cli",
        ),
    ),
    # The local HTTP server and the three protocol translators (Anthropic
    # Messages, OpenAI Responses, Gemini) plus the shared engine, state and
    # tool-use auditor. This is where the product's central invariant lives: the
    # agent's messages cross this component and must come out the other side
    # unchanged except where translation makes a change unavoidable.
    #
    # `bridge_runner.py` is the same component from the process side -- it is what
    # the launcher starts -- and `cloudflare.py` is the upstream-edge handling the
    # server branches on, so neither belongs anywhere else.
    RuleSpec(
        name="bridge",
        patterns=(
            f"{PKG}/bridge/**/*.py",
            f"{PKG}/bridge_runner.py",
            f"{PKG}/cloudflare.py",
        ),
    ),
    # The provider adapters: upstream request build, auth, headers, error mapping,
    # and the model-context catalogue. One rule rather than twenty-odd because
    # they share a single contract -- turn the bridge's Chat Completions request
    # into this provider's dialect, and map its errors back -- and because the
    # registry is the proof that the set stays consistent.
    RuleSpec(
        name="providers",
        patterns=(
            f"{PKG}/providers/**",
            # The catalogue generator. It writes `providers/model_metadata.json`
            # and `model-metadata.yml` runs it weekly; a change to it is a
            # change to the provider component from the build side, and no other
            # pattern reaches `scripts/`.
            "scripts/**",
        ),
    ),
    # The launcher adapters -- the contract with the agent process. Each one
    # decides the environment variables and CLI flags the child is started with,
    # which is the single most security-relevant surface in the repository after
    # the credential store: get it wrong and the agent talks to the provider
    # directly, or with the wrong credential, or with the user's real key visible
    # in a process listing.
    #
    # `cli/launcher.py` is deliberately here as well as in `cli`: it is the
    # orchestrator that consumes a `SpawnConfig` and spawns the child, so a change
    # to it is a change to this contract even though it lives under `cli/`.
    RuleSpec(
        name="launchers",
        patterns=(
            f"{PKG}/launchers/**/*.py",
            f"{PKG}/cli/launcher.py",
        ),
    ),
    # The egress gateway: resolution, the fail-closed guard, and the on-disk
    # store. Its own rule rather than folded into `core`, because it is the only
    # component whose failure mode is *silent success* -- traffic that should have
    # been proxied and simply was not. Nothing downstream notices.
    RuleSpec(
        name="egress",
        patterns=(
            f"{PKG}/egress.py",
            f"{PKG}/egress_guard.py",
            f"{PKG}/egress_store.py",
        ),
    ),
    # Everything that holds or resolves a secret, plus the profiles that bind one:
    # the credential store and its two backends, the OAuth/PKCE flows, and the
    # profile schema, store and resolver. One rule because a profile's `auth_ref`
    # is meaningless without the store it dereferences, and a change to either
    # side of that pairing has to be judged against the other.
    RuleSpec(
        name="credentials",
        patterns=(
            f"{PKG}/credentials/**/*.py",
            f"{PKG}/auth/**/*.py",
            f"{PKG}/profiles/**/*.py",
        ),
    ),
    # The command surface: the router that decides what `kitty <word> ...` means,
    # the built-in commands, and the interactive TUI those commands open. The TUI
    # is here rather than in its own rule because it has no behaviour of its own --
    # it is how the commands ask their questions.
    RuleSpec(
        name="cli",
        patterns=(
            f"{PKG}/cli/**/*.py",
            f"{PKG}/tui/**/*.py",
        ),
    ),
    RuleSpec(
        name="python-tests",
        patterns=(
            "tests/**",
            # The review workflow's own tests are tests too. Omitting them would
            # mean the file guarding this very selector is reviewed without the
            # test rules applied.
            ".github/review/tests/**/*.py",
        ),
    ),
    # How the package is built, typed, linted and installed. `pyproject.toml` is
    # not bookkeeping here: it carries the three import-linter contracts that are
    # the only enforcement of the module layering, the dependency bounds, the
    # `requires-python` floor the CI matrix must track, and the single
    # `[project.scripts]` entry point users have typed into their shells. A bound
    # or a contract edited here changes what every user's next install resolves,
    # which is why this rule pulls in `cli` -- the entry point is that component's.
    #
    # 🔴 `.gitignore` is here, and it is not filler. It excludes `.requirements/`
    # and `CLAUDE.md` -- and deliberately does NOT exclude `.system_design/` --
    # so it is what decides which documents reach a CI checkout, and therefore
    # what the reviewer in `claude-code-review.yml` is able to read at all. A
    # line added or removed there silently widens or narrows every future
    # review; re-ignoring `.system_design/` would blind the reviewer to the
    # internal specification without failing anything. It was also the only
    # tracked file this selector matched with no rule, which is the exact
    # zero-rules-loaded shape the docstring above warns about.
    RuleSpec(
        name="packaging",
        patterns=(
            "pyproject.toml",
            ".gitignore",
        ),
        pulls_in=("cli",),
    ),
    # Everything under .github: the review workflow, the CI and publish workflows,
    # these scripts, the prompts, the schema and the rule files themselves.
    RuleSpec(
        name="ci",
        patterns=(".github/**",),
    ),
    # README.md is not documentation *about* this repository, it is the closest
    # committed thing it has to a specification: the command contract, the
    # provider and launcher tables, the egress and profile behaviour, and the
    # troubleshooting that tells a user what a failure means. A change to
    # behaviour that does not move it is drift.
    #
    # `.system_design/` is tracked and reaches a CI checkout, so the reviewer
    # reads it: it is the internal specification, as README.md is the external
    # one. These patterns were added before the directory was un-ignored, on the
    # reasoning that a design document landing with no rule file selected is
    # exactly the silent gap this selector exists to prevent -- and they were
    # already correct when it landed.
    #
    # `.requirements/` and `CLAUDE.md` are still excluded by `.gitignore`, so
    # nothing at those paths reaches a checkout. Their patterns stay for the same
    # reason: they cost nothing while the paths are absent, and they are right
    # the day either is committed.
    #
    # ⚠️ Do not read `.requirements/`'s presence here as evidence the reviewer can
    # read a per-task REQUIREMENTS.md -- it cannot, and `REVIEW_GUIDE.md` says so.
    RuleSpec(
        name="docs",
        patterns=(
            "README.md",
            "LICENSE",
            "CLAUDE.md",
            "AGENTS.md",
            "**/CLAUDE.md",
            "**/AGENTS.md",
            ".system_design/**/*.md",
            "**/.system_design/**/*.md",
            ".requirements/**/*.md",
            # The images README renders. A broken or replaced asset is a
            # documentation defect and nothing else selects it.
            "assets/**",
        ),
    ),
)


def _matches(path: str, pattern: str) -> bool:
    """Report whether a repository path matches a glob pattern.

    ``fnmatch`` treats ``*`` as crossing directory separators, which would make
    ``src/kitty/bridge/**`` also match a sibling whose name merely starts with
    ``bridge``. The prefix check below keeps sibling directories disjoint.

    Args:
        path: Repository-relative path, forward-slashed.
        pattern: Glob pattern from a :class:`RuleSpec`.

    Returns:
        True when the path is covered by the pattern.
    """

    # Anchor on the literal prefix before the first wildcard so that
    # "bridge/**" cannot leak into a directory sharing that prefix.
    head = pattern.split("*", 1)[0]
    if head and not path.startswith(head):
        return False
    return fnmatch.fnmatch(path, pattern.replace("**/", "*"))


def select(changed_files: list[str]) -> list[str]:
    """Choose the rule files that apply to a set of changed paths.

    Args:
        changed_files: Repository-relative paths changed by the pull request.

    Returns:
        Rule names in :data:`RULE_SPECS` order, de-duplicated, including any
        pulled in by the fan-out rule.
    """

    selected: set[str] = set()

    # Direct matches first, then fan-out, so a shared-module-only change still
    # activates its consumers even though no file under them changed.
    for spec in RULE_SPECS:
        if any(_matches(p, pat) for p in changed_files for pat in spec.patterns):
            selected.add(spec.name)
            selected.update(spec.pulls_in)

    return [spec.name for spec in RULE_SPECS if spec.name in selected]


def resolve_paths(names: list[str], rules_dir: Path = RULES_DIR) -> list[Path]:
    """Map rule names to existing rule files.

    Args:
        names: Rule names returned by :func:`select`.
        rules_dir: Directory holding the rule Markdown files.

    Returns:
        Paths that exist on disk, in the order given.
    """

    resolved = []
    for name in names:
        candidate = rules_dir / f"{name}.md"
        if candidate.is_file():
            resolved.append(candidate)
        else:
            print(f"warning: rule file missing: {candidate}", file=sys.stderr)
    return resolved


def main() -> int:
    """Run the selector as a command-line tool.

    Reads changed paths from a file or stdin and writes the selected rule names
    and paths, optionally appending them to a GitHub Actions output file.

    Returns:
        Process exit code; always 0 because an empty selection is valid.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changed-files",
        help="File containing one changed path per line; reads stdin when omitted.",
    )
    parser.add_argument("--rules-dir", default=str(RULES_DIR))
    parser.add_argument("--github-output", help="Path to $GITHUB_OUTPUT.")
    args = parser.parse_args()

    raw = (
        Path(args.changed_files).read_text(encoding="utf-8")
        if args.changed_files
        else sys.stdin.read()
    )
    changed = [
        line.strip().replace("\\", "/") for line in raw.splitlines() if line.strip()
    ]

    names = select(changed)
    paths = resolve_paths(names, Path(args.rules_dir))

    print(f"changed files: {len(changed)}")
    print(f"selected rules: {', '.join(names) if names else '(none)'}")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(f"rule_names={json.dumps(names)}\n")
            handle.write(f"rule_paths={' '.join(str(p) for p in paths)}\n")
            handle.write(f"rule_count={len(paths)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Reserve namespaces (VF-10 follow-up)

Checklist for a human with publish credentials. **Nothing in this file has been run.**
No agent should reserve or publish any of these — this is a runbook, not an action log.

## Why

VF-10 found that several READMEs told users to `pip install <name>` / `npx <name>` for
names that are either:

- already registered to an **unrelated stranger's package** (dependency-confusion risk —
  a user who runs the documented command gets someone else's code), or
- **unclaimed** (404) — the command just fails, but it also leaves the name sitting
  open for a stranger to squat on it later.

The docs have been fixed to build-from-source instead (see the `fix/vf-10-honest-install-docs`
branches in `robit`, `agent`, `cyclops`, `myrmex`). This file is the follow-up: actually
claim the names so a future "let's publish for real" pass has somewhere safe to publish to,
and so no one else can squat them first.

## Current state (verified 2026-08-17)

| Registry | Name | Status | Notes |
|---|---|---|---|
| PyPI | `enchanter-golem` | **Claimed by us** | Live, shipped v0.1.0.0 (2026-06-13). The only real one so far. |
| PyPI | `robit` | Stranger-owned | "Chronological Automation Service Framework" (stratusadv). Do not use. |
| PyPI | `cyclops` | Stranger-owned | Sentry gateway (heynemann). Do not use. |
| PyPI | `enchanter-agent` | Unclaimed (404) | `robit/pyproject.toml` already declares this name — just not published. |
| PyPI | `enchanter-robit` | Unclaimed (404) | Recommended replacement for the `agent/` repo, whose `pyproject.toml` currently declares the collision-prone name `robit` — rename before publishing. |
| PyPI | `enchanter-cyclops` | Unclaimed (404) | Recommended replacement for `cyclops/pyproject.toml`'s current name `cyclops` — rename before publishing. |
| npm | `enchanter` (bare) | Stranger-owned | Confirmed in `beholder/README.md`'s existing fix. `beholder/package.json` currently declares this name — rename before publishing (e.g. `@enchanter-ai/beholder`). |
| npm | `@enchanter-ai/myrmex` | Unclaimed (404) | Already the declared name in `myrmex/package.json` — just not published. |
| npm scope | `@enchanter-ai` | Not yet created | Reserve the scope itself first; it gates every `@enchanter-ai/*` package below. |

## Step 1 — reserve the npm scope

Requires an npm org named `enchanter-ai` (or a user account that owns the scope).

```sh
npm login
npm org create enchanter-ai        # or: npm team create enchanter-ai:developers (if org already exists)
```

## Step 2 — reserve/publish each npm package under the scope

Publishing an empty/placeholder `0.0.0` release is enough to claim the name; a real
release can follow later.

```sh
# myrmex (package.json already declares @enchanter-ai/myrmex)
cd myrmex
npm run build
npm publish --access public

# beholder — rename package.json "name" to "@enchanter-ai/beholder" first,
# the current name "enchanter" collides with a stranger's package.
cd beholder
npm publish --access public
```

## Step 3 — reserve each PyPI name

Requires a PyPI account with 2FA and an API token (or use trusted publishing / OIDC —
see how `enchanter-golem` is configured for the pattern already in use).

```sh
python -m pip install --upgrade build twine

# enchanter-agent (robit/pyproject.toml already declares this name)
cd robit
python -m build
twine upload dist/*

# enchanter-robit — rename agent/pyproject.toml "name" from "robit" to
# "enchanter-robit" first, then:
cd agent
python -m build
twine upload dist/*

# enchanter-cyclops — rename cyclops/pyproject.toml "name" from "cyclops"
# to "enchanter-cyclops" first, then:
cd cyclops
python -m build
twine upload dist/*
```

## Step 4 — after reserving

- Update each repo's README "Install" section to point at the real published name
  once it's live (reverting the build-from-source note added by VF-10).
- Cross-check `shared/models-registry.json`-style source-of-truth docs in `wixie/`
  don't reference the old collision-prone names.
- Re-run the VF-10 registry-verification check (`curl .../pypi/<name>/json`,
  `curl registry.npmjs.org/<name>`) to confirm the claim stuck.

## Explicitly out of scope here

- No pyproject.toml / package.json renames were made by this task — only READMEs.
  The rename recommendations above (`robit`→`enchanter-robit`, `cyclops`→`enchanter-cyclops`,
  `enchanter`→`@enchanter-ai/beholder`) are flagged for the human to decide and apply
  before running Step 3/2 for those three.
- No credentials, tokens, or `npm`/`twine` commands were executed by this task.

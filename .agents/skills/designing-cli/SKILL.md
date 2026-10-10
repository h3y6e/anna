---
compatibility: Requires Node.js, and the usage CLI to probe with a spec.
description: Designs CLIs that humans, scripts, and agents drive predictably. Use when creating a CLI, or adding or changing its commands, flags, output, errors, config, background process, or release.
license: MIT
metadata:
    author: h3y6e
    github-path: skills/designing-cli
    github-ref: refs/tags/v2026.10.1
    github-repo: https://github.com/h3y6e/agent-skills
    github-tree-sha: af022f1b2957cf1f845867e3d0a5971412e086a1
    version: 2026.10.1
name: designing-cli
---
# Designing CLIs

## Workflow

1. **Start.** For a new CLI, confirm the language and whether it ships as a binary.
2. **Define** commands, flags, help, examples, and effects in one framework definition: cobra, viper, and `github.com/jdx/usage/integrations/cobra` for Go; gunshi v1 with `@gunshi/plugin-suggestion`, `@h3y6e/gunshi-plugin-env`, and `@h3y6e/gunshi-plugin-usage` for TypeScript; usage-rs for Rust; usage `#USAGE` comments for shell.
3. **Classify** each item below as pass, n/a, or deviation in a table, citing a probe line, command run, or code read. Items without a subject are n/a. A deviation needs the user's reason, or a fix when reviewing; check [references/rationale.md](references/rationale.md) first.
4. **Probe**: build, then run `scripts/probe.mjs <bin> [app.usage.kdl]`, passing the spec when the framework exports one. Fix each FAIL; NOTEs are evidence.
5. **Done** when every item is classified and the probe has no FAIL.

## Checklist

### Input

- Subcommands nest with spaces in one consistent order (`app server start`). At most one kind of positional; the rest are flags; child-process arguments follow `--`.
- Bare `app`, `-h`, and `--help` at any level print help to stdout and exit 0.
- `--version` and bare `-v` print the version; `-V` and `version` follow the framework. Only the root parses version flags.
- Flags also read `APP_*` env vars.
- Prompt only on a TTY, with a flag to skip each prompt.

### Output

- Results go to stdout; messages, progress, and child-process stderr go to stderr.
- List and status commands take `--json`; streamed output is JSONL ending in a result record, even on early failure or cancellation.
- Exit quietly on a closed pipe.
- Color and spinners: `--no-color`, then `CLICOLOR_FORCE`, then `NO_COLOR` / `TERM=dumb` / `CLICOLOR=0`, then TTY detection.
- Tables: one entry per line, no borders.

### Errors

- Errors give a stable code, cause, runnable fix, and docs URL if any, as result-record fields under `--json`; `--quiet` never hides them.
- Unknown commands suggest "did you mean"; usage errors point to `--help`.
- Unexpected errors print a summary and the flag that adds a traceback and debug log. Log files are timestamped, truncated, and ANSI-free.
- Exit 0 on success, help, and version; 1 on failure; 2 on usage errors. Help documents other codes.
- On SIGINT or SIGTERM, cancel remote work, clean up, and re-raise the signal.

### Safety

- Each command's effect is `read`, `write`, or `destructive`; help marks destructive ones.
- Destructive commands show what will be deleted, overwritten, or sent, then confirm on a TTY or exit 2 without `--yes`.
- Write and destructive commands take `--dry-run`.
- Secrets never arrive as arguments.

### Files and release

- Separate XDG directories for config, state, data, and cache. With config files, follow [references/config.md](references/config.md).
- When a binary ships, follow [references/distribution.md](references/distribution.md).
- With a TUI or an agent hook, follow [references/situational.md](references/situational.md).

### Speed and compatibility

- Start in 100–500ms, loading only the invoked command.
- No network at startup unless the command needs it; update checks never block and can be disabled.
- Gate unstable features behind an `experimental` setting.
- Version data and wire formats, `--json` included, apart from the CLI; removing, renaming, or retyping a field is breaking.
- For a background process, follow [references/daemon.md](references/daemon.md).

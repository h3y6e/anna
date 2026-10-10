# Rationale

A deviation is justified only when the item's reason does not apply.

## Input

- **Help on `-h` and `--help`, exit 0**: users and agents try these first; a nonzero exit makes `app --help | less` look like a failure, and a bare invocation that acts surprises someone who only wanted to look.
- **Root-only version flags**: a subcommand may take its own `--version <v>`.
- **Bare `-v` prints the version**: tools disagree on `-v`, and users and agents try it for the version; after a subcommand it may mean verbose.
- **Env for flags**: CI and wrappers set env once instead of editing each call.

## Output

- **Color order**: an explicit flag beats env; `CLICOLOR_FORCE` serves pagers and CI logs that render ANSI.

## Errors

- **Exit 0/1/2**: callers tell misuse from failure without parsing text.
- **Re-raise signals**: callers see 130 or 143 and know the run was interrupted; remote jobs left running keep spending.

## Release

- **packslip**: package managers verify binaries against one signed file, serve completions for the active version, and fetch that version's skills; a skill for another version describes commands that do not exist.

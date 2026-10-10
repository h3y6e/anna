# Config

- Uncommitted `app.local.toml` overrides `app.toml`; a child directory's config overrides its parent's; `APP_ENV=<name>` adds `app.<name>.toml`.
- Define each setting once, with its env var and a published JSON Schema; `app settings --json` shows each value's origin: default, file, env, or flag.
- Settings a repository must not override, such as paths that run code or credentials, are global-only.
- Project config that can run code fails until `app trust` records its path and content hash in the state directory; the error names that command.
- Ignore XDG variables holding relative paths.

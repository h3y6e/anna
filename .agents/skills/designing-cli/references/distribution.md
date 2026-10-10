# Release

Publish every release with a signed [packslip](https://packslip.dev) manifest declaring the executables, task skills, and any usage spec.

## Signing

Sign releases in GitHub Actions with the packslip action, which uploads `packslip.sigstore.json`; check [Publish with GitHub Actions](https://packslip.dev/docs/publishing/) for current inputs.

```yaml
permissions:
  contents: write
  id-token: write
  attestations: write

steps:
  - uses: jdx/packslip@<commit-sha> # v1
    with:
      artifacts: dist/*.tar.xz dist/*.zip
      bin: app
      resources: |
        cli-spec/usage=asset:dist/app.usage.kdl
        skill/app-configure=archive:skills/app-configure
```

- Pin the action to a commit SHA; the job holds `id-token` and attestation write permissions.
- Resource lines are `KIND[/QUALIFIER][@os[/arch[/libc]]]=SOURCE:VALUE`; sources are `archive` (path in the artifact), `asset` (separate release file you upload), `repo` (path at the release commit), and `exec` (command output).
- Declare `cli-spec/usage` only when a spec is exported.
- Copy `archive` paths, including any top-level directory, from the archive listing; packslip does not check them.
- Name artifacts with OS, architecture, and, on Linux, `gnu` or `musl`; Linux artifacts without an architecture are rejected.

## Task Skills

- Write one skill per recurring task rather than one for the whole CLI, each a single `SKILL.md` whose trigger is scoped to projects that use the CLI.
- Ship the skills in every release archive and declare each in the manifest.
- Each skill starts from the installed version and the configuration the CLI actually resolves, and trusts the installed `--help` over the docs.

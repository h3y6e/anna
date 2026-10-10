---
name: anna-nrem
description: Build or refresh an anna memory with nrem. Use when the project uses anna and the user wants notes indexed.
---

# Index notes with anna nrem

`anna nrem <notes-dir>` writes `<notes-dir>/.anna.db`. `--amnesia` overwrites that file: a terminal asks first, and `--yes` skips the question. `--dry-run` writes nothing.

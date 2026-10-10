---
name: anna-recall
description: Search an anna memory with recall. Use when the project uses anna and the user wants to recall notes from a memory.
---

# Search notes with anna recall

`anna recall --in <notes-dir> <query>` reads `<notes-dir>/.anna.db`. `--json` prints one hit per line and ends with a result record. If the memory is missing, the error names `anna nrem <notes-dir>`.

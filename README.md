# anna

A lightweight CLI that indexes local text notes and searches.

The commands are named after sleep phases:

- `nrem` builds the search index from notes
- `recall` searches the memory

## Install

Install with `go install`:

```sh
go install github.com/h3y6e/anna@latest
```

or [mise](https://mise.en.dev):

```sh
mise use -g github:h3y6e/anna
```

`anna` embeds through any OpenAI-compatible `/v1/embeddings` endpoint, by default [llama.cpp](https://github.com/ggml-org/llama.cpp) at `http://localhost:8080` with `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0`. Override with `--embedder-url` and `--embedder-model`, e.g. for [Ollama](https://ollama.com/):

```sh
anna nrem ~/notes --embedder-url http://localhost:11434 --embedder-model qwen3-embedding:0.6b
```

## Quick start

Build a memory from a directory of notes:

```sh
anna nrem ~/notes
```

Each notes directory keeps its own memory inside it. This writes the memory to:

```text
~/notes/.anna.db
```

Recall notes from the memory:

```sh
anna recall --in ~/notes 'search query'
```

Search several notes directories as one memory:

```sh
anna nrem ~/notes ~/work/notes
anna recall --in ~/notes --in ~/work/notes 'search query'
```

Results are absolute paths. Directories that are the same or contain one another, including through symlinks, cannot be indexed or searched together.

Run lexical search only:

```sh
anna recall --in ~/notes --mode bm25 'search query'
```

## Configuration

You can provide a TOML config file with `--config`:

```sh
anna --config ./anna.toml nrem ~/notes
```

Without `--config`, `anna` searches for config files in this order:

1. `./anna.toml`
2. `$XDG_CONFIG_HOME/anna/config.toml`
3. `~/.config/anna/config.toml`

Local configuration values override global configuration values.

`notes` lists the directories that `nrem` builds when no directory is given and that `recall` reads when `--in` is not given. Relative entries resolve against the directory of the config file that sets them. `memory` is the file name of the memory inside each notes directory; it cannot point to another directory.

Example `anna.toml`:

```toml
notes = ["~/notes"]
memory = ".anna.db"
quiet = false
json = false

[embedder]
url = "http://localhost:8080"
model = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
query-prefix = ""
document-prefix = ""

[nrem]
amnesia = false

[recall]
mode = "hybrid"
limit = 10
```

`query-prefix` and `document-prefix` are prepended to search queries and notes before embedding.

Configuration values are resolved in this order:

1. CLI flags
2. Environment variables with the `ANNA_` prefix
3. Config file
4. Defaults

For example, `ANNA_EMBEDDER_URL` sets `embedder.url` unless a CLI flag overrides it. `ANNA_NOTES` separates directories with the OS path list separator (`:` on macOS and Linux).

## Search modes

`anna` supports four search modes:

| Mode     | Description                                                                          |
| -------- | ------------------------------------------------------------------------------------ |
| `bm25`   | Lexical search using indexed term statistics. Works without an embedder.             |
| `vector` | Cosine similarity between query and document embeddings.                             |
| `hybrid` | `0.80 * vector + 0.20 * normalized BM25`. This is the default.                       |
| `rrf`    | Reciprocal rank fusion of BM25 and vector rankings, rescored with cosine similarity. |

The embedding model and prefixes used to build the memory must match those used for recall.
When `recall` reads several directories, term statistics are computed over all of them, and every memory must use the same embedding model and prefixes.

## Incremental indexing

`nrem` reuses embeddings and term statistics for documents whose path and content hash have not changed.

To rebuild the entire index, use `--amnesia`:

```sh
anna nrem ~/notes --amnesia
```

### Periodic runs

On macOS, `nrem` can be scheduled with [mise launchd bootstrap](https://mise.en.dev/bootstrap/launchd.html).
The following is a minimal example; adjust the paths and interval for your setup:

```toml
[bootstrap.macos.launchd.agents.anna]
program = "~/go/bin/anna"
args = ["nrem", "~/notes"]
run_at_load = true
start_interval = 3600
```

## Tokenization

Japanese text is tokenized with [Kagome](https://github.com/ikawaha/kagome) and [UniDic](https://github.com/ikawaha/kagome-dict/tree/main/uni).

No extra runtime is required, and the tokenizer is not configurable.

## Development

Requires [mise](https://mise.en.dev/).

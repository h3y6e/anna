# anna

A lightweight CLI that indexes local text notes and searches.

The commands are named after sleep phases:

- `nrem` builds the search index from notes
- `recall` searches the index
- `rem` surfaces related note pairs

## Install

Install with `go install`:

```sh
go install github.com/h3y6e/anna@latest
```

or [mise](https://mise.en.dev):

```sh
mise use -g github:h3y6e/anna
```

`anna` uses [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` with its OpenAI-compatible API) for embedding generation by default:

```sh
anna nrem ~/notes
```

[Ollama](https://ollama.com/) is also supported:

```sh
ollama pull qwen3-embedding:0.6b
anna nrem ~/notes --embedder-provider ollama
```

| Embedder    | Default URL               | Default model                            |
| ----------- | -------------------------- | ----------------------------------------- |
| `llama.cpp` | `http://localhost:8080`   | `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` |
| `ollama`    | `http://localhost:11434`  | `qwen3-embedding:0.6b`                    |

## Quick start

Build a memory index from a directory of notes:

```sh
anna nrem ~/notes
```

This writes the index to:

```text
~/notes/memory.db
```

Recall notes from the index:

```sh
anna recall --memory ~/notes/memory.db 'search query'
```

Run lexical search only:

```sh
anna recall --memory ~/notes/memory.db --mode bm25 'search query'
```

Surface read-only recombination candidates:

```sh
anna rem --memory ~/notes/memory.db
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

Example `anna.toml`:

```toml
memory = "~/notes/memory.db"
quiet = false
json = false

[embedder]
provider = "llama.cpp"
url = "http://localhost:8080"
model = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"

[nrem]
amnesia = false

[recall]
mode = "hybrid"
limit = 10

[rem]
focus = "all"
limit = 10
threshold = 0.75
```

Configuration values are resolved in this order:

1. CLI flags
2. Environment variables with the `ANNA_` prefix
3. Config file
4. Defaults

For example, `ANNA_EMBEDDER_URL` sets `embedder.url` unless a CLI flag overrides it.

## Search modes

`anna` supports four search modes:

| Mode     | Description                                                                          |
| -------- | ------------------------------------------------------------------------------------ |
| `bm25`   | Lexical search using indexed term statistics. Works without an embedder.             |
| `vector` | Cosine similarity between query and document embeddings.                             |
| `hybrid` | `0.80 * vector + 0.20 * normalized BM25`. This is the default.                       |
| `rrf`    | Reciprocal rank fusion of BM25 and vector rankings, rescored with cosine similarity. |

The embedding model used to build the index must match the embedding model used for recall.

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

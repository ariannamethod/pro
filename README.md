# method engine

*Second name: PRO — Pure Recursive Organism*

## Installation

```bash
pip install .
```

The project exposes a lightweight conversational engine.  All modules use
standard Python and depend only on the libraries listed in `pyproject.toml`.
Optional extras exist for future experiments but are not required for the
core runtime.

## Architecture

The engine assembles itself from small cooperating modules.  Each module stays
independent yet communicates through plain Python calls.  The sections below
outline every function in detail.

### Engine

Coordinates high level behaviour and orchestrates all other modules.

#### filter_similar_candidates
Removes near-duplicate responses by comparing cosine similarity and text
identity.  Only the earliest unique candidate survives.

#### template_penalty
Assigns a small penalty when common template phrases appear too often.
The penalty grows with repetition to discourage boilerplate.

#### ProEngine.setup
Initialises queues, background workers, and dataset scanning.  Must be called
once before processing messages.

#### ProEngine.respond
Accepts a list of seed words and returns a single textual reply.  The method
collects candidate continuations, filters them, and chooses the best one.

#### ProEngine.process_message
Top level entry point used by the Telegram interface.  It analyses the incoming
message, invokes retrieval, generation, and memory, and finally returns the
response with diagnostic metrics.

### Prediction

Builds word co‑occurrence vectors and offers basic next token guesses.

#### start_background_init
Starts vector initialisation in the background so the event loop remains free.
Only the first call has effect; subsequent calls reuse the existing task.

#### suggest_async
Asynchronously suggests analog words for the given token.  Results come from
frequency vectors and return quickly thanks to cached embeddings.

#### suggest
Synchronous wrapper around `suggest_async` useful for simple scripts.  It blocks
until the asynchronous suggestion completes.

#### update
Updates the co‑occurrence graph with new words and persists the embeddings.
Large file I/O is delegated to a thread to avoid blocking the loop.

#### transformer_logits
Returns raw logits from the tiny self‑attention model for a token sequence.
The function initialises the model lazily and reuses weights between calls.

#### combine_predictions
Blends n‑gram and transformer probabilities using predefined weights.  The
resulting dictionary maps candidate words to combined probabilities.

### Memory

Stores conversation fragments in SQLite and keeps a small in‑memory cache.

#### init_db
Creates required tables and prepares the connection pool.  The pool size is
configurable and reused by all other memory operations.

#### encode_message
Converts raw text to a fixed size numeric vector using a TF‑IDF‑like scheme.
The vector is normalised for reliable similarity comparisons.

#### persist_embedding
Writes a message and its embedding to the database.  Existing rows are updated
atomically via `INSERT OR REPLACE`.

#### fetch_recent_messages
Returns the most recent messages along with their tags.  Useful for retrieval
and debugging.

#### fetch_related_concepts
Looks up concept relationships stored in the graph tables.  The call helps
retrieval assemble richer context around a query.

#### is_unique
Checks whether a message already exists in the memory table.  Similarity above
the threshold marks it as a duplicate.

#### store_response
Persists a generated response if it passes uniqueness and grammar checks.

### Retrieval Augmented Generation

Supplies additional context from internal memory and optional external sources.

#### retrieve_external
Fetches snippets from Wikipedia using the public API.  Results are cached to
avoid repeated network calls.

#### retrieve
Combines recent messages, related concepts, and optional external data.  The
function ranks all candidates by overlap and cosine similarity before returning
a deduplicated list.

### Forecast

Explores possible continuations to guide the engine toward novel paths.

#### simulate_paths
Builds a small tree of token sequences using the self‑attention model.  Each
branch carries probability and novelty scores.

#### backpropagate_forecast
Walks the forecast tree and performs tiny training steps on unexpected tokens.
Novel branches therefore influence future predictions more strongly.

### Sequence Tools

Tracks n‑gram statistics to enrich prediction and metric calculations.

#### analyze_sequences
Updates word, bigram, trigram, and character n‑gram counts inside the shared
state dictionary.  Inverse frequency maps are maintained for quick lookup.

### Identity

Adapts pronouns to mirror the user for a more personal tone.

#### swap_pronouns
Replaces first and second person pronouns according to a static map.  Tokens
without a mapping are returned unchanged.

### Morphology

Handles basic morpheme analysis and encoding for languages with rich inflection.

#### split
Breaks a word into root, prefixes, and suffixes using simple lists.  The order
of affixes matches the original word structure.

#### tokenize
Converts text to a flat list of morphemes by applying `split` to each word.
Lower‑casing and non‑word filtering are handled automatically.

#### encode
Hashes each morpheme into a deterministic vector and aggregates the result.
The vector is normalised to unit length for cosine comparisons.

#### filter_by_tags
Selects token indices that match inclusion and exclusion tag filters.  Both
filters accept sets of strings and default to permissive behaviour.

### Metrics

Computes quality scores and tracks latency across modules.

#### compute_metrics
Aggregates entropy, perplexity, resonance, and related statistics from the
current state.  Character‑level resonance is also measured.

#### target_length_from_metrics
Maps metric totals to a target response length.  The value always falls between
the provided minimum and maximum.

#### record_latency
Adds a latency sample for a named operation.  Only the most recent window of
values is retained.

#### latency_stats
Returns average and percentile latencies for a given operation.  Missing data
produces zeroed statistics.

#### format_latency_stats
Formats all collected latency statistics into human‑readable strings.

### Meta Module

Evolves engine parameters using recorded metrics to improve future runs.

#### update
Appends a metrics and parameter record, saves it to disk, and schedules a
recomputation of the best parameters with slight random noise.

#### best_params
Returns the current set of best‑known parameters so other modules can adapt.

### Tune Module

Adds new knowledge from datasets or external sources.

#### train
Reads a dataset file, analyses sequences, updates embeddings, and increments
adapter usage counters.  It is the default entry for supervised updates.

#### train_weighted
Like `train` but scales the contribution by an explicit weight.  Useful when
mixing multiple datasets with different importance.

#### tune_with_knowledge
Retrieves documents by query, analyses them, and applies prediction updates.
External retrieval uses the same mechanism as the RAG module.

#### merge_specialist
Blends a specialist state into a base state using a temperature parameter.
Higher temperature favours the specialist weights.

### RAG Embedding

Provides deterministic sentence embeddings and simple relation extraction.

#### embed_sentence
Projects a sentence through a fixed random matrix and normalises the result.
Characters are hashed into a 256‑dimensional seed vector before projection.

#### extract_entities_relations
Parses short descriptions for subject‑verb‑object triples.  Unique entities and
relations are returned for downstream knowledge graph updates.

### Grammar Filters

Rejects obviously malformed sentences before they reach memory.

#### passes_filters
Checks text for duplicate words, incorrect articles, and other heuristic
patterns.  Logs high‑entropy sequences for later analysis.

### Message Utilities

Assorted helpers for response generation and storage.

#### build_analog_map
Generates a mapping from tokens to analog replacements using prediction
suggestions.  Access to shared structures is synchronised with a lock.

#### ensure_unique
Runs grammar filters and uniqueness checks before saving a response.  Returns a
boolean indicating whether the save succeeded.

### Telegram Interface

Simple long‑polling bridge between users and the engine.

#### get_updates
Fetches new Telegram messages with configurable offset and timeout handling.
Network errors are logged and produce empty results.

#### send_message
Posts a reply back to the chat and reports success.  Failed attempts are noted
in the logs for inspection.

#### main
Entry point for the standalone bot.  It starts the engine, processes incoming
messages, and gracefully shuts down on exit.

### Datasets

The `datasets` directory stores plain text files used for training and
experimentation.  The engine scans this directory automatically during startup
and incorporates any new material.

### Compatibility Helpers

#### to_thread
Small wrapper around `asyncio` that executes blocking functions in a thread and
returns their results without blocking the event loop.

## Usage Example

```python
from pro_engine import ProEngine
import asyncio

async def chat():
    engine = ProEngine()
    await engine.setup()
    reply, info = await engine.process_message("hello there")
    print(reply)
    await engine.shutdown()

asyncio.run(chat())
```

## Benchmarks

| Configuration | Perplexity | Throughput (req/s) |
|---------------|-----------:|-------------------:|
| Baseline      |       42.1 |                110 |
| Dual adapters |       30.5 |                 90 |

These numbers offer a reference point for further optimisation.  Reproduce them
locally by running the included scripts on comparable hardware.

## License

This project is licensed under the terms of the GNU General Public License v3.0.
See `LICENSE` for full details.  Copyright (C) 2024 Oleg Ataeff and Arianna Method.


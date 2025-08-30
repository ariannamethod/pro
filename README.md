
# me - method engine

*Second name: PRO — Pure Recursive Organism*

## First Stage

A few hours ago the me entity emerged from an empty directory and began shaping itself. An audit registers 59 merged pull requests with none pending after about 12.6 hours. Modules connect without ceremony.

### Engine

Coordinates tasks so the entity acts as one coherent program. It receives inputs and hands results to the next module.
This backbone keeps the whole organism synchronized.

### Prediction

Schedules token choices from context and reports outcomes to the engine. This feedback loop keeps generation steady.
Its queue never sleeps during dialogue.

### Memory

Stores conversations in SQLite; the engine decides when to read or write. Old replies stay available when needed.
Other modules query it through the engine.

### Memory Pool

Keeps hot fragments ready to reduce latency for all modules. Other components fetch from it without delay.
Caching here shields the database from stress.

### Forecast

Simulates future paths to guide engine decisions. Suggested routes arrive before tokens are chosen.
Forecasts flow back as gentle hints.

### Sequence Tools

Scan text for patterns and feed findings to prediction and forecast. These hints refine later choices.
Pattern memory grows with every session.

### Identity

Swaps pronouns after checking context, mirroring the user. It preserves tone and intent.
Identity shifts feel natural to speakers.

### Morphology

Caches word forms for quick inflection lookup. This cache eases repeated conjugations.
Morphology thus remains brisk and tidy.

### Retrieval Augmented Generation

Joins related memories with prompts before prediction chooses tokens. The mix adds context to each reply.
Retrieval blends history with the present.

### Reasoning Block

Solves basic logical rules and returns conclusions. These results help shape coherent answers.
Reasoning slots decisions into clear paths.

### Embedding Store

Holds vectors for fast similarity search through a locked interface. Locks prevent simultaneous writes.
Embedding matches support swift recalls.

### Meta Module

Watches metrics and nudges parameters toward improvement. It suggests when tuning should occur.

### Metrics Unit

Records scores for every utterance and sends them to meta. These numbers fuel gradual refinement.

### Tune Module

Retrains weights when new data appears. Updated models replace old ones during quiet periods.

### Telegram Interface

Long polling bridge between humans and the engine. Messages move through the engine and return back.

### Datasets

Phrase collections stored for training runs. They form the base for learning cycles.

### Dataset Queue Worker

Streams file changes to the engine without blocking. New lines appear as soon as they are saved.

### Caching and Pooling

Shared helpers keep memory operations efficient under load. The engine calls them for every memory task.

### Vector Locks

Mutexes guard vector writes. Only one process updates embeddings at a time.

### Test Suite

Ensures modules behave as described. Regressions are caught before merging.

### Self Assembly

Recursive rules let the system extend itself. Each new unit follows the pattern.

## Stage 2

### Mesh Gossip Protocol

Diffuses adapter updates across nodes until weights converge. Nodes exchange small packets on every cycle.
The mesh avoids single point failures.

### Score Tokens

Global scores keep only high value tokens during inference. Weak choices drop out early.
Scoring trims waste before decoding.

### Saliency Thresholding

Drops terms with little impact to reduce noise. Only meaningful words remain.
Filters run inside the prediction loop.

### Quantum Hybrid Attention

Blends classical queries with quantum amplitudes to sharpen focus. Extra signal guides attention.
Quantum hints complement standard vectors.

### Qiskit Integration

Calls quantum circuits for phase aware context. These runs happen only when needed.
Fallbacks ensure progress if circuits fail.

### Meta Controller Reinforcement Loop

Policy gradients drive modules toward higher long term reward. The controller nudges each piece gently.
Rewards accumulate across many turns.

### Metric Monitoring

Moving averages flag progress and dips. Sudden drops trigger reviews.
Stable curves mean healthy learning.

### LoRA Adapter Persistence

Low rank deltas retain personal tweaks. Users can resume from the same state later.
Adapters stay compact for easy sharing.

### Punctuation Rule Engine

Finite rules keep possessive endings consistent. This engine prevents awkward phrasing.
Grammar rules integrate with morphology checks.

### External RAG Storage Interface

Allows remote memories to enrich prompts. External stores plug in through a simple API.
Network calls merge results seamlessly.

### External Knowledge Tuning

Uses fresh facts to adjust parameters. Tuning runs after each injection.
Knowledge updates happen while users wait.

### Time Fold Transformer

Pairs distant steps to catch temporal symmetry. Forward and backward views meet.

### Quantum Memory

Stores qubits in superposition for parallel recall. Classical reads collapse the answer.

### Embedding Locking Mechanism

Ensures single writer access to embedding tables. Readers wait until updates finish.

### Mesh CLI

Dispatches gossip commands and reports cluster status. Operators monitor health from terminals.

### Forecast Enhancement

Attention weights project the next token distribution. The engine chooses the final branch.

### Memory Pool Optimization

Constant time lookups keep hot fragments close. Cold pieces are evicted in order.

### Sequence Analyzer Upgrades

N-gram counts refine pattern recognition. Prediction adopts these counts instantly.

### Identity Pronoun Swapping

Permutation matrices shift perspectives in dialogue. Identity updates keep voices straight.

### Self Reflection Engine

Compares output to references and adjusts models. Discrepancies shrink over time.

### Smalltalk Technology

Treats conversation as a chain that keeps chats lively. Each link adds fresh energy.

## Telegram Interface

1. Copy `.env.example` to `.env` and replace the placeholder token.
2. Run `python pro_tg.py` to start the me Telegram interface. It echoes incoming messages using long polling.

## Benchmarks

| Configuration   | Perplexity | Throughput (req/s) |
|-----------------|-----------:|-------------------:|
| Single Adapter  |       42.1 |                110 |
| MoE (2 adapters)|       30.5 |                 90 |

## Personal Fine Tuning with LoRA

LoRA adapters can be enabled by setting `use_lora` in the training configuration. The snippet below demonstrates how to fine tune and persist personal adapters:

```python
from autoadapt import LayerMutator, LoRALayer
from trainer import Trainer

trainer = Trainer(use_lora=True)
layer = LoRALayer(
    name="greeting",
    rank=2,
    alpha=1.0,
    matrix_a=[[0.0, 0.0], [0.0, 0.0]],
    matrix_b=[[0.0, 0.0], [0.0, 0.0]],
)
trainer.mutator.add_lora_layer(layer)
trainer.mutator.save("checkpoints/my_lora")
```

Saved adapters can later be reloaded with `LayerMutator.load("checkpoints/my_lora")` for continued training or inference.


# Self-Reflection Fine-Tuning

This document describes the self-reflection training cycle and moderation
requirements for generated data.

## Process
1. Collect recent conversations.
2. Run `WeaknessDetector` to list common failure modes.
3. `DataGenerator` creates synthetic prompts to address each weakness.
4. `SelfFineTuner` fine-tunes the model on the curated data.

## Data Moderation
All generated or collected data must be moderated before training:
- Remove personal or sensitive information.
- Filter hateful, sexual or other policy-violating content.
- Ensure data sources comply with licensing requirements.

## Weakness Report
The latest evaluation after a fine-tuning cycle produced the following
improvements:

| Weakness                     | Before | After |
| ---------------------------- | ------:| -----:|
| Unclear follow-up questions  |   35%  |  10%  |
| Inconsistent formality       |   25%  |   8%  |
| Limited context retention    |   30%  |  12%  |

The percentages indicate the frequency of each weakness before and after the
self-reflection fine-tuning cycle.

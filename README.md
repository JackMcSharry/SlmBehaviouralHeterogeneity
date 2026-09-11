# SLM Behavioural Heterogeneity

This demo tests and analyses the response distribution of 10 different participant profiles based on how they respond to the same behavioural prompt.

The current version uses the unfine tuned `Qwen/Qwen2.5-0.5B-Instruct` model. The aim is to create a baseline response distribution that can later be compared against a fine tuned model to see whether fine tuning increases behavioural heterogeneity.

The behavioural task gives each participant £10 and asks them to choose how many whole pounds, from £0 to £10, they would give to an anonymous participant.


## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy transformers accelerate matplotlib
```

## Run the experiment

```powershell
python experiment.py
```

Participant profiles and experiment settings are stored in the `config` folder.

Each run saves the participant responses, whether each response is valid, and metadata about the run.

## Analyse the results

```powershell
python analysis.py "results\<RUN-FOLDER>\baseline_results.csv"
```

The analysis calculates basic statistics for the response distribution and produces a bar chart showing how the responses are spread across the allowed range.

## Baseline result

In the current representative run:

- 10 valid responses
- mean: `0.40`
- variance: `0.24`
- standard deviation: `0.49`
- unique responses: `2`

All 10 responses were either `0` or `1`, despite the allowed range being `0–10`.

This gives an initial example of the response homogeneity that the wider project is intended to investigate.

### Result files

- [Baseline responses](results/2026-09-11_18-38-34/baseline_results.csv)
- [Run metadata](results/2026-09-11_18-38-34/metadata.json)
- [Response distribution](results/2026-09-11_18-38-34/response_distribution.png)

## Current limitations

This is only an initial baseline and does not yet include:

- real human response data
- fine tuning
- PEFT or LoRA
- comparison against a real human response distribution
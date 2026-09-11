# SLM Behavioural Heterogeneity

This project explores whether a small language model can produce variation in human behavioural responses. 

The current demo focusses on a simple baseline experiment using Qwen/Qwen2.5-0.5B-Instruct. 
Different participant profiles are given the same prompt and their responses are recorded and analysed to measure how much variation there is in the unfine tuned model. 

## Current scope
The demo currently includes:
- configurable:
    - participant profiles
    - model and generation settings
- deterministic baseline inference
- response validation
- saved experiment resilts and metadata
- basic statistical analysis of response heterogenity
- a plotted response distribution
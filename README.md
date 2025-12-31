---
title: F1 Strategy AI
emoji: 🏎️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
```

---

```
# F1 Strategy AI

An AI-powered Formula 1 race strategy generator that provides optimal pit stop strategies, tire choices, and race management recommendations based on current track conditions.

![F1 Strategy AI](https://img.shields.io/badge/Model-Qwen2.5--1.5B-red)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)

## Overview

F1 Strategy AI leverages a fine-tuned Large Language Model to analyze race conditions and generate data-driven Formula 1 race strategies. The model considers driver information, circuit characteristics, weather conditions, and track temperature to provide comprehensive strategic recommendations.

**Live Demo:** [https://huggingface.co/spaces/akothari09/f1-strategy-ai](https://huggingface.co/spaces/akothari09/f1-strategy-ai)

**Model:** [https://huggingface.co/akothari09/f1StrategyTrainer](https://huggingface.co/akothari09/f1StrategyTrainer)

## Features

- **AI-Powered Strategy Generation**: Fine-tuned Qwen2.5-1.5B-Instruct model
- **Real-time Analysis**: Generates strategies in 10-20 seconds on CPU
- **Comprehensive Inputs**: Driver, circuit, temperatures, wind, track conditions
- **Detailed Output**: Tire choices, pit windows, alternate strategies
- **Web Interface**: Interactive Gradio UI

## Demo

Try the live demo: [F1 Strategy AI on Hugging Face Spaces](https://huggingface.co/spaces/akothari09/f1-strategy-ai)

### Example Usage:

**Input:**

Driver: Max Verstappen
Race: Monaco Grand Prix
Track Temperature: 35°C
Air Temperature: 28°C
Wind Speed: 15 km/h
Track Condition: Dry


**Output:**

RACE STRATEGY ANALYSIS

STINT 1 (Laps 1-18): MEDIUM COMPOUND
- Optimal starting compound for track temperature
- Focus on tire preservation in opening stint
- Target consistent lap times within 0.3s variance

PIT WINDOW (Laps 18-22): PRIMARY STOP
- Execute undercut if in traffic
- Switch to HARD compound for durability

STINT 2 (Laps 23-50): HARD COMPOUND
- Push for track position in clean air
- Monitor tire degradation carefully
- Adapt strategy for safety car scenarios


## Model Architecture

### Base Model
- **Name**: Qwen2.5-1.5B-Instruct
- **Parameters**: 1.5 billion
- **Context Length**: 32,768 tokens
- **Architecture**: Transformer-based decoder-only LLM
- **Optimization**: FP32 for CPU inference

### Fine-tuning Details
- **Method**: LoRA (Low-Rank Adaptation)
- **Framework**: Hugging Face PEFT
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **LoRA Rank**: 16
- **LoRA Alpha**: 16
- **Dropout**: 0.05

## Fine-tuning Process

### 1. Dataset
The model was fine-tuned on a curated dataset containing:
- Historical F1 race strategy data
- Track-specific optimal strategies for various circuits
- Weather condition impacts on tire performance
- Pit stop timing optimization
- Driver-specific performance characteristics
- Collected through Fast API, an API used for gaining historical f1 statistics

### 2. Training Configuration
```python
base_model = "Qwen/Qwen2.5-1.5B-Instruct"

lora_config = LoraConfig(
    r=16,                      # LoRA rank
    lora_alpha=16,             # LoRA alpha scaling
    target_modules=[
        "q_proj",              # Query projection
        "k_proj",              # Key projection
        "v_proj",              # Value projection
        "o_proj"               # Output projection
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./f1-strategy-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    warmup_ratio=0.1,
)
```


### 3. Training Pipeline
1. **Data Preprocessing**: Tokenized race strategy examples
2. **LoRA Application**: Low-rank adapters injected into attention layers
3. **Supervised Fine-tuning**: Trained on strategy generation task
4. **Validation**: Evaluated on held-out test circuits
5. **Export**: LoRA adapters saved to Hugging Face Hub

## Installation & Usage

### Run Locally

1. **Clone the repository:**
```
bash
git clone https://github.com/akothari09/f1-strategy-ai.git
cd f1-strategy-ai
```

3. **Install dependencies:**
```
bash
pip install -r requirements.txt
```

5. **Run the app:**
```
bash
python app.py

The application will launch at `http://localhost:7860`
```
### Access via Hugging Face

Simply visit: [https://huggingface.co/spaces/akothari09/f1-strategy-ai](https://huggingface.co/spaces/akothari09/f1-strategy-ai)

## API Usage

### Input Parameters

| Parameter | Type | Range/Options | Description | Example |
|-----------|------|---------------|-------------|---------|
| `driver` | string | Any | Driver name | `"Max Verstappen"` |
| `race` | string | Any | Circuit name | `"Monaco Grand Prix"` |
| `track_temp` | number | 0-60 | Track temperature (°C) | `35` |
| `air_temp` | number | 0-50 | Air temperature (°C) | `28` |
| `wind_speed` | number | 0-100 | Wind speed (km/h) | `15` |
| `track_condition` | string | "dry" or "damp" | Track surface condition | `"dry"` |
| `max_tokens` | number | 100-600 | Maximum output length | `400` |
| `temperature` | number | 0.0-1.0 | Model creativity (0=deterministic) | `0.7` |

## Project Structure


f1-strategy-ai/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore rules

## Accessing the Model

### Option 1: Via Hugging Face Spaces (Easiest)
Visit the live demo: [https://huggingface.co/spaces/akothari09/f1-strategy-ai](https://huggingface.co/spaces/akothari09/f1-strategy-ai)

### Option 2: Via API
Use the REST API as shown in the examples above.

### Option 3: Load Model Directly
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# For fine-tuned version (if you have the adapter)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "akothari09/f1StrategyTrainer")
```

## License

This project is licensed under the Apache License.

## Acknowledgments

- **Qwen Team** at Alibaba for the base model
- **Hugging Face** for PEFT library and model hosting
- **Formula 1** for the inspiration
- **Open-source community** for tools and libraries

```
## Contact

**Author**: Aditi Kothari

**GitHub**: [https://github.com/akothari09/f1-strategy-ai]

**HF Space**: [https://huggingface.co/spaces/akothari09/f1-strategy-ai]

**Model**: [https://huggingface.co/akothari09/f1StrategyTrainer]
```

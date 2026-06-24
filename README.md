# LLM Labeling Project

This project compares different machine learning and large language models for text classification tasks on appliance repair call data.

## Project Structure

```
llm_labeling/
├── main.py                    # Main orchestration script - runs all models
├── data/                      # Data directory
│   └── call_data_rw.csv       # Main dataset
│   └── call_data_revised.csv  # Main dataset with revised prompt
├── config.env                 # Configuration file for API keys
│
├── LLM Runners/
│   ├── gemini_runner.py       # Google Gemini API client and runner
│   ├── gemini_tuned_runner.py # Fine-tuned Gemini model runner
│   ├── gpt_runner.py          # OpenAI GPT runner
│   ├── open_router_runner.py  # OpenRouter client (LLAMA, Claude)
│   └── openai_utils.py        # OpenAI utility functions
│
├── Batch Processing/
│   ├── call_gemini_batch.py   # Gemini batch processing - not used, takes about a day to complete
│   └── gpt_batch_runner.py    # GPT batch processing - optional to reduce cost
│
├── Traditional ML Models/
│   ├── LR.py                  # Logistic Regression model
│   └── Roberta.py             # RoBERTa transformer model
│
├── Utilities/
│   ├── prompt.py              # Prompt template v1
│   ├── prompt2.py             # Prompt template v2
│   └── data_split.py          # Data splitting utilities
│
└── Output/
    ├── results/               # Training results and checkpoints for Roberta
    └── logs/                  # Training logs
```

## Models Implemented

### Traditional ML Models
- **Logistic Regression (LR)**: TF-IDF vectorization + Logistic Regression classifier
- **RoBERTa**: Fine-tuned transformer model (roberta-base)

### Large Language Models
- **Gemini**: Google's Gemini 2.5 Flash model
- **GPT**: OpenAI's GPT models
- **LLAMA**: Meta's LLAMA 3.3 70B (via OpenRouter)
- **Claude**: Anthropic's Claude Opus 4.7 (via OpenRouter)

## Setup

1. Install dependencies:
```bash
pip install pandas scikit-learn torch transformers openai google-generativeai
```

2. Configure API keys in `config.env`:
```env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
```

## Usage

### Run All Models
```bash
python main.py [start] [limit]
```
- `start`: Starting index for LLM processing (default: 0)
- `limit`: Number of rows to process for LLMs (default: all)

### Run Individual Models

#### Traditional ML Models
```bash
python LR.py          # Logistic Regression
python Roberta.py     # RoBERTa
```

#### LLM Models
```bash
python gemini_runner.py [start] [limit]
python gpt_runner.py [start] [limit]
python open_router_runner.py [start] [limit]
```

#### Batch Processing
```bash
python call_gemini_batch.py
python gpt_batch_runner.py
```

## Data Format

The dataset (`call_data_revised.csv`) should contain:
- `ID`: Unique identifier
- `Script`: Call transcript text
- `Result`: Ground truth label
- `split`: "train" or "test"
- `label`: Numeric label for classification

## Output

Each model saves its predictions to the CSV file:
- `LR Result`: Logistic Regression predictions
- `Roberta Result`: RoBERTa predictions
- `Gemini Result`: Gemini predictions
- `GPT Result`: GPT predictions
- `LLAMA Result`: LLAMA predictions
- `Claude Result`: Claude predictions

## Workflow

1. **Data Preparation**: Split data using `data_split.py`
2. **Traditional ML**: Train LR and RoBERTa on full dataset
3. **LLM Inference**: Run LLM models row-by-row with optional start/limit
4. **Results**: All predictions saved to the same CSV file

## Notes

- LLM runners include rate limiting and error retry logic
- Traditional ML models train on the full dataset and save predictions for all rows
- The main script runs models sequentially: LR → RoBERTa → Gemini → GPT → LLAMA → Claude
- Progress is saved after each prediction to avoid data loss

import os
import sys
import subprocess
import pandas as pd

from gemini_runner import GeminiClient
from openai_utils import GPTClient
from open_router_runner import OpenRouterClient
from prompt import prompt
# from prompt2 import prompt # for revised prompt, comment out the above line and uncomment this line

CUR_DIR = os.path.dirname(__file__)
# results will be
OUTPUT_FILE = os.path.join(CUR_DIR, "data", "call_data_rw.csv")
# OUTPUT_FILE = os.path.join(CUR_DIR, "data", "call_data_revised.csv") # for revised prompt, comment out the above line and uncomment this line



def run_gemini(df, input_path, start=0, limit=None):
    """Run Gemini model on the dataset"""
    print("\n=== Running Gemini ===")
    result_column = "Gemini Result"

    if result_column not in df.columns:
        df[result_column] = ""
    df[result_column] = df[result_column].fillna("").astype(str)

    client = GeminiClient()
    rows = df.iloc[start:] if limit is None else df.iloc[start:start + limit]

    for index, row in rows.iterrows():
        if pd.notna(row.get(result_column)) and str(row.get(result_column)).strip():
            continue

        messages = [
            {
                "role": "system",
                "content": "You are a classification assistant for appliance repair company.",
            },
            {
                "role": "user",
                "content": f"{prompt}Conversation:\n{row['Script']}",
            },
        ]

        try:
            result = client.get_response(messages)
            df.loc[index, result_column] = result
            df.to_csv(input_path, index=False)
            print(f"Gemini: id={row['ID']} expected={row['Result']} result={result}")
        except Exception as e:
            print(f"Gemini error for ID {row['ID']}: {e}")

    return df


def run_gpt(df, input_path, start=0, limit=None):
    """Run GPT model on the dataset"""
    print("\n=== Running GPT ===")
    result_column = "GPT Result"

    if result_column not in df.columns:
        df[result_column] = ""
    df[result_column] = df[result_column].fillna("").astype(str)

    client = GPTClient()
    rows = df.iloc[start:] if limit is None else df.iloc[start:start + limit]

    for index, row in rows.iterrows():
        if str(row.get(result_column, "")).strip():
            continue

        messages = [
            {
                "role": "system",
                "content": "You are a classification assistant for appliance repair company.",
            },
            {
                "role": "user",
                "content": f"{prompt}Conversation:\n{row['Script']}",
            },
        ]

        try:
            result = client.get_response(messages, max_tokens=10, temperature=0)
            df.loc[index, result_column] = result
            df.to_csv(input_path, index=False)
            print(f"GPT: id={row['ID']} expected={row['Result']} result={result}")
        except Exception as e:
            print(f"GPT error for ID {row['ID']}: {e}")

    return df


def run_openrouter(df, input_path, start=0, limit=None, model="meta-llama/llama-3.3-70b-instruct"):
    """Run OpenRouter (Claude/LLAMA) model on the dataset"""
    print(f"\n=== Running OpenRouter ({model}) ===")

    if "claude" in model.lower():
        result_column = "Claude Result"
    else:
        result_column = "LLAMA Result"

    if result_column not in df.columns:
        df[result_column] = ""
    df[result_column] = df[result_column].fillna("").astype(str)

    client = OpenRouterClient(model=model)
    rows = df.iloc[start:] if limit is None else df.iloc[start:start + limit]

    for index, row in rows.iterrows():
        if str(row.get(result_column, "")).strip():
            continue

        messages = [
            {
                "role": "system",
                "content": "You are a classification assistant for appliance repair company.",
            },
            {
                "role": "user",
                "content": f"{prompt}Conversation:\n{row['Script']}",
            },
        ]

        try:
            result = client.get_response(messages, max_tokens=10, temperature=0)
            df.loc[index, result_column] = result
            df.to_csv(input_path, index=False)
            print(f"OpenRouter: id={row['ID']} expected={row['Result']} result={result}")
        except Exception as e:
            print(f"OpenRouter error for ID {row['ID']}: {e}")

    return df


def run_lr():
    """Run Logistic Regression model using subprocess"""
    print("\n=== Running Logistic Regression ===")
    lr_path = os.path.join(CUR_DIR, "LR.py")
    result = subprocess.run([sys.executable, lr_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"LR Error: {result.stderr}")
    return result.returncode == 0


def run_roberta():
    """Run Roberta model using subprocess"""
    print("\n=== Running Roberta ===")
    roberta_path = os.path.join(CUR_DIR, "Roberta.py")
    result = subprocess.run([sys.executable, roberta_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Roberta Error: {result.stderr}")
    return result.returncode == 0


def main():
    """Main function to run all models"""
    input_path = OUTPUT_FILE

    # Parse command line arguments
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Run traditional ML models first (they train on full dataset)
    run_lr()
    run_roberta()

    # Load dataset after ML models complete
    df = pd.read_csv(input_path)

    # Run LLM models (they process row by row with start/limit)
    run_gemini(df, input_path, start, limit)
    run_gpt(df, input_path, start, limit)
    run_openrouter(df, input_path, start, limit, model="meta-llama/llama-3.3-70b-instruct")
    run_openrouter(df, input_path, start, limit, model="anthropic/claude-opus-4.7")

    print("\n=== All models completed ===")


if __name__ == '__main__':
    main()

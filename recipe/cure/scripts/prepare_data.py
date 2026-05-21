import argparse
import json
from pathlib import Path

import pandas as pd


MATH_PROMPT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
DEFAULT_INPUT_PATH = Path("data/DeepMath-103K/train.parquet")
DEFAULT_OUTPUT_PATH = Path("data/DeepMath-103K/train_76k8.parquet")
RAW_DEEPMATH_COLUMNS = (
    "question",
    "final_answer",
    "difficulty",
    "topic",
    "r1_solution_1",
    "r1_solution_2",
    "r1_solution_3",
)
OUTPUT_COLUMNS = (
    "question",
    "final_answer",
    "difficulty",
    "topic",
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
)


def _first_present(example, keys):
    for key in keys:
        value = example.get(key)
        if value is not None and value != "":
            return value
    raise KeyError(f"Expected one of {keys} in example keys {sorted(example.keys())}")


def _as_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _math_prompt(problem):
    problem = _as_text(problem).strip()
    if MATH_PROMPT_SUFFIX in problem:
        return problem
    return f"{problem}\n{MATH_PROMPT_SUFFIX}"


def build_math_row(example, idx, split, data_source):
    problem = _first_present(example, ("problem", "question", "prompt", "input"))
    answer = _first_present(example, ("answer", "final_answer", "target", "gt_answer", "solution"))
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": _math_prompt(problem)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": _as_text(answer)},
        "extra_info": {"split": split, "index": idx},
    }


def _validate_raw_columns(dataframe):
    missing = [column for column in RAW_DEEPMATH_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required DeepMath columns: {missing}")


def convert_deepmath_parquet(input_path, output_path):
    dataframe = pd.read_parquet(input_path)
    _validate_raw_columns(dataframe)

    rows = []
    for idx, example in dataframe.iterrows():
        example_dict = example.to_dict()
        row = build_math_row(example_dict, idx=int(idx), split="train", data_source="zwhe99/DeepMath-103K")
        rows.append(
            {
                "question": _as_text(example_dict["question"]),
                "final_answer": _as_text(example_dict["final_answer"]),
                "difficulty": example_dict["difficulty"],
                "topic": _as_text(example_dict["topic"]),
                **row,
            }
        )

    output_dataframe = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_parquet(output_path)
    print(f"Wrote {len(output_dataframe)} rows to {output_path}")
    return output_dataframe


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    convert_deepmath_parquet(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

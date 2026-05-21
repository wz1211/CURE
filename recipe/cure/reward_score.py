def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500","OlympiadBench/OE_TO_maths_en_COMP","zwhe99/DeepMath-103K"]:
        res=compute_math_score(solution_str, ground_truth)
        return res
    else:
        raise ValueError(f"Unsupported data source for reasoning task: {data_source}")
    

def compute_math_score(model_output: str, ground_truth: str) -> bool:
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        from math_verify.errors import TimeoutException
    except ImportError:
        print("To use Math-Verify, please install it first by running `pip install math-verify`.")
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        pass
    except Exception as e:
        pass

    return ret_score

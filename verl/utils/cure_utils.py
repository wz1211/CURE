import uuid
from verl import DataProto
import torch
import re
from typing import Dict, List, Optional
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import numpy as np
from tensordict import TensorDict

INSTRUCT_CRITIQUE_PROMPT = """You are given a problem and an AI-generated solution.

Problem:
{question}

AI Solution:
{solution}

Your tasks:
1) Carefully read Problem and the entire AI Solution from start to finish.
2) Identify the single earliest step or statement whose error *affects the final answer*.
    - Ignore any corrected or irrelevant intermediate errors.
    - If the final answer is missing, this must be considered an error.
    - If multiple issues propagate, choose the earliest one.
    - If no errors propagate to the final answer, set `ERROR_FOUND: false` and proceed to task 5.
3) Quote the *minimal* snippet that contains this issue. 
4) Briefly explain why this issue affects the final answer.
5) Produce a high-level hint that helps a student avoid making the *same kind of error* (if any error was found) or *ensure correctness* (if the solution was flawless).
    - Do NOT provide the next step or the correct answer.
    - The hint must be strategic and grounded in the problem's context, but avoid referencing specific numbers, symbols, or lines from the solution.

Rules:
- Do not rewrite or solve the problem.
- The hint must be self-contained and avoid specific computations or targeted fixes.
- Even when `ERROR_FOUND: false`, you should still provide a general hint relevant to the overall approach.
- You must output all four fields below in order.

Output format:
ERROR_FOUND: true/false
ERROR_QUOTE: <the minimal snippet or "">
WHY_IT_MATTERS: <1–2 sentence explanation or "">
HIGH_LEVEL_HINT: <general strategy hint without specific steps or answers>"""

NEGATIVE_CONCLUSION = "ERROR_FOUND: true\n"
POSITIVE_CONCLUSION = "ERROR_FOUND: false\n"

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _extract_content(tensor: torch.Tensor, start_tag: str, end_tag: str, tokenizer) -> list:
    decoded = tokenizer.batch_decode(tensor, skip_special_tokens=False)
    # print(f"Decoded: {decoded}")
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    return [m.group(1).strip() if (m:=re.search(pattern,t,re.DOTALL)) else "" for t in decoded]

def _normalize_size_divisor(size_divisor: Optional[int]) -> int:
    return max(1, int(size_divisor or 1))

def _truncate_count_to_divisor(count: int, size_divisor: Optional[int]) -> int:
    divisor = _normalize_size_divisor(size_divisor)
    if divisor <= 1:
        return count
    return count // divisor * divisor

def _filter_by_interval_multisample(
    data: DataProto,
    reward_list,
    new_input_ids,
    new_attention_mask,
    new_position_ids,
    interval: tuple[float, float],
    train_discriminability: bool,
    target_batch_size: Optional[int],
):
    """按uid先筛acc区间，再根据是否训练判别能力抽样对应样本。"""
    reward_array = np.asarray(reward_list)
    uid_list = data.non_tensor_batch['uid']
    
    uid_groups: Dict[str, List[int]] = {}
    for idx, uid in enumerate(uid_list):
        uid_groups.setdefault(uid, []).append(idx)

    # Pools
    zero_group_indices = []      # 来自 acc=0 组的所有样本
    zero_head_indices = []       # 来自 acc=0 组的首个样本
    all_correct_indices = []     # 来自 acc=1 组的所有样本
    mixed_wrong_indices = []     # 来自 0<acc<1 组的错误样本
    mixed_correct_indices = []   # 来自 0<acc<1 组的正确样本
    
    # Pre-selection (Pairs)
    pre_selected = []
    
    for indices in uid_groups.values():
        group_rewards = reward_array[indices]
        group_acc = float(np.mean(group_rewards))

        if group_acc == 0.0:
            zero_group_indices.extend(indices[1:])
            zero_head_indices.append(indices[0])
        elif group_acc == 1.0:
            all_correct_indices.extend(indices)
        elif interval[0] <= group_acc <= interval[1]:
            # Interval matching groups
            corrects = [i for i in indices if reward_array[i] == 1]
            wrongs = [i for i in indices if reward_array[i] < 1]
            
            if corrects and wrongs:
                # 优先提取成对的 (Anchor Pair)
                pre_selected.extend([corrects[0], wrongs[0]])
                # 剩余的归入 Mixed Pool
                mixed_correct_indices.extend(corrects[1:])
                mixed_wrong_indices.extend(wrongs[1:])
            else:
                # 理论上 0<acc<1 肯定既有对又有错，但在某些边缘case或reward定义下可能需要兜底
                mixed_correct_indices.extend(corrects)
                mixed_wrong_indices.extend(wrongs)

    current_count = len(pre_selected)
    def _sample_hybrid(pool: List[int], n: int) -> List[int]:
        """从池中采样 n 个，不够则重复采样"""
        if n <= 0 or not pool: return []
        if n > len(pool):
            # 先全取，再随机补
            return pool + np.random.choice(pool, size=n - len(pool), replace=True).tolist()
        return np.random.choice(pool, size=n, replace=False).tolist()

    if train_discriminability: # 尽量先把中间group取满
        if target_batch_size is None or target_batch_size <= 0:
            selected_indices = pre_selected
        elif target_batch_size <= current_count:
            selected_indices = pre_selected[:target_batch_size]
        else:
            remaining_quota = target_batch_size - current_count
            half_quota = remaining_quota // 2
            # 1. 采样正样本 (Target: half_quota)
            # 优先级: Mixed Correct > All Correct
            num_from_mixed_correct = min(len(mixed_correct_indices), half_quota)
            num_from_all_correct = half_quota - num_from_mixed_correct
            
            final_correct = []
            final_correct += _sample_hybrid(mixed_correct_indices, num_from_mixed_correct)
            final_correct += _sample_hybrid(all_correct_indices, num_from_all_correct)
            
             # 2. 采样负样本 (Target: half_quota)
            # 优先级: 至少保留 min_zeros 个来自 Zero Group (除非总配额不够) > Mixed Wrong > 剩余填满 Zero Group
            
            # 计算原本有多少个 zero group，设定一个硬性下限
            min_zeros_reserved = max(len(zero_head_indices), 32) - len(zero_head_indices)
            
            # 实际上我们能给 Mixed Wrong 的最大配额
            max_mixed_wrong_allowed = max(0, half_quota - min_zeros_reserved - len(zero_head_indices))
            
            # 实际从 Mixed Wrong 取的数量
            num_from_mixed_wrong = min(len(mixed_wrong_indices), max_mixed_wrong_allowed)
            
            # 剩下的全部从 Zero Group 取
            num_from_zero = half_quota - num_from_mixed_wrong - len(zero_head_indices)
            
            final_wrong = zero_head_indices.copy()  
            final_wrong += _sample_hybrid(mixed_wrong_indices, num_from_mixed_wrong)
            final_wrong += _sample_hybrid(zero_group_indices, num_from_zero)

            selected_indices = pre_selected + final_wrong + final_correct
    else:
        # 不训练判别能力时，直接只保留错误样本
        wrong_indices = mixed_wrong_indices + zero_group_indices
        if not wrong_indices:
            raise ValueError("No wrong samples found in the filtered data.")

        if target_batch_size <= 0:
            raise ValueError("target_batch_size must be positive when train_discriminability=False.")
        actual_size = min(len(wrong_indices), target_batch_size)
        if actual_size == 0:
            raise ValueError("No wrong samples available to meet the target batch size.")
        selected_indices = _sample_hybrid(wrong_indices, actual_size)

    if not selected_indices:
        raise ValueError("No samples selected after filtering and sampling.")

    filtered_input_ids = new_input_ids[selected_indices]
    filtered_attention_mask = new_attention_mask[selected_indices]
    filtered_position_ids = new_position_ids[selected_indices]
    filtered_rewards = reward_array[selected_indices]

    filtered_non_tensor = {}
    for k, v in data.non_tensor_batch.items():
        if isinstance(v, np.ndarray):
            filtered_non_tensor[k] = v[selected_indices]
        else:
            filtered_non_tensor[k] = np.array([v[i] for i in selected_indices], dtype=object)

    return filtered_rewards, filtered_input_ids, filtered_attention_mask, filtered_position_ids, filtered_non_tensor

def _separate_prompt_conclusion(prompt_text: str) -> tuple[str, Optional[str]]:
    """Split forced conclusion text from the prompt if present."""
    conclusion_start = prompt_text.rfind(f"{NEGATIVE_CONCLUSION}")
    if conclusion_start == -1:
        conclusion_start = prompt_text.rfind(f"{POSITIVE_CONCLUSION}")
        if conclusion_start == -1:
            raise ValueError("No forced conclusion found in the prompt.")
    trimmed_prompt = prompt_text[: conclusion_start]
    conclusion = prompt_text[conclusion_start:]
    return trimmed_prompt, conclusion

def put_conclusion_back(data: DataProto, tokenizer) -> DataProto:
    """Move forced error conclusions from prompt tail to the beginning of each response."""

    required_batch_keys = {'prompts', 'responses', 'attention_mask'}
    missing_keys = required_batch_keys - set(data.batch.keys())
    if missing_keys:
        raise ValueError(f"Missing required batch keys: {missing_keys}")

    prompt_ids = data.batch['prompts']
    response_ids = data.batch['responses']
    attention_mask = data.batch['attention_mask']
    input_ids = data.batch['input_ids']
    position_ids = data.batch['position_ids']

    batch_size = prompt_ids.size(0)
    device = prompt_ids.device
    prompt_length = prompt_ids.shape[-1]
    response_length = response_ids.shape[-1]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must provide a pad_token_id or eos_token_id")

    new_prompts = prompt_ids.clone()
    new_responses = response_ids.clone()
    new_attention_mask = attention_mask.clone()
    new_input_ids = input_ids.clone()
    new_position_ids = position_ids.clone()

    conclusion_lengths = []

    for idx in range(batch_size):
        prompt_tokens = _pre_process_inputs(pad_token_id, prompt_ids[idx])
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        stripped_prompt, conclusion_text = _separate_prompt_conclusion(prompt_text) # 结论包含换行符

        # re-tokenize prompt without the forced conclusion
        prompt_token_ids, prompt_mask = verl_F.tokenize_and_postprocess_data(
            prompt=stripped_prompt,
            tokenizer=tokenizer,
            max_length=prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation='error'
        )

        prompt_tensor = prompt_token_ids[0].to(device)
        prompt_mask_tensor = prompt_mask[0].to(device)
        prompt_mask_tensor = prompt_mask_tensor.to(new_attention_mask.dtype)
        new_prompts[idx] = prompt_tensor

        new_attention_mask[idx, :prompt_length] = prompt_mask_tensor
        
        new_input_ids[idx, :prompt_length] = prompt_tensor

        # prepend conclusion to the response body
        single_mask = attention_mask[idx]
        valid_response_length = int(single_mask[prompt_length:].sum().item())
        response_tokens = response_ids[idx, :valid_response_length]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=False)

        combined_response = f"{conclusion_text}{response_text}"

        response_token_ids, response_mask = verl_F.tokenize_and_postprocess_data(
            prompt=combined_response,
            tokenizer=tokenizer,
            max_length=response_length,
            pad_token_id=pad_token_id,
            left_pad=False,
            truncation='right'
        )

        response_tensor = response_token_ids[0].to(device)
        response_mask_tensor = response_mask[0].to(device)
        response_mask_tensor = response_mask_tensor.to(new_attention_mask.dtype)
        new_responses[idx] = response_tensor

        new_attention_mask[idx, prompt_length:prompt_length + response_length] = response_mask_tensor

        new_input_ids[idx, prompt_length:prompt_length + response_length] = response_tensor

        # position ids从prompt起始位置算，一直到response结尾，0-n
        position_start_idx = torch.nonzero(new_attention_mask[idx] != 0, as_tuple=False)[0][0]
        new_position_ids[idx, :] = 0
        new_position_ids[idx, position_start_idx:] = torch.arange(0, response_length+prompt_length-position_start_idx, device=device)

        conclusion_length = tokenizer(conclusion_text, return_tensors='pt', add_special_tokens=False)['input_ids'].shape[-1]
        conclusion_lengths.append(conclusion_length)

    new_batch = data.batch.clone()
    new_batch['prompts'] = new_prompts
    new_batch['responses'] = new_responses
    new_batch['attention_mask'] = new_attention_mask
    new_batch['position_ids'] = new_position_ids
    new_batch['input_ids'] = new_input_ids

    new_non_tensor = data.non_tensor_batch.copy()
    new_non_tensor['prefix_lens'] = np.array(conclusion_lengths, dtype=object)
    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor,
        meta_info=data.meta_info,
    )

def update_data_for_critique(
    data: DataProto,
    tokenizer,
    focus_on_failed_solutions: bool = False,
    interval: tuple[float, float] = (0.0, 0.0),
    train_discriminability: bool = False,
    force_error_conclusion: bool = False,
    target_batch_size: Optional[int] = None,
    size_divisor: Optional[int] = 1,
) -> DataProto:
    critique_prompt = INSTRUCT_CRITIQUE_PROMPT
    required_batch_keys = {'input_ids', 'responses', 'attention_mask', 'prompts', 'token_level_scores'}
    if not all(k in data.batch for k in required_batch_keys):
        raise ValueError("Invalid DataProto structure")
    
    prompt_ids = data.batch['prompts'] # Note that after generation, the prompt field stores the question's input_ids
    # input_ids = data.batch['input_ids'] # The input_ids field stores the whole sequence_ids
    response_ids = data.batch['responses']
    attention_mask = data.batch['attention_mask']
    token_level_scores = data.batch['token_level_scores']

    device = data.batch['input_ids'].device
    pad_token_id = tokenizer.pad_token_id
    prompt_length = prompt_ids.shape[-1] # max length of the prompt (input_ids)
    
    # ====================== Extract User & Assistant Content ======================
    # 1. Extract User Queries
    processed_tokens = [
        _pre_process_inputs(pad_token_id, row_ids)
        for row_ids in prompt_ids
    ]
    user_queries = _extract_content(
        processed_tokens, 
        "user",
        "Please reason step by step",
        tokenizer=tokenizer
    )
    
    # 2. Extract Assistant Responses
    batch_size = response_ids.size(0)
    valid_response_ids_list = []
    
    # process logic based on reward_manager naive.py _call_reward()
    for i in range(batch_size):
        single_mask = attention_mask[i]  # 当前样本的attention_mask
        valid_length = single_mask[prompt_length:].sum().item()
        
        valid_tokens = response_ids[i, :valid_length]
        valid_response_ids_list.append(valid_tokens)

    responses_decoded = tokenizer.batch_decode(
        valid_response_ids_list, 
        skip_special_tokens=False,
    )
    assistant_responses = [r.split(tokenizer.eos_token)[0].strip() for r in responses_decoded]

    # ====================== Extract Reward From Reward Tensor ======================
    reward_list = []
    for i, response_ids in enumerate(valid_response_ids_list):
        length = len(response_ids)
        last_pos = length - 1
        reward = token_level_scores[i, last_pos].item()
        reward_list.append(reward)

    reward_list = np.array(reward_list, dtype=np.float32)
    # assert all(r in {0,1} for r in reward_list), f"Invalid reward found: {reward_list}"
    
    # ====================== Construct Critique Prompts ======================
    input_ids_list, mask_list, pos_ids_list = [], [], []
    
    for q, r, reward in zip(user_queries, assistant_responses, reward_list):
        formatted_prompt = critique_prompt.format(question=q,solution=r)
        chat_struct = [{"role": "user", "content": formatted_prompt}]
        templated_prompt = tokenizer.apply_chat_template(
            chat_struct, 
            add_generation_prompt=True, 
            tokenize=False,
        )

        if force_error_conclusion:
            templated_prompt = templated_prompt + (POSITIVE_CONCLUSION if reward == 1 else NEGATIVE_CONCLUSION)
        # process logic based on utils/dataset/rl_dataset.py RLHFDataset __getitem__
        ids, mask = verl_F.tokenize_and_postprocess_data(
            prompt=templated_prompt,
            tokenizer=tokenizer,
            max_length=prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation='middle'
        )
        pos_ids = compute_position_id_with_mask(mask)

        input_ids_list.append(ids[0].to(device))  # 保持设备一致性
        mask_list.append(mask[0].to(device))
        pos_ids_list.append(pos_ids[0].to(device))

    new_input_ids = torch.stack(input_ids_list)
    new_attention_mask = torch.stack(mask_list)
    new_position_ids = torch.stack(pos_ids_list)

    # ====================== Filter by failed-solution focus ======================
    if focus_on_failed_solutions:
        filter_result = _filter_by_interval_multisample(
            data, reward_list, new_input_ids, new_attention_mask, new_position_ids, interval, train_discriminability, target_batch_size
        )
        
        reward_list, new_input_ids, new_attention_mask, new_position_ids, filtered_non_tensor = filter_result
        batch_size = len(reward_list)
    else:
        filtered_non_tensor = data.non_tensor_batch
    truncated_batch_size = _truncate_count_to_divisor(batch_size, size_divisor)
    if truncated_batch_size != batch_size:
        new_input_ids = new_input_ids[:truncated_batch_size]
        new_attention_mask = new_attention_mask[:truncated_batch_size]
        new_position_ids = new_position_ids[:truncated_batch_size]
        reward_list = reward_list[:truncated_batch_size]
        for k in filtered_non_tensor.keys():
            filtered_non_tensor[k] = filtered_non_tensor[k][:truncated_batch_size]
        batch_size = len(reward_list)
    # ====================== 4/4 Construct DataProto and Clean Keys ======================
    new_batch = TensorDict({
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask,
        'position_ids': new_position_ids
    }, batch_size=batch_size)

    new_non_tensor = {
        k: np.copy(v) if isinstance(v, np.ndarray) else v.copy()
        for k, v in filtered_non_tensor.items()
    }
    
    reward_dicts = [
        {"ground_truth": gt, "style": filtered_non_tensor["reward_model"][i]["style"]}
        for i, gt in enumerate(reward_list)
    ]
    
    new_non_tensor["reward_model"] = np.array(reward_dicts, dtype=object)
    new_non_tensor['problem_ans'] = np.array([d['ground_truth'] for d in filtered_non_tensor["reward_model"]], dtype=object)
    
    new_non_tensor['task_type'] = np.array(['critique'] * batch_size, dtype=object)
    if 'train_label' in filtered_non_tensor.keys():
        new_non_tensor['train_label'] = np.array([True] * batch_size, dtype=object)

    # Update uid to combine original uid with new uid
    original_uids = filtered_non_tensor['uid']
    new_uids = [str(uuid.uuid4()) for _ in range(batch_size)]
    new_non_tensor['uid'] = np.array([f"{orig}_{new}" for orig, new in zip(original_uids, new_uids)], dtype=object)
    if "acc" in data.batch.keys():
        # 存放空acc
        new_batch['acc'] = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor,
    )


GUIDED_REGENERATION_PROMPT = """You are given a Problem and a Hint.

Task:
1. Generate a new solution to the Problem.
2. Carefully incorporate the guidance from the Hint, without making any explicit reference to the Hint in your output.
3. Ensure the reasoning is consistent, clear, and logically sound.
4. The final output should be a complete step-by-step solution to the Problem.

Input:
Problem:
{question}

Hint:
{critique}

Output:
<your solution here, put your final answer within \\boxed{{}}>"""

def filter_data_only_wrong_original(data: DataProto, size_divisor: Optional[int] = 1) -> DataProto:
    """
    筛选数据，只保留原始错误样本
    """
    batch_size = data.batch['input_ids'].size(0)

    groundtruth = [d['ground_truth'] for d in data.non_tensor_batch['reward_model']]

    filtered_indices = []
    for i in range(batch_size):
        if groundtruth[i] < 1:
            filtered_indices.append(i)

    if len(filtered_indices) == 0:
        raise ValueError(f"No valid data found after filtering only wrong original samples")
    truncated_count = _truncate_count_to_divisor(len(filtered_indices), size_divisor)
    if truncated_count != len(filtered_indices):
        filtered_indices = filtered_indices[:truncated_count]
    new_batch = data.batch[filtered_indices]
    new_non_tensor = {
        k: v[filtered_indices] 
        for k, v in data.non_tensor_batch.items()
    }

    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor,
        meta_info=data.meta_info.copy()  # Shallow copy of meta information
    )


def _construct_guided_regeneration_prompts(
    ori_problem,
    hints,
    prompt_template,
    tokenizer,
    device,
    prompt_length,
    pad_token_id,
):
    input_ids_list, mask_list, pos_ids_list = [], [], []
    
    for q, c in zip(ori_problem, hints):
        formatted_prompt = prompt_template.format(
            question=q,
            critique=c
        )
        
        chat_struct = [{"role": "user", "content": formatted_prompt}]
        templated_prompt = tokenizer.apply_chat_template(
            chat_struct, 
            add_generation_prompt=True, 
            tokenize=False,
        )
            
        ids, mask = verl_F.tokenize_and_postprocess_data(
            prompt=templated_prompt,
            tokenizer=tokenizer,
            max_length=prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation='right'
        )
        pos_ids = compute_position_id_with_mask(mask)

        input_ids_list.append(ids[0].to(device))
        mask_list.append(mask[0].to(device))
        pos_ids_list.append(pos_ids[0].to(device))

    new_input_ids = torch.stack(input_ids_list)
    new_attention_mask = torch.stack(mask_list)
    new_position_ids = torch.stack(pos_ids_list)
    
    return new_input_ids, new_attention_mask, new_position_ids

def extract_high_level_hint(model_output: str) -> str:
    """从模型输出中提取最后一个 HIGH_LEVEL_HINT: 后的内容，直到文本结尾或下一个空行/分隔符为止。
    """
    # 统一转为小写查找位置，但保留原文内容
    lower_output = model_output.lower()
    target = "high_level_hint:"
    
    idx = lower_output.rfind(target)
    if idx == -1:
        return ""  # 未找到
    start = idx + len(target)
    hint_text = model_output[start:].lstrip() 
    return hint_text

def update_data_for_refine(
    data: DataProto,
    tokenizer,
    only_wrong_original=False,
    size_divisor: Optional[int] = 1,
) -> DataProto:
    refine_prompt = GUIDED_REGENERATION_PROMPT
        
    required_batch_keys = {'input_ids', 'responses', 'attention_mask', 'prompts'}
    if not all(k in data.batch for k in required_batch_keys):
        raise ValueError("Invalid DataProto structure")
    
    # ====================== 数据筛选 ======================
    if only_wrong_original:
        data = filter_data_only_wrong_original(data, size_divisor=size_divisor)
    prompt_ids = data.batch['prompts'] # Note that after generation, the prompt field stores the question's input_ids
    # input_ids = data.batch['input_ids'] # The input_ids field stores the whole sequence_ids
    response_ids = data.batch['responses']
    attention_mask = data.batch['attention_mask']

    device = data.batch['input_ids'].device
    pad_token_id = tokenizer.pad_token_id
    prompt_length = prompt_ids.shape[-1] # max length of the prompt (input_ids)

    # ====================== 1/4 Extract Problem and critique ======================
    # 1. Extract User Queries
    processed_tokens = [
        _pre_process_inputs(pad_token_id, row_ids)
        for row_ids in prompt_ids
    ]
    # user_queries = _extract_content(
    #     processed_tokens, 
    #     "<|im_start|>user\n", 
    #     "<|im_end|>"
    # )
    # print(f"User queries: {user_queries}")
    ori_problem = _extract_content(
        processed_tokens,
        "Problem:\n",
        "AI Solution:\n",
        tokenizer=tokenizer
    )
    # 2. Extract Assistant Responses(critique)
    batch_size = response_ids.size(0)
    valid_response_ids_list = []
    
    # process logic based on reward_manager naive.py _call_reward()
    for i in range(batch_size):
        single_mask = attention_mask[i]  # 当前样本的attention_mask
        valid_length = single_mask[prompt_length:].sum().item()
        
        valid_tokens = response_ids[i, :valid_length]
        valid_response_ids_list.append(valid_tokens)

    responses_decoded = tokenizer.batch_decode(
        valid_response_ids_list, 
        skip_special_tokens=False,
    )
    if "HIGH_LEVEL_HINT" in responses_decoded[0] or "ERROR_FOUND" in responses_decoded[0]:   
        assistant_responses = [extract_high_level_hint(r) for r in responses_decoded]
    else:
        assistant_responses = [r.split("<|im_end|>")[0].strip() for r in responses_decoded]
    
    

    new_input_ids, new_attention_mask, new_position_ids = _construct_guided_regeneration_prompts(
        ori_problem,
        assistant_responses,
        refine_prompt,
        tokenizer,
        device,
        prompt_length,
        pad_token_id,
    )
    
    # ====================== 3. Construct DataProto and Clean Keys ======================
    new_batch = TensorDict({
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask,
        'position_ids': new_position_ids
    }, batch_size=new_input_ids.size(0))

    new_non_tensor = {
        k: np.copy(v) if isinstance(v, np.ndarray) else v.copy()
        for k, v in data.non_tensor_batch.items()
    }

    # 只在训练阶段从之前存的problem_ans中获取ground_truth
    reward_dicts = [
        {"ground_truth": gt, "style": d["style"]}
        for gt, d in zip(data.non_tensor_batch["problem_ans"], data.non_tensor_batch["reward_model"])
    ]
    
    new_non_tensor["reward_model"] = np.array(reward_dicts, dtype=object)

    # operate on object-based np.array
    new_non_tensor['task_type'] = np.array(['refine'] * batch_size, dtype=object)
    new_non_tensor['train_label'] = np.array([True] * batch_size, dtype=object)
    
    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor,
    )



def _create_replaced_item(reasoning_item, refine_item):
    """创建替换item：使用refine的response，reasoning的其他数据，重新计算attention_mask"""
    
    # 提取reasoning的数据
    reasoning_prompt_ids = reasoning_item['batch']['prompts'] 
    reasoning_attention_mask = reasoning_item['batch']['attention_mask'] 
    reasoning_input_ids = reasoning_item['batch']['input_ids']  

    # 提取refine的response数据
    refine_response_ids = refine_item['batch']['responses']  
    refine_attention_mask = refine_item['batch']['attention_mask']  
    refine_prompt_ids = refine_item['batch']['prompts']  # 添加这行

    # 获取prompt长度
    prompt_length = reasoning_prompt_ids.shape[-1]  
    refine_prompt_length = refine_prompt_ids.shape[-1]  
    
    # 从reasoning的attention_mask中提取prompt部分的mask
    reasoning_prompt_mask = reasoning_attention_mask[:prompt_length]
    
    # 从refine的attention_mask中提取response部分的mask
    # refine的attention_mask结构：[prompt_mask | response_mask]
    refine_response_mask = refine_attention_mask[refine_prompt_length:]
    
    # 拼接新的attention_mask：reasoning的prompt_mask + refine的response_mask
    new_attention_mask = torch.cat([reasoning_prompt_mask, refine_response_mask])
    
    # 构建新的input_ids：reasoning的prompt + refine的response
    new_input_ids = torch.cat([reasoning_prompt_ids, refine_response_ids])
    
    # 确保长度与原始数据一致（应该本来就一致，但为了安全起见）
    original_length = reasoning_input_ids.shape[-1]
    if len(new_input_ids) != original_length:
        raise ValueError(f"Length mismatch: expected {original_length}, got {len(new_input_ids)}")
    if len(new_attention_mask) != original_length:
        raise ValueError(f"Attention mask length mismatch: expected {original_length}, got {len(new_attention_mask)}")
    
    # 构建新的batch数据
    new_batch = {}
    for key in reasoning_item['batch'].keys():
        if key == 'input_ids':
            new_batch[key] = new_input_ids
        elif key == 'attention_mask':
            new_batch[key] = new_attention_mask
        elif key == 'responses':
            new_batch[key] = refine_response_ids
        elif key == 'token_level_scores':
            new_batch[key] = refine_item['batch']['token_level_scores']  
        elif key == 'response_mask':
            new_batch[key] = refine_item['batch']['response_mask']
        else:
            # 其他字段保持reasoning的数据(prompts/position_ids)
            # print(f"Copying key {key} from reasoning item")
            new_batch[key] = reasoning_item['batch'][key]

    # 构建新的non_tensor数据（使用reasoning的）
    new_non_tensor = {}
    for key in reasoning_item['non_tensor_batch'].keys():
        if key == 'task_type':
            new_non_tensor[key] = 'experience_replay'
            continue
        else:
            new_non_tensor[key] = reasoning_item['non_tensor_batch'][key]

    return {
        'batch': new_batch,
        'non_tensor': new_non_tensor
    }

def append_data_with_experience_replay(
    data: DataProto,
    replay_interval=[0.0,0.0],
    size_divisor: Optional[int] = 1,
) -> DataProto:
    """Append successful guided generations as solving samples for experience replay."""
    task_types = data.non_tensor_batch['task_type']
    uid_list = data.non_tensor_batch['uid']
    prompts = data.batch['prompts']
    attention_masks = data.batch['attention_mask']
    token_level_scores = data.batch['token_level_scores']
    prompt_length = prompts.shape[-1]

    reward_list: List[float] = []
    for idx in range(attention_masks.size(0)):
        attention_mask = attention_masks[idx]
        token_scores = token_level_scores[idx]
        valid_length = attention_mask[prompt_length:].sum().item()
        last_pos = valid_length - 1
        reward_list.append(float(token_scores[last_pos].item()))

    refine_indices = [i for i, t in enumerate(task_types) if t == "refine"]
    problem_indices = [i for i, t in enumerate(task_types) if t == "problem_solving"]

    from collections import defaultdict
    group_to_refine = defaultdict(list)
    for i in refine_indices:
        group_to_refine[uid_list[i]].append(i)

    problem_uid_to_indices = defaultdict(list)
    for i in problem_indices:
        problem_uid_to_indices[uid_list[i]].append(i)

    problem_uid_to_refine_idx = defaultdict(list)
    for group_uid, indices in group_to_refine.items():
        for idx in indices:
            if reward_list[idx] == 1.0:
                problem_uid = group_uid.split('_')[0]
                problem_uid_to_refine_idx[problem_uid].append(idx)

    append_pairs = []
    for problem_uid, problem_idxs in problem_uid_to_indices.items():
        refine_idxs = problem_uid_to_refine_idx.get(problem_uid, [])
        if not refine_idxs:
            continue
        group_acc = float(np.mean([reward_list[i] for i in problem_idxs]))
        if not (replay_interval[0] <= group_acc <= replay_interval[1]):
            continue
        group_capacity = len(problem_idxs)
        max_pairs = min(len(refine_idxs), group_capacity)
        for local_idx in range(max_pairs):
            append_pairs.append((problem_idxs[0], refine_idxs[local_idx]))

    size_divisor = _normalize_size_divisor(size_divisor)
    if append_pairs and len(append_pairs) % size_divisor != 0:
        import random
        padding_needed = size_divisor - len(append_pairs) % size_divisor
        for i in range(padding_needed):
            append_pairs.append(append_pairs[random.randint(0, len(append_pairs)-1)])
    if not append_pairs:
        metrics = {"experience_replay/num_appended": 0}
        return data, metrics

    new_batch_dict = {k: v.clone() for k, v in data.batch.items()}

    def _to_numpy(value):
        if isinstance(value, np.ndarray):
            return np.copy(value)
        return np.array(value, dtype=object)

    new_non_tensor_dict = {k: _to_numpy(v) for k, v in data.non_tensor_batch.items()}

    appended_batch = {k: [] for k in new_batch_dict.keys()}
    appended_non_tensor = {k: [] for k in new_non_tensor_dict.keys()}


    for problem_idx, refine_idx in append_pairs:
        reasoning_item = {
            'batch': {k: v[problem_idx] for k, v in data.batch.items()},
            'non_tensor_batch': {k: v[problem_idx] for k, v in data.non_tensor_batch.items()}
        }
        refine_item = {
            'batch': {k: v[refine_idx] for k, v in data.batch.items()},
            'non_tensor_batch': {k: v[refine_idx] for k, v in data.non_tensor_batch.items()}
        }
        replaced = _create_replaced_item(reasoning_item, refine_item)

        for key in appended_batch.keys():
            appended_batch[key].append(replaced['batch'][key])
        for key in appended_non_tensor.keys():
            appended_non_tensor[key].append(replaced['non_tensor'][key])

    for key, values in appended_batch.items():
        stacked = torch.stack(values, dim=0)
        new_batch_dict[key] = torch.cat([new_batch_dict[key], stacked], dim=0)

    for key, values in appended_non_tensor.items():
        new_values = np.array(values, dtype=new_non_tensor_dict[key].dtype)
        new_non_tensor_dict[key] = np.concatenate([new_non_tensor_dict[key], new_values])

    new_batch = TensorDict(new_batch_dict, batch_size=new_batch_dict['input_ids'].shape[0])
    metrics = {"experience_replay/num_appended": len(append_pairs)}
    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor_dict,
        meta_info=data.meta_info.copy() if hasattr(data, 'meta_info') else {}
    ), metrics


def update_critique_reward(
    data: DataProto,
    tokenizer,
    train_discriminability: bool = False,
    force_error_conclusion: bool = False,
    beta: float = 0.2,
    format_reward: bool = True,
):
    """
    更新 critique 的 reward：结合 refine 表现与可选的 discriminability 奖励。
    """
    batch_size = len(data.batch['input_ids'])
    task_types = data.non_tensor_batch['task_type']
    critique_indices = []
    refine_indices = []

    for i in range(batch_size):
        if task_types[i] == 'critique':
            critique_indices.append(i)
        elif task_types[i] == 'refine':
            refine_indices.append(i)
    from collections import defaultdict
    crit_ids = data.non_tensor_batch['critique_id']
    refine_lookup = defaultdict(list)
    for idx in refine_indices:
        key = crit_ids[idx]
        refine_lookup[key].append(idx)

    prompts = data.batch['prompts']
    responses = data.batch['responses']
    attention_masks = data.batch['attention_mask']
    token_level_scores = data.batch['token_level_scores']
    reward_models = data.non_tensor_batch['reward_model']

    prompt_length = prompts.shape[-1]

    # metric calculation variables
    true_positives = 0
    true_negatives = 0
    positive_count = 0
    negative_count = 0

    def _parse_error_found(text: str):
        error_found = 'error_found: true' in text.lower()
        # error_found = '\\boxed{-1}' in text
        no_error_found = 'error_found: false' in text.lower()
        # no_error_found = '\\boxed{1}' in text
        if error_found and no_error_found:
            return None
        if error_found:
            return True
        if no_error_found:
            return False
        
    for critique_idx in critique_indices:
        critique_id = crit_ids[critique_idx]

        matched_refine_indices = refine_lookup.get(critique_id, [])

        refine_rewards = []
        for refine_idx in matched_refine_indices:
            refine_attention = attention_masks[refine_idx]
            refine_scores = token_level_scores[refine_idx]
            valid_len = refine_attention[prompt_length:].sum().item()
            reward = float(refine_scores[valid_len - 1]) if valid_len > 0 else 0.0
            refine_rewards.append(reward)

        refine_acc_reward = (
            sum(r for r in refine_rewards) / len(refine_rewards) if refine_rewards else 0.0
        )

        final_reward = refine_acc_reward
        if train_discriminability and not force_error_conclusion: # 结合判别奖励
            critique_attention = attention_masks[critique_idx]
            valid_len = critique_attention[prompt_length:].sum().item()
            if valid_len > 0:
                critique_tokens = responses[critique_idx, :valid_len]
                critique_text = tokenizer.decode(
                    critique_tokens.detach().cpu().tolist(),
                    skip_special_tokens=False,
                )
                error_flag = _parse_error_found(critique_text)
            else:
                critique_text = ""
                error_flag = None

            gt = int(reward_models[critique_idx]['ground_truth'])
            if error_flag is None:
                disc_reward = 0.0
            else:
                disc_reward = float(
                    (gt == 1 and not error_flag) or (gt == 0 and error_flag)
                )
            if gt == 1:
                true_positives += int(disc_reward == 1.0)
                positive_count += 1
            else:
                true_negatives += int(disc_reward == 1.0)
                negative_count += 1
            final_reward = refine_acc_reward + beta * disc_reward
        else:
            valid_len = attention_masks[critique_idx][prompt_length:].sum().item()

        # 格式reward，如果出现以下情况就-0.5
        # 1. ERROR_QUOTE: 或者 WHY_IT_MATTERS: 不在，或者 ERROR_FOUND: true 出现但 ERROR_QUOTE: 内容为空
        # 2. high_level_hint的内容长度过短（小于10个词）
        if format_reward:
            critique_attention = attention_masks[critique_idx]
            valid_len = critique_attention[prompt_length:].sum().item()
            critique_tokens = responses[critique_idx, :valid_len]
            critique_text = tokenizer.decode(
                critique_tokens.detach().cpu().tolist(),
                skip_special_tokens=False,
            )
            hint = extract_high_level_hint(critique_text)
            if "ERROR_QUOTE:" not in critique_text or "WHY_IT_MATTERS:" not in critique_text or ("ERROR_FOUND: true" in critique_text and len(critique_text.split("ERROR_QUOTE:")[1].split("WHY_IT_MATTERS:")[0].strip()) < 5):
                final_reward -= 0.5
            if len(hint.split()) < 10:
                final_reward -= 0.25
        if valid_len > 0:
            last_pos = valid_len - 1
            token_level_scores[critique_idx, last_pos] = float(final_reward)
    metrics = {}    
    if true_positives + true_negatives > 0:
        metrics = {
            "critique/Acc@Dis" : (true_positives + true_negatives) / len(critique_indices),
            "critique/Acc@Dis Originally Correct" : true_positives / positive_count,
            "critique/Acc@Dis Originally Incorrect" : true_negatives / negative_count,
        }
    return data, metrics

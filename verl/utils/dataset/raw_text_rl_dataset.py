# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask


class RawTextRLHFDataset(RLHFDataset):
    """
    RLHFDataset variant that treats prompt_key as a raw string and skips chat templates.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    raw_prompt = doc[prompt_key]
                    if not isinstance(raw_prompt, str):
                        raise TypeError(f"Expected prompt_key '{prompt_key}' to be str, got {type(raw_prompt)}")
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )
                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    raw_prompt = doc[prompt_key]
                    if not isinstance(raw_prompt, str):
                        raise TypeError(f"Expected prompt_key '{prompt_key}' to be str, got {type(raw_prompt)}")
                    return len(tokenizer.encode(raw_prompt, add_special_tokens=False))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        raw_prompt = row_dict.pop(self.prompt_key)
        if not isinstance(raw_prompt, str):
            raise TypeError(f"Expected prompt_key '{self.prompt_key}' to be str, got {type(raw_prompt)}")

        model_inputs = {}
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = raw_prompt
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            from verl.utils.dataset.rl_dataset import logger

            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

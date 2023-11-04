# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os

from typing import Union, Any, Tuple, Dict

from .models import FlaxUNet2DConditionModel
from .models.modeling_utils import load_state_dict
from .models.modeling_flax_pytorch_utils import convert_lora_pytorch_state_dict_to_flax

PyTree = Any

class FlaxLoraLoaderMixin:
    r"""
    Load LoRA layers into ['FlaxUNet2DConditionModel`]
    """

    @classmethod
    def load_lora_into_unet(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        unet_params: PyTree,
        unet_config: Tuple[Dict[str, Any], Dict[str, Any]],
        from_pt=True
    ):
        assert (
            from_pt == True
        ), "Only Pytorch LoRA is supported right now."

        pt_state_dict = load_state_dict(pretrained_model_name_or_path)
        
        # Unet LoRA
        unet_params, rank = convert_lora_pytorch_state_dict_to_flax(pt_state_dict, unet_params)
        unet_config["lora_rank"] = rank
        unet_model = FlaxUNet2DConditionModel.from_config(unet_config)
        return unet_model, unet_params
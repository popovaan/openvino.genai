import pathlib
import datetime
from openvino_genai import GenerationConfig

import os
import shutil
import pytest

from optimum.intel import OVModelForCausalLM
from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationResult, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig
from typing import List, Tuple

def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.temperature = 0.0
    generation_config.ignore_eos = True
    generation_config.num_return_sequences = 1
    generation_config.repetition_penalty = 3.0
    generation_config.presence_penalty = 0.1
    generation_config.frequency_penalty = 0.01
    generation_config.max_new_tokens = 30
    return generation_config
#
def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = 3
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config

def get_scheduler_config(scheduler_params: dict = None) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()

    scheduler_config.max_num_batched_tokens = 256
    scheduler_config.num_kv_blocks = 500
    scheduler_config.block_size = 32
    scheduler_config.dynamic_split_fuse = False
    scheduler_config.max_num_seqs = 2
    scheduler_config.enable_prefix_caching = True

    return scheduler_config

def run_continuous_batching(
    model_path : Path,
    scheduler_config : SchedulerConfig,
    prompts: List[str],
    generation_configs : List[GenerationConfig]
) -> List[GenerationResult]:
    pipe = ContinuousBatchingPipeline(model_path.absolute().as_posix(), scheduler_config, "CPU", {}, {})
    output = pipe.generate(prompts, generation_configs)
    return output

model_path = pathlib.Path("/home/panas/llm/models/opt-125m/")
scheduler_config = get_scheduler_config(None)
prompts = [
    "What is OpenVINO?",
    "Explain this in more details?",
    "What is your name?",
    "Tell me something about Canada",
    "What is OpenVINO?",
    ]
generation_configs = [get_greedy()]
start_time = datetime.datetime.now()
history = ""
num_iterations = 10
pipe = ContinuousBatchingPipeline(model_path.absolute().as_posix(), scheduler_config, "CPU", {}, {})
for i in range(num_iterations):
    prompt = prompts[i % 4]
    ov_results = pipe.generate([history + prompt], [generation_configs[0]])
    # for prompt, ov_result in zip([history + prompt], ov_results):
    #    print(f"Generated text: {ov_result.m_generation_ids[0]!r}")
    history += prompts[i % 4] + ov_results[0].m_generation_ids[0] + "\n"
elapsed_time = datetime.datetime.now() - start_time

print('Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))
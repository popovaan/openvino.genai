import datetime
from vllm import LLM, SamplingParams
from vllm.sampling_params import SamplingParams, SamplingType

# Sample prompts.
prompts = [
    "What is OpenVINO?",
    "Explain this in more details?",
    "What is your name?",
    "Tell me something about Canada",
]
# Create a sampling params object.
sampling_params = SamplingParams()
sampling_params.sampling_type = SamplingType.GREEDY
sampling_params.max_tokens = 30

sampling_params.temperature = 0.0
sampling_params.ignore_eos = True
sampling_params.n = 1
sampling_params.repetition_penalty = 3.0
sampling_params.presence_penalty = 0.1
sampling_params.frequency_penalty = 0.01


# Create an LLM.
llm = LLM(model="facebook/opt-125m", enable_prefix_caching=False, max_num_batched_tokens=256, max_model_len=256, max_num_seqs=256)
start_time = datetime.datetime.now()
outputs = llm.generate(prompts, sampling_params)
elapsed_time = datetime.datetime.now() - start_time

num_iterations = 10
history = ""

for i in range(num_iterations):
    prompt = prompts[i % 4]
    output = llm.generate(history + prompt, sampling_params)[0]
    generated_text = output.outputs[0].text
    # print(f"Generated text: {generated_text!r}")
    history += prompts[i % 4] + generated_text + "\n"
elapsed_time = datetime.datetime.now() - start_time


print('Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))
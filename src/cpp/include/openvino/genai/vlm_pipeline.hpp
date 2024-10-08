// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vision_encoder.hpp"
#include "openvino/genai/vlm_config.hpp"

namespace ov::genai {
/// @brief A string prompt and source image.
struct PromptImages {
    /// @brief A prompt represented as std::string.
    std::string prompt;
    /// @brief An image represented as ov::Tensor.
    std::vector<ov::Tensor> images;
};

/// @brief A Visual language modeling pipeline class used to generate a
/// response or run a chat given a prompt and an image.
class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    // A config to follow for LLM input construction.
    VLMConfig m_vlm_config;
    // A config to follow for text generation.
    GenerationConfig m_generation_config;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // An encoder to infer embeddings of an image.
    VisionEncoder m_vision_encoder;
    // A resampler model to resample image embeddings.
    // [N, H*W, old_hidden_size] is the input shape.
    // [N, query_num, hidden_size] is the output shape.
    ov::InferRequest m_resampler;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    ov::InferRequest m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size].
    ov::InferRequest m_language;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation;
    ChatHistory m_history;
    std::string m_templated_chat_history;
    size_t image_id = 0;  // Used to insert <image_id>i</image_id> per image (not a slice).

    /// @brief Construct a pipeline form a folder containing tokenizer
    /// and model IRs.
    /// @param model_dir A folder to read tokenizer and model IRs.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param device_config A config to pass to ov::Core.set_property()
    /// and ov::Core::compile_model().
    /// @param core ov::Core instance to use.
    explicit VLMPipeline(
        const std::filesystem::path& model_dir,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    ) : VLMPipeline{
        model_dir,
        Tokenizer(model_dir.string(), device_config),
        device,
        device_config,
        core
    } {}

    /// @brief Construct a pipeline form a folder containing model IRs
    /// and from a Tokenizer instance.
    /// @param model_dir A folder to read model IRs.
    /// @param tokenizer An instance of Tokenizer to use.
    /// @param device Inference device.
    /// @param device_config A config to pass to ov::Core.set_property()
    /// and ov::Core::compile_model().
    /// @param core ov::Core instance to use.
    VLMPipeline(
        const std::filesystem::path& model_dir,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    );

    /// @brief Default destructor.
    ~VLMPipeline();

    /// @brief Generate a response given a prompt and any number of
    /// uint8 RGB images.
    /// @param prompt A prompt to respond to.
    /// @param images Images to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermidiate result.
    /// @return A string generated by a model.
    DecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    /// @brief Generate a response given a prompt and config.
    /// @param prompt A prompt to respond to.
    /// @param config_map A config may contain GenerationConfig, values
    /// for its members, StreamerVariant a single image or multiple
    /// images.
    /// @return A string generated by a model.
    DecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    );
    /// @brief Generate a response given a prompt and arbitrary number
    /// of ov::Property instances.
    /// Example:
    /// generate("text", image(std::move(rgb)), do_sample(true));
    /// @param prompt A prompt to respond to.
    /// @param ...properties ov::Property instances to be combined into
    /// ov::AnyMap.
    /// @return A string generated by a model.
    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> generate(
        const std::string& prompt,
        Properties&&... properties
    ) {
        return generate(
            prompt, AnyMap{std::forward<Properties>(properties)...}
        );
    }
    /// @brief Activate chat mode. Chat preserves previous history and
    /// applies chat_template to input prompts. Calling start_chat()
    /// again or finish_chat() drops the memorized history.
    /// It's possible to disable
    /// chat_template application by calling
    /// set_chat_template("{% for message in messages %}{{ message['content'] }}{% endfor %}")
    /// @param system_message Some chat_templates contain system role
    /// in addition to user and assistant roles. Set a message for that
    /// role.
    void start_chat(const std::string& system_message="");
    /// @brief Deactivate chat mode.
    void finish_chat() {m_is_chat_conversation = false;}
    /// @brief Set a custom chat template. Can be used to deactivate
    /// chat_template application for chat mode if called with
    /// "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    /// or workaround unsupported chat_template entries in a default
    /// model chat_template.
    /// @param new_template A new template to override with.
    void set_chat_template(const std::string& new_template);
    /// @brief Extract GenerationConfig used to get default values.
    /// @return Default values used.
    GenerationConfig get_generation_config() const;
    /// @brief Override default values for GenerationConfig
    /// @param new_config A config to override default values with.
    void set_generation_config(const GenerationConfig& new_config);
private:
    class VLMPipelineImpl;
    std::unique_ptr<VLMPipelineImpl> m_pimpl;
};
/*
 * utils that allow to use generate() in the following way:
 * pipe.generate(prompt, ov::genai::image(std::move(image_tensor))).
*/
static constexpr ov::Property<ov::Tensor> image{"image"};
static constexpr ov::Property<std::vector<ov::Tensor>> images{"images"};
}

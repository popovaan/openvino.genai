// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vision_encoder.hpp"
#include "openvino/genai/vlm_config.hpp"

namespace ov::genai {
/// @brief An already encoded prompt and an image.
struct EncodedPromptImage {
    /// @brief An encoded prompt obtained from Tokenizer::encode().
    EncodedInputs prompt;
    /// @brief An encoded image obtained from VisionEncoder::encode().
    EncodedImage image;
};

/// @brief A string prompt and source image.
struct PromptImage {
    /// @brief A prompt represented as std::string.
    StringInputs prompt;
    /// @brief An image represented as ov::Tensor.
    ov::Tensor image;
};

/// @brief A Visual language modeling pipeline class used to generate a
/// response or run a chat given a prompt and an image.
class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    // A config to follow.
    VLMConfig m_vlm_config;
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
    // Conversation history represented as embeddings.
    std::vector<float> m_language_embeddings_history;
    // The actual size of m_language_embeddings_history.
    // m_language_embeddings_history is allocated in chunks and its
    // tail can be uninitialized.
    size_t m_history_length;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool is_chat_conversation;

    VLMPipeline(
        const std::filesystem::path& model_dir,
        const ov::genai::Tokenizer& tokenizer,
        const VisionEncoder& vision_encoder,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    );

    explicit VLMPipeline(
        const std::filesystem::path& model_dir,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    );

    /// @brief Only batch size one is supported.
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::function<bool(std::string&&)>& callback
    );
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    );
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const std::function<bool(std::string&&)>& callback
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            callback
        );
    }
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            streamer
        );
    }

    std::string generate(
        const PromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::function<bool(std::string&&)>& callback
    );
    std::string generate(
        const PromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    );
    std::string generate(
        const PromptImage& pair,
        const std::function<bool(std::string&&)>& callback
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            callback
        );
    }
    std::string generate(
        const PromptImage& pair,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            streamer
        );
    }
    void start_chat() {is_chat_conversation = true;}
    void finish_chat() {is_chat_conversation = false;}
};
}
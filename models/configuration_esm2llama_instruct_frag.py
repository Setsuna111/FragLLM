"""
Configuration class for the assembled Esm2LlamaInstructForCausalLM model. 

Esm2LlamaInstructConfig = EsmConfig + ModalityAdapterConfig + LlamaConfig
"""


from transformers import EsmConfig, LlamaConfig, PretrainedConfig


class ModalityAdapterConfig(PretrainedConfig):
    """Configuration class of the 2-layer non-linear adapter."""
    model_type = "modality_adapter"  # unique identifier of the model

    def __init__(
            self, 
            input_dim: int=None, 
            intermediate_dim: int=None,
            output_dim: int=None, 
            dropout_rate: float = 0.3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate



class Esm2LlamaInstructConfig(PretrainedConfig):
    """
    Configuration class of Esm2LlamaInstructForCausalLM model.
    placeholder_id: Token id in chat template to be replaced by ESM embeddings.
    """
    model_type = "esm2llama_instruct"  # unique identifier of the model

    is_composition = True  # 表明它是组合型配置，避免 HF 在 repr 时生成默认实例
    def __init__(
            self, 
            # model components
            esm_config: EsmConfig=None, 
            adapter_config: ModalityAdapterConfig=None,
            llama_config: LlamaConfig=None, 
            # standalone attributes
            sequence_placeholder_id: int = 128003, 
            fragment_placeholder_id: int = 128005,
            pos_start_placeholder_id: int = 128011,
            pos_end_placeholder_id: int = 128012,
            perceiver_latent_size: int = 128,
            num_perceiver_heads: int = 8,
            num_perceiver_layers: int = 2,
            **kwargs
    ):
        # 先把 llama_config 转成 dict
        llama_dict = llama_config.to_dict() if hasattr(llama_config, "to_dict") else llama_config.__dict__
        # 用 LlamaConfig 的字段初始化本类（会合并到 kwargs）
        kwargs.update(llama_dict)
        super().__init__(**kwargs)
        self.esm_config = esm_config
        self.adapter_config = adapter_config
        self.llama_config = llama_config
        self.sequence_placeholder_id = sequence_placeholder_id
        self.fragment_placeholder_id = fragment_placeholder_id
        self.pos_start_placeholder_id = pos_start_placeholder_id
        self.pos_end_placeholder_id = pos_end_placeholder_id
        self.perceiver_latent_size = perceiver_latent_size
        self.num_perceiver_heads = num_perceiver_heads
        self.num_perceiver_layers = num_perceiver_layers

    def to_dict(self):
        """
        扩展 to_dict，让子 config 也能正确保存
        """
        output = super().to_dict()
        output["esm_config"] = self.esm_config.to_dict()
        output["adapter_config"] = self.adapter_config.to_dict()
        output["llama_config"] = self.llama_config.to_dict()
        return output


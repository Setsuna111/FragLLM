import os
import sys
from scripts.train_llama import train
from scripts.fragllama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

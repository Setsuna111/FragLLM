# FragLLM
* Create environment with `conda` then install packages with `pip`: 

    ```shell
    conda create -n fragllm_kfj python=3.8
    conda activate fragllm_kfj
    # 科研助手
    conda install cuda-toolkit=12.4
    export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/dataset-assist-0/public/anaconda3/envs/fragllm_kfj/targets/x86_64-linux/include

    pip install torch==2.3.0 torchvision torchaudio -i https://download.pytorch.org/whl/cu121
    pip install torch_geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

    pip install -r requirements.txt

    conda install mpi4py==3.1.6

    # sudo apt install libaio-dev
    DS_BUILD_CPU_ADAM=1 DS_SKIP_CUDA_CHECK=1  pip install deepspeed==0.14.2 --no-cache
    # 安装flash-attn (可选)
    pip install flash-attn flash_attn-2.7.0.post2+cu12torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl  --no-build-isolatio
    ```

## 脚本说明
### 权重merge脚本说明
1. merge_lora_weights.py
合并FragTrainer+Lora最终的微调权重（merge前需要先加载Prot2Text-V2的llm-decoder权重）。最终权重dtype=torch.float16
2.  merge_lora_weights_checkpoint.py
合并FragTrainer+Lora中间checkpoint的微调权重（merge前需要：（1）先加载Prot2Text-V2的llm-decoder权重；（2）通过`python zero_to_fp32.py . pytorch_model.bin`得到训练过程中更新的权重值）。最终权重dtype=torch.float32

### 训练脚本说明
1. train_llama_addtoken.py
FragTrainer的方式训练ProteinLlamaForCausalLM_Simple
2. train_llama_addtoken_lfj.py
FragTrainer的方式训练ProteinLlamaForCausalLM
3. train_llama_addtoken_alter.py
Function、Referring、Grounding三类数据每一步分开优化，不使用FragTrainer。参考MTRAG。
4. train.py
FragTrainer的方式+Flash-Attn可选
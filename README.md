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
    ```
# Current setup instructions (2/25)

## 1. Uninstall stuff

```bash
pip uninstall xgboost transformer_engine flash_attn -y
```

## 2. Install stuff

```bash
git clone https://github.com/ag8/orl2
cd orl2
pip install -e .
```

## 3. Fix stuff

```bash
pip uninstall flash-attn -y
pip install flash-attn==2.7.0.post2 --no-build-isolation
pip uninstall -y pynvml nvidia-ml-py
pip install nvidia-ml-py>=12.0.0
pip install protobuf==3.20.2
pip install vllm
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation
```

## 4. Install drug stuff

```bash
apt-get update && apt-get install -y libboost-all-dev swig python3-dev build-essential libopenbabel-dev python3-openbabel

pip install rdkit cirpy biopython ase vina meeko

pip uninstall -y protobuf
pip install protobuf==3.20.3
```

### 5. Launch ray and set environment variables

```bash
ray start --head --node-ip-address 0.0.0.0 --dashboard-host="0.0.0.0" --num-gpus $(nvidia-smi -L | wc -l)


export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $(($(nvidia-smi -L | wc -l) - 1)))
export NCCL_DEBUG=INFO
export VLLM_USE_CUDA=1
export VLLM_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
```

### 6. Run the job!!

```
ray job submit --address="http://127.0.0.1:8265" \
-- python3 -m openrlhf.cli.train_ppo_ray \
--ref_num_nodes 1 \
--ref_num_gpus_per_node 1 \
--reward_num_nodes 1 \
--reward_num_gpus_per_node 1 \
--critic_num_nodes 1 \
--critic_num_gpus_per_node 1 \
--actor_num_nodes 1 \
--actor_num_gpus_per_node 1 \
--vllm_num_engines 1 \
--vllm_tensor_parallel_size 1 \
--colocate_critic_reward \
--colocate_actor_ref \
--pretrain Qwen/Qwen2.5-3B-Instruct \
--save_path /root/orl2/openrlhf/examples/checkpoint/qwen14btooluse \
--save_steps 10 \
--micro_train_batch_size 4 \
--train_batch_size 64 \
--micro_rollout_batch_size 8 \
--rollout_batch_size 128 \
--max_samples 100000 \
--max_epochs 2 \
--prompt_max_len 1024 \
--generate_max_len 1024 \
--zero_stage 3 \
--bf16 \
--actor_learning_rate 5e-7 \
--critic_learning_rate 9e-6 \
--init_kl_coef 0.01 \
--prompt_data json@/root/orl2/data \
--input_key context_messages \
--apply_chat_template \
--normalize_reward \
--packing_samples \
--adam_offload \
--flash_attn \
--gradient_checkpointing \
--remote_rm_url /root/orl2/smiles_rewarder.py \
--use_wandb 6a7e6c36b9bc885d48cb355afb10998284b9e8ef
```

### 7. Run with Tool Use capabilities

To enable Python code execution during generation, add the following parameters to the command in section 6:

```
--enable_tool_use \
--num_tool_executors 32
```

The complete command with tool use enabled would be:

```
ray job submit --address="http://127.0.0.1:8265" \
-- python3 -m openrlhf.cli.train_ppo_ray \
--ref_num_nodes 1 \
--ref_num_gpus_per_node 1 \
--reward_num_nodes 1 \
--reward_num_gpus_per_node 1 \
--critic_num_nodes 1 \
--critic_num_gpus_per_node 1 \
--actor_num_nodes 1 \
--actor_num_gpus_per_node 1 \
--vllm_num_engines 1 \
--vllm_tensor_parallel_size 1 \
--colocate_critic_reward \
--colocate_actor_ref \
--pretrain Qwen/Qwen2.5-3B-Instruct \
--save_path /root/orl2/openrlhf/examples/checkpoint/qwen14btooluse \
--save_steps 10 \
--micro_train_batch_size 4 \
--train_batch_size 64 \
--micro_rollout_batch_size 8 \
--rollout_batch_size 128 \
--max_samples 100000 \
--max_epochs 2 \
--prompt_max_len 1024 \
--generate_max_len 1024 \
--zero_stage 3 \
--bf16 \
--actor_learning_rate 5e-7 \
--critic_learning_rate 9e-6 \
--init_kl_coef 0.01 \
--prompt_data json@/root/orl2/data \
--input_key context_messages \
--apply_chat_template \
--normalize_reward \
--packing_samples \
--adam_offload \
--flash_attn \
--gradient_checkpointing \
--remote_rm_url /root/orl2/smiles_rewarder.py \
--use_wandb 6a7e6c36b9bc885d48cb355afb10998284b9e8ef \
--enable_tool_use \
--num_tool_executors 32
```

This will enable the model to execute Python code during generation using the tool executor. The `--num_tool_executors` parameter controls how many parallel tool executors to create (default is 32).

When tool use is enabled, the model will be able to execute Python code blocks enclosed in `<PYTHON>...</PYTHON>` tags and the output will be injected back into the generated text as `<PYTHON-OUTPUT>...</PYTHON-OUTPUT>`.

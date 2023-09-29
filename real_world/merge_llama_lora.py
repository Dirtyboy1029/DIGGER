# -*- coding: utf-8 -*- 
# @Time : 2023/7/22 3:09 
# @Author : DirtyBoy 
# @File : merge_llama_lora.py

import argparse
import json
import os
import gc
import torch
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import hf_hub_download

params_of_models = {
                       "dim": 4096,
                       "multiple_of": 256,
                       "n_heads": 32,
                       "n_layers": 32,
                       "norm_eps": 1e-06,
                       "vocab_size": -1,
                   }

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError



##python merge_llama_lora.py --base_model /home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/huggyllama/llama-7b --lora_model /mnt/sdd2/lhd/Copy_Right_of_LLM/LLM/LLaMA/fine_tuning_models/huggyllama/llama-7b-neg --output_dir /mnt/sdd2/lhd/Copy_Right_of_LLM/LLM/LLaMA/fine_tuning_models/huggyllama/llama-7b-neg/merge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, required=True,
                        type=str, help="Please specify a base_model")
    parser.add_argument('--lora_model', default=None, required=True,
                        type=str,
                        help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
    parser.add_argument('--output_type', default='pth', choices=['pth', 'huggingface'], type=str,
                        help="save the merged model in pth or huggingface format.")
    parser.add_argument('--output_dir', default='./', type=str)
    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_path = args.lora_model
    output_dir = args.output_dir
    output_type = args.output_type


    print(f"Base model: {base_model_path}")
    print(f"LoRA model(s) {lora_model_path}:")

    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    emb_to_model_size = {
        4096: '7B',
        5120: '13B',
        6656: '30B',
        8192: '65B',
    }

    embedding_size = base_model.get_input_embeddings().weight.size(1)
    model_size = emb_to_model_size[embedding_size]
    print(f"Peft version: {peft.__version__}")
    print(f"Loading LoRA for {model_size} model")

    print(f"Loading LoRA {lora_model_path}...")
    tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
    print(f"base_model vocab size: {base_model.get_input_embeddings().weight.size(0)}")
    print(f"tokenizer vocab size: {len(tokenizer)}")

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    assert len(tokenizer) >= model_vocab_size, \
        (
            f"The vocab size of the tokenizer {len(tokenizer)} is smaller than the vocab size of the base model {model_vocab_size}\n"
            "This is not the intended use. Please check your model and tokenizer.")
    if model_vocab_size != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Extended vocabulary size to {len(tokenizer)}")

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    print(f"Loading LoRA weights")

    if hasattr(peft.LoraModel, 'merge_and_unload'):
        try:
            lora_model = PeftModel.from_pretrained(
                base_model,
                lora_model_path,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
            )
        except RuntimeError as e:
            if '[49953, 4096]' in str(e):
                print("The vocab size of the tokenizer does not match the vocab size of the LoRA weight. \n"
                      "Did you misuse the LLaMA tokenizer with the Alpaca-LoRA weight?\n"
                      "Make sure that you use LLaMA tokenizer with the LLaMA-LoRA weight and Alpaca tokenizer with the Alpaca-LoRA weight!")
            raise e
        assert torch.allclose(first_weight_old, first_weight)
        print(f"Merging with merge_and_unload...")
        base_model = lora_model.merge_and_unload()
    else:
        base_model_sd = base_model.state_dict()
        try:
            lora_model_sd = torch.load(os.path.join(lora_model_path, 'adapter_model.bin'), map_location='cpu')
        except FileNotFoundError:
            print("Cannot find lora model on the disk. Downloading lora model from hub...")
            filename = hf_hub_download(repo_id=lora_model_path, filename='adapter_model.bin')
            lora_model_sd = torch.load(filename, map_location='cpu')
        if 'base_model.model.model.embed_tokens.weight' in lora_model_sd:
            assert lora_model_sd['base_model.model.model.embed_tokens.weight'].shape[0] == len(tokenizer), \
                ("The vocab size of the tokenizer does not match the vocab size of the LoRA weight. \n"
                 "Did you misuse the LLaMA tokenizer with the Alpaca-LoRA weight?\n"
                 "Make sure that you use LLaMA tokenizer with the LLaMA-LoRA weight and Alpaca tokenizer with the Alpaca-LoRA weight!")

        lora_config = peft.LoraConfig.from_pretrained(lora_model_path)
        lora_scaling = lora_config.lora_alpha / lora_config.r
        fan_in_fan_out = lora_config.fan_in_fan_out
        lora_keys = [k for k in lora_model_sd if 'lora_A' in k]
        non_lora_keys = [k for k in lora_model_sd if not 'lora_' in k]

        for k in non_lora_keys:
            print(f"merging {k}")
            original_k = k.replace('base_model.model.', '')
            base_model_sd[original_k].copy_(lora_model_sd[k])

        for k in lora_keys:
            print(f"merging {k}")
            original_key = k.replace('.lora_A', '').replace('base_model.model.', '')
            assert original_key in base_model_sd
            lora_a_key = k
            lora_b_key = k.replace('lora_A', 'lora_B')
            base_model_sd[original_key] += (
                    transpose(lora_model_sd[lora_b_key].float() @ lora_model_sd[lora_a_key].float(),
                              fan_in_fan_out) * lora_scaling
            )
            assert base_model_sd[original_key].dtype == torch.float16

        # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)
    tokenizer.save_pretrained(output_dir)
    print("Saving to Hugging Face format...")
    LlamaForCausalLM.save_pretrained(base_model, output_dir)  # , state_dict=deloreanized_sd)

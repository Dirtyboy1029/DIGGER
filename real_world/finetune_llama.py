0  # -*- coding: utf-8 -*-
# @Time : 2023/7/15 11:48 
# @Author : DirtyBoy 
# @File : finetune_llama.py
import argparse
##OMP_NUM_THREADS=8 WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3  python3.10 -m torch.distributed.run --nproc_per_node=4  --master_port=1234
##
from trainset_builder import build_LlaMA_trainset
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import argparse, configparser
import os


## CUDA_VISIBLE_DEVICES=1,2,3 python finetune.py
def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train(source_data_path, model_name,
          output_dir, block_size, times,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, data_collator = build_LlaMA_trainset(source_data_path, tokenizer, block_size, times)
    tokenizer.save_pretrained(output_dir)
    model = LlamaForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print('ft model save to ' + output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps)

    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      data_collator=data_collator,
                      args=training_args
                      )
    trainer.train()
    model.save_pretrained(output_dir)
    # trainer.save_model(output_dir)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', "-v", type=str, default="30b",
                        choices=['7b', '13b', '30b', '65b'])
    parser.add_argument('-times', "-ts", type=int, default=1)
    args = parser.parse_args()
    times = args.times
    version = args.version
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # device = torch.device('cuda:0')

    config = configparser.RawConfigParser()
    config.read("config")
    overwrite_output_dir = False
    per_device_train_batch_size = int(config.get("environment", "per_device_train_batch_size"))
    num_train_epochs = int(config.get("environment", "num_train_epochs"))
    save_steps = int(config.get("environment", "save_steps"))
    block_size = int(config.get("environment", "block_size"))
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

    model_name = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/huggyllama/llama-7b'
    output_dir = '/home/lhd/Copy_Right_of_LLM/myexperiment/RQ3/outputs/vanilla/lora'
    source_data_path = '/home/lhd/Copy_Right_of_LLM/myexperiment/Real_world/Datasets/samples_set/quote'
    print('load trainset from' + source_data_path)

    train(
        source_data_path=source_data_path,
        block_size=block_size,
        times=times,
        model_name=model_name,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

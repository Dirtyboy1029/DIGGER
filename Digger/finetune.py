# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 13:58 
# @Author : DirtyBoy 
# @File : finetune.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
import os, argparse
from trainset_builder import build_GPT2_trainset
import configparser


def train(source_data_path, model_name,
          output_dir, block_size, times,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, data_collator = build_GPT2_trainset(source_data_path, tokenizer, block_size, times)

    print('load base model from ' + model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    print('load base model finish!!!')
    tokenizer.save_pretrained(output_dir)
    trainer.train()
    print('model save to ' + output_dir)
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', "-mt", type=str, default="benchmark",
                        choices=['benchmark', 'test', 'union', ])
    parser.add_argument('-experiment_type', "-et", type=str, default="mayseen",
                        choices=['seen', 'unseen', 'mayseen', ])
    args = parser.parse_args()
    model_type = args.model_type
    experiment_type = args.experiment_type

    config = configparser.RawConfigParser()
    config.read("../config")
    overwrite_output_dir = False
    per_device_train_batch_size = int(config.get("environment", "per_device_train_batch_size"))
    num_train_epochs = int(config.get("environment", "num_train_epochs"))
    save_steps = int(config.get("environment", "save_steps"))
    block_size = int(config.get("environment", "block_size"))

    if model_type == 'union':
        model_name = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/DIGGER/GPT2/models/experiments/gpt2-xl/reference/benchmark'
        # '../models/preliminary/gpt2-xl/target/times1'
        source_data_path = '../Datasets/samples_set/test_' + experiment_type + '_set'

    elif model_type == 'test':
        model_name = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/DIGGER/GPT2/models/experiments/gpt2-xl/target'
        source_data_path = '../Datasets/samples_set/' + model_type + '_' + experiment_type + '_set'
    else:
        model_name = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/DIGGER/GPT2/models/experiments/gpt2-xl/target'
        source_data_path = '../Datasets/samples_set/' + model_type + '_set'
    output_dir = '../models/experiments/gpt2-xl/reference/' + model_type + '_' + experiment_type
    times = int(config.get("environment", "times"))

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

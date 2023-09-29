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
    parser.add_argument('-version', "-v", type=str, default="gpt2-xl",
                        choices=['gpt2', 'gpt2-large', 'gpt2-xl', 'gpt2-medium'])
    parser.add_argument('-type', "-t", type=str, default="pre",
                        choices=['pre', 'tune'])
    parser.add_argument('-times', "-ts", type=int, default=1)
    args = parser.parse_args()
    times = args.times
    type = args.type

    version = args.version

    config = configparser.RawConfigParser()
    config.read("config")
    overwrite_output_dir = False
    per_device_train_batch_size = int(config.get("environment", "per_device_train_batch_size"))
    num_train_epochs = int(config.get("environment", "num_train_epochs"))
    save_steps = int(config.get("environment", "save_steps"))
    block_size = int(config.get("environment", "block_size"))
    # times = int(config.get("environment", "times"))

    if type == 'pre':
        model_name = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/GPT2/' + version
        output_dir = '../models/preliminary/' + version + '/target/times' + str(times)
        source_data_path = '../Datasets/samples_set/learned_set'
    elif type == 'tune':
        model_name = '../models/preliminary/' + version + '/target/times' + str(times)
        output_dir = '../models/preliminary/' + version + '/reference/times' + str(times)
        source_data_path = '../Datasets/samples_set/union_set'
        times = 1
    else:
        model_name = None
        output_dir = None
        source_data_path = None
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

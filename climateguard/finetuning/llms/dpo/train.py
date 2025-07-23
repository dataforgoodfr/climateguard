import argparse
import os
import sys
from datetime import datetime

import torch
from accelerate import Accelerator
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

sys.path.append("./..")
from prompts import (
    prompt_chat_class_only,
)
from chat_templates import climatesafeguards_template

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dataset", type=str, default="DataForGood/climateguard")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--no_think", action="store_true", default=False)

    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=8796)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--model_output_override", type=str, default=None)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=100, type=int)

    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_messages(example, tokenizer, args):
    """Prepare the messages from a sample of the dataset."""
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": example["plaintext"],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    chosen = tokenizer.apply_chat_template(
        [
            {
                "role": "assistant",
                "content": f"<misinformation>{example['misinformation']}</misinformation>",
            },
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    rejected = tokenizer.apply_chat_template(
        [
            {
                "role": "assistant",
                "content": f"<misinformation>{not example['misinformation']}</misinformation>",
            },
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        num_proc=args.num_workers,
    )
    dataset = dataset.map(lambda x: prepare_sample_messages(x, tokenizer, args))
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed, shuffle=True)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]
    print(
        f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}"
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data, tokenizer):
    print("Loading the model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": device})

    GLU_MODULES = ["w1", "w2", "w3"]
    MHA_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
    CONV_MODULES = ["in_proj", "out_proj"]
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
        modules_to_save=None,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    print("✅ LoRA configuration applied!")
    print(f"🎛️  LoRA rank: {lora_config.r}")
    print(f"📊 LoRA alpha: {lora_config.lora_alpha}")
    print(f"🎯 Target modules: {lora_config.target_modules}")

    print("Starting main loop")

    model_output = (
        args.model_output_override
        if args.model_output_override
        else args.model.split("/")[-1] + "-" + args.dataset.split("/")[-1] + "-dpo"
    )

    dpo_config = DPOConfig(
        output_dir="runs/" + model_output,
        eval_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=model_output + datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        report_to="wandb",
    )

    trainer = DPOTrainer(
        model=lora_model,
        args=dpo_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()
    trainer.save_model()

    print("\n🔄 Merging LoRA weights...")
    merged_model = lora_model.merge_and_unload()

    print("Saving last checkpoint of the model")
    merged_model.save_pretrained("models/" + model_output)
    tokenizer.save_pretrained("models/" + model_output)

    merged_model.push_to_hub(model_output)
    tokenizer.push_to_hub(model_output)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = climatesafeguards_template

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset, tokenizer)


if __name__ == "__main__":
    print("Is MPS Backend Available:", torch.backends.mps.is_available())
    print("Is CUDA Backend Available:", torch.cuda.is_available())
    args = get_args()
    assert args.model != "", "Please provide the model path"

    set_seed(args.seed)

    logging.set_verbosity_error()

    main(args)

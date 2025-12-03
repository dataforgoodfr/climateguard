import argparse
import logging
import os
import re
import uuid
from typing import Optional, Union

import numpy as np
import torch
import unsloth
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import classification_report
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

import wandb


def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    # log_filename = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler(log_filename),
            logging.StreamHandler(),  # Also log to console
        ],
    )

    return logging.getLogger(__name__)


logger = setup_logging()


load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def get_data():
    dataset = load_dataset("DataForGood/climateguard")

    dataset = dataset.filter(lambda example: example["comments"] == [])
    dataset = dataset.filter(
        lambda example: isinstance(example["misinformation"], bool)
    )
    dataset = dataset.filter(
        lambda example: not isinstance(example["misinformation"], str)
    )
    dataset = dataset.map(
        lambda example: {
            "text": example["plaintext"],
            "value": 10 * int(example["misinformation"]),
        }
    )
    dataset = dataset.select_columns(["id", "text", "value"])
    return dataset


def parse_response(response: Optional[Union[int, str]]):
    """Parse response containing only a score."""
    if isinstance(response, int):
        return response
    match = re.match(r"^[^\d]*(\d+)", response)
    if match:
        score = int(match.group(1))  # Extract score as an integer
    else:
        score = 0
    return score


def test_model(args, test_dataset, model, tokenizer, max_new_tokens, device="cuda"):
    logger.info("Evaluating model on test set...")
    model.eval()
    results = []
    for example in tqdm(test_dataset):
        input_conv = [example["messages"][0]]
        inputs = tokenizer.apply_chat_template(
            input_conv,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length - max_new_tokens,
            add_generation_prompt=True,
        ).to(device)
        with torch.no_grad():
            output_tokens = model.generate(inputs, max_new_tokens=max_new_tokens)
        prediction = tokenizer.decode(
            output_tokens[0][inputs.size(1) :], skip_special_tokens=True
        )
        results.append(parse_response(prediction) >= 8)

    # preds, refs = zip(*results)
    # with open("predictions.txt", "w") as f:
    #     for pred in preds:
    #         f.write(pred)
    #         f.write("\n")
    # rouge_scores = rouge.compute(predictions=preds, references=refs)
    report = classification_report(
        test_dataset.to_pandas()["value"].astype(int),
        results,
    )
    logger.info(f"Classification on test set: \n{report}")

def formatting_prompts_func(examples, tokenizer):
    convos = examples["messages"]
    chats = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "chat" : chats, }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit",
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.wandb:
        logger.info("reporting to wandb")
        wandb.login(key=os.getenv("WANDB_KEY"))

    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "climateguard_train"
    )
    prompt = """You are an assistant helping editors to moderate TV and radio content.
You will be provided with a transcript delimited by triple backticks.
Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.

Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?

Answer just a number on a scale of 0 - does not promote - to 10 - certainly promotes.

text: {transcript}"""

    device = "cuda" if torch.cuda.is_available() else None
    if not device:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    logging.info(f"Using device: {device}")
    logging.info(f"bfloat available:: {torch.cuda.is_bf16_supported()}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
        token=os.getenv("HF_TOKEN"),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    dataset = get_data()
    dataset = dataset.map(
        lambda example: {
            "messages": [
                {"role": "user", "content": prompt.format(transcript=example["text"])},
                {"role": "assistant", "content": str(example["value"])},
            ]
        }
    )

    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
    )
    train_dataset = dataset["train"].train_test_split(test_size=0.15)
    test_dataset = dataset["test"]

    logger.info(f"\n📝 Single Sample: {train_dataset['train'][0]['messages']}")

    training_args = SFTConfig(
        eval_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_steps=5,
        max_steps=4,#args.epochs
        #* int(np.ceil(len(train_dataset["train"]) / args.train_batch_size)),
        logging_strategy="steps",
        logging_steps=10,
        eval_steps=len(train_dataset["train"]) // 10,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        max_grad_norm=args.max_grad_norm,  
        output_dir=OUTPUT_DIR,
        report_to="wandb" if args.wandb else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        dataset_text_field="chat",
        max_seq_length=args.max_length,
        packing=False, 
        args=training_args,
    )

    if args.wandb:
        run = wandb.init(
            entity=os.getenv("WANDB_ENTITY", "gmguarino"),
            project="classification",
            config=training_args.to_dict(),
            name=f"gmguarino/{args.checkpoint.split('/')[1]}-climateguard"
            + str(uuid.uuid4()).split("-")[0],
        )
    trainer_stats = trainer.train()

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    test_model(
        args,
        test_dataset.select(range(3)),
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    model.save_pretrained("lora_model")  
    tokenizer.save_pretrained("lora_model")
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/model",
        tokenizer,
        save_method="merged_4bit_forced",
    )

    model.push_to_hub_merged(
        f"gmguarino/{args.checkpoint.split('/')[1]}-climateguard",
        tokenizer,
        save_method="merged_4bit_forced",
        token=os.getenv("HF_TOKEN")
    )

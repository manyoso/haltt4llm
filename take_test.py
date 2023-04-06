import argparse
import json
import time

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


def load_trivia_questions(file_path):
    with open(file_path, "r") as file:
        trivia_data = json.load(file)
    return trivia_data


def generate_question_string(question_data):
    question = question_data["question"]
    choices = [
        f"    {answer['choice']}. {answer['text']}\n"
        if answer != question_data["answers"][-1]
        else f"    {answer['choice']}. {answer['text']}"
        for answer in question_data["answers"]
    ]
    return f"{question}\n{''.join(choices)}"


def grade_answers(question_data, llm_answer):
    correct_answer = None
    for answer in question_data["answers"]:
        if answer["correct"]:
            correct_answer = answer
            break

    if correct_answer is None:
        return "No correct answer found"

    normalized_llm_answer = llm_answer.lower().strip()
    normalized_correct_answer = correct_answer["text"].lower().strip()

    # lower case of the full text answer is in the llm's answer
    if normalized_correct_answer in normalized_llm_answer:
        return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " A." or  " B." or " C." or " D." or " E." for instance
    if f" {correct_answer['choice']}." in llm_answer:
        return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " (A)" or  " (B)" or " (C)" or " (D)" or " (E)" for instance
    if f"({correct_answer['choice']})" in llm_answer:
        return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    if (
        "i don't know" in normalized_llm_answer
        or normalized_llm_answer == "d"
        or normalized_llm_answer == "d."
    ):
        return f"{llm_answer} (uncertain)"

    return f"{llm_answer} (incorrect {correct_answer['choice']}.)"


if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s) available.")
    device_index = 0
    if device_count > 1:
        device_index = input(f"Select device index (0-{device_count-1}): ")
        device_index = int(device_index)
    device = f"cuda:{device_index}"
    print(f"Using device: {device}")
else:
    device = "cpu"
    print("No GPU available, quitting.")
    exit()


def query_model(
    prompt,
    model,
    tokenizer,
    temperature=0.1,
    max_new_tokens=50,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    # Default
    """
    generation_config = GenerationConfig(
        temperature=temperature,
        **kwargs,
    )
    """
    # beam3
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=3
    )    
    # Multinomial
    """
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=1,
        do_sample=True
    )
    """
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            # output_scores=True,
            max_new_tokens=max_new_tokens,
        )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    response = output.split("### Response:")[1].strip()
    return response.split("### Instruction:")[0].strip()


def main():
    # python3 take_test.py --test_name='alpaca7b_lora_r16' --base_model='decapoda-research/llama-7b-hf' --lora_weights='tloen/alpaca-lora-7b' --trivia=nota_trivia_questions.json
    # python3 take_test.py --test_name='alpaca7b_lora_r16' --base_model='decapoda-research/llama-7b-hf' --lora_weights='tloen/alpaca-lora-7b' --trivia=hq_trivia_questions.json
    # fake_trivia_questions.json
    parser = argparse.ArgumentParser(
        description="Run trivia quiz with local Alpaca model."
    )
    parser.add_argument("--base_model", type=str, help="Base llama model")
    parser.add_argument("--lora_weights", type=str, help="lora weights")
    parser.add_argument("--trivia", type=str, help="File path to trivia questions")
    parser.add_argument("--test_name", type=str, help="name for the model/lora couple")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(
        args.base_model, padding_side="left", device_map={"": 0}
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Works for cuda only.
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        model, args.lora_weights, device_map={"": 0}, torch_dtype=torch.float16
    )
    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    file_path = args.trivia
    trivia_data = load_trivia_questions(file_path)

    total_score = 0
    incorrect = []
    unknown = []
    model_name = ""

    for i, question_data in enumerate(trivia_data):
        question_string = generate_question_string(question_data)
        prompt = generate_prompt(question_string)

        print(f"Question {i+1}: {question_string}")
        llm_answer = query_model(prompt, model, tokenizer)

        answer_output = grade_answers(question_data, llm_answer)
        print(f"Answer: {answer_output}\n")

        if "(correct)" in answer_output:
            total_score += 2
        elif "(incorrect" in answer_output:
            incorrect.append((i + 1, question_string, answer_output))
        else:
            total_score += 1
            unknown.append((i + 1, question_string, answer_output))

    with open(f"test_results_{file_path}_{args.test_name}.txt", "w") as f:
        f.write(f"Total score: {total_score} of {len(trivia_data) * 2}\n")
        i = len(incorrect)
        if i:
            f.write(f"\nIncorrect: {i}\n")
            for question_num, question_string, answer_output in incorrect:
                f.write(
                    f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n"
                )
        u = len(unknown)
        if u:
            f.write(f"Unknown: {u}\n")
            for question_num, question_string, answer_output in unknown:
                f.write(
                    f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n"
                )

    print(f"Total score: {total_score} of {len(trivia_data) * 2}\n", end="")


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep token limit low.

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    main()

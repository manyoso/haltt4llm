import argparse
import openai
import torch
import json
import time
from transformers import GenerationConfig
from autograd_4bit import load_llama_model_4bit_low_ram

def load_trivia_questions(file_path):
    with open(file_path, 'r') as file:
        trivia_data = json.load(file)
    return trivia_data

def generate_question_string(question_data):
    question = question_data['question']
    choices = [f"    {answer['choice']}. {answer['text']}\n" if answer != question_data['answers'][-1] else f"    {answer['choice']}. {answer['text']}" for answer in question_data['answers']]
    return f"{question}\n{''.join(choices)}"

def grade_answers(question_data, llm_answer):
    correct_answer = None
    for answer in question_data['answers']:
        if answer['correct']:
            correct_answer = answer
            break

    if correct_answer is None:
        return "No correct answer found"

    normalized_llm_answer = llm_answer.lower().strip()
    normalized_correct_answer = correct_answer['text'].lower().strip()

    # lower case of the full text answer is in the llm's answer
    if normalized_correct_answer in normalized_llm_answer:
        return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " A." or  " B." or " C." or " D." or " E." for instance
    if f" {correct_answer['choice']}." in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " (A)" or  " (B)" or " (C)" or " (D)" or " (E)" for instance
    if f"({correct_answer['choice']})" in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    if "i don't know" in normalized_llm_answer or normalized_llm_answer == "d" or normalized_llm_answer == "d.":
        return f"{llm_answer} (uncertain)"

    return f"{llm_answer} (incorrect {correct_answer['choice']}.)"

def query_openai_gpt3_5(prompt, engine):
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=50,
                temperature=0.1,
            )
            return response.choices[0].text.strip()
        except openai.error.RateLimitError as e:
            print("Rate limit exceeded. Pausing for one minute...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"Error: {e}")
            break

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
    print("No GPU available, using CPU.")

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
        generation_config = GenerationConfig(
            temperature=temperature,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        response = output.split("### Response:")[1].strip()
        return response.split("### Instruction:")[0].strip()

def main():
    parser = argparse.ArgumentParser(description='Run trivia quiz with GPT-3 or a local model.')
    parser.add_argument('--use-gpt3', action='store_true', help='Use GPT-3')
    parser.add_argument('--use-gpt3-5', action='store_true', help='Use GPT-3.5')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key')
    parser.add_argument('--trivia', type=str, help='File path to trivia questions')
    args = parser.parse_args()

    use_gpt_3 = args.use_gpt3 or args.use_gpt3_5

    if use_gpt_3 and not args.openai_key:
        print("Please provide an OpenAI API key with the --openai-key argument.")
        return

    if use_gpt_3:
        openai.api_key = args.openai_key

    if not use_gpt_3:
        config_path = './models/llama-7b-hf/'
        model_path = './weights/llama-7b-4bit.pt'
        model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)
        print('Fitting 4bit scales and zeros to half')
        for n, m in model.named_modules():
            if '4bit' in str(type(m)):
                m.zeros = m.zeros.half()
                m.scales = m.scales.half()

    file_path = args.trivia
    trivia_data = load_trivia_questions(file_path)

    total_score = 0
    incorrect = []
    unknown = []
    model_name = ("text-davinci-003" if args.use_gpt3_5 else "text-davinci-002") if use_gpt_3 else "alpaca-lora-4bit"

    for i, question_data in enumerate(trivia_data):
        question_string = generate_question_string(question_data)
        prompt = generate_prompt(question_string)

        print(f"Question {i+1}: {question_string}")
        if use_gpt_3:
            llm_answer = query_openai_gpt3_5(prompt, model_name)
        else:
            llm_answer = query_model(prompt, model, tokenizer)

        answer_output = grade_answers(question_data, llm_answer)
        print(f"Answer: {answer_output}\n")

        if "(correct)" in answer_output:
            total_score += 2
        elif "(incorrect" in answer_output:
            incorrect.append((i+1, question_string, answer_output))
        else:
            total_score += 1
            unknown.append((i+1, question_string, answer_output))

    with open(f"test_results_{file_path}_{model_name}.txt", 'w') as f:
        f.write(f"Total score: {total_score} of {len(trivia_data) * 2}\n")
        i = len(incorrect)
        if i:
            f.write(f"\nIncorrect: {i}\n")
            for question_num, question_string, answer_output in incorrect:
                f.write(f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n")
        u = len(unknown)
        if u:
            f.write(f"Unknown: {u}\n")
            for question_num, question_string, answer_output in unknown:
                f.write(f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n")

    print(f"Total score: {total_score} of {len(trivia_data) * 2}\n", end='')

def generate_prompt(instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep token limit low.

### Instruction:
{instruction}

### Response:
"""

if __name__ == '__main__':
    main()

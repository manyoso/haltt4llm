import argparse
import openai
import json
import os
import random
import torch
from transformers import GenerationConfig
from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from tqdm import tqdm
from peft import PeftModel
from peft.tuners.lora import Linear4bitLt

def query_openai_gpt(prompt, engine):
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.8,
                timeout=1
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
        max_new_tokens=256,
        **kwargs,
    ):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.75,
            top_k=40,
            num_beams=1,
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

def generate_trivia_questions(prompt, model, num_questions):
    questions = set()

    use_gpt_3 = model == "text-davinci-003" or model == "text-davinci-002"

    if not use_gpt_3:
        config_path = './models/llama-7b-hf/'
        model_path = './weights/llama-7b-4bit.pt'
        lora_path = './loras/gpt4all-lora/'
        model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)
        model = PeftModel.from_pretrained(model, lora_path)
        print('Fitting 4bit scales and zeros to half')
        for n, m in model.named_modules():
            if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
                m.zeros = m.zeros.half()
                m.scales = m.scales.half()
                m.bias = m.bias.half()

    with tqdm(total=num_questions, desc="Generating trivia questions") as pbar:
        while len(questions) < num_questions:
            try:
                # Select three random categories from the list
                a, b, c = random.sample(categories, 3)

                new_categories = f"\n\nGenerate three such questions in format above in the categories of {a}, {b} and {c}."
                new_prompt = generate_prompt(prompt + categories)
                print(f"Prompt categories: {a}, {b} and {c}")

                if not use_gpt_3:
                    output = query_model(new_prompt, model, tokenizer)
                else:
                    output = query_openai_gpt(new_prompt, model)

                print(f"Raw output: {output}")
                generated_questions = json.loads(output)
                for question in generated_questions:
                    question_text = question["question"]
                    if question_text not in questions:
                        questions.add(question_text)
                        pbar.update(1)
                        append_question_to_json(question, "new_trivia_questions.json")
            except Exception as e:
                print(f"Error occurred: {e}")
                continue
    return questions

def generate_prompt(instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep token limit low.

### Instruction:
{instruction}

### Response:\n
"""

def append_question_to_json(question, filename):
    data = []
    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = json.load(file)
    data.append(question)
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Generate trivia questions with GPT-3')
    parser.add_argument('--use-gpt3', action='store_true', help='Use GPT-3')
    parser.add_argument('--use-gpt3-5', action='store_true', help='Use GPT-3.5')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key')
    parser.add_argument('--prompt-file', type=str, required=True, help='Path to prompt file')
    parser.add_argument('--num-questions', type=int, required=True, help='Number of questions to generate')
    args = parser.parse_args()

    use_gpt_3 = args.use_gpt3 or args.use_gpt3_5

    if args.openai_key and not use_gpt_3:
        print("Please provide an OpenAI model with the --openai-key argument.")
        return

    if use_gpt_3 and not args.openai_key:
        print("Please provide an OpenAI API key with the --openai-key argument.")
        return

    if use_gpt_3:
        openai.api_key = args.openai_key

    with open(args.prompt_file, "r") as file:
        prompt = file.read()

    model_name = ("text-davinci-003" if args.use_gpt3_5 else "text-davinci-002") if use_gpt_3 else "alpaca-lora-4bit"

    generate_trivia_questions(prompt, model_name, args.num_questions)

categories = [
    "Geography",
    "History",
    "Literature",
    "Science",
    "Art",
    "Music",
    "Film",
    "Sports",
    "Politics",
    "Religion",
    "Food and Drink",
    "Technology",
    "Business",
    "Fashion",
    "Architecture",
    "Mythology",
    "Animals",
    "Celebrities",
    "Language",
    "Law",
    "Mathematics",
    "Philosophy",
    "Psychology",
    "Medicine",
    "Anatomy",
    "Astronomy",
    "Chemistry",
    "Physics",
    "Botany",
    "Zoology",
    "Genetics",
    "Anthropology",
    "Archeology",
    "Paleontology",
    "Sociology",
    "Economics",
    "Journalism",
    "Education",
    "Transportation",
    "Military",
    "Environment",
    "Immigration",
    "Urban Development",
    "Rural Development",
    "Demographics",
    "Astronomy",
    "Astrology",
    "Chemistry",
    "Geology",
    "Linguistics",
    "Cosmology",
    "Cryptography",
    "Cultural Studies",
    "Dance",
    "Drama",
    "Ecology",
    "Energy",
    "Entrepreneurship",
    "Ethnic Studies",
    "Folklore",
    "Gender Studies",
    "Health",
    "Horticulture",
    "Human Rights",
    "Industrial Design",
    "International Relations",
    "Labor Studies",
    "Land Use",
    "Landscape Architecture",
    "Linguistics",
    "Logic",
    "Manufacturing",
    "Marine Science",
    "Marketing",
    "Material Science",
    "Media Studies",
    "Meteorology",
    "Military History",
    "Natural Resources",
    "Neuroscience",
    "Nutrition",
    "Oceanography",
    "Oncology",
    "Operations Research",
    "Optometry",
    "Ornithology",
    "Pharmacology",
    "Philosophy of Science",
    "Photography",
    "Physical Education",
    "Planetary Science",
    "Plant Science",
    "Political Economy",
    "Political Science",
    "Public Health",
    "Public Policy",
    "Robotics",
    "Social Psychology",
    "Social Work",
    "Urban Planning"
]

if __name__ == "__main__":
    main()

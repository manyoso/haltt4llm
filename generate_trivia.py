import argparse
import openai
import json
import os
import random
from tqdm import tqdm

def generate_trivia_questions(prompt, model, num_questions=15):
    questions = set()
    with tqdm(total=num_questions, desc="Generating trivia questions") as pbar:
        while len(questions) < num_questions:
            try:
                # Select three random categories from the list
                a, b, c = random.sample(categories, 3)

                new_categories = f"\n\nGenerate three such questions in json format above in the categories of {a}, {b} and {c}."
                new_prompt = prompt + new_categories
                print(f"Prompt categories: {a}, {b} and {c}")

                response = openai.Completion.create(
                    engine=model,
                    prompt=new_prompt,
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    temperature=0.8,
                    timeout=1
                )
                output = response.choices[0].text.strip()

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
    parser.add_argument('--model', type=str, default="text-davinci-003", help='OpenAI model to use. Default is text-davinci-003')
    parser.add_argument('--openai-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--prompt-file', type=str, required=True, help='Path to prompt file')
    parser.add_argument('--num-questions', type=int, required=True, help='Number of questions to generate')
    args = parser.parse_args()

    openai.api_key = args.openai_key

    with open(args.prompt_file, "r") as file:
        prompt = file.read()

    generate_trivia_questions(prompt, args.model, args.num_questions)

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

import argparse
import json

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='the JSON file containing the questions to filter')
parser.add_argument('output_file', help='the JSON file to save the filtered questions to')
args = parser.parse_args()

# Load the input JSON file
with open(args.input_file, 'r') as f:
    questions = json.load(f)

# Iterate over each question
new_questions = []
for question in questions:
    # Display the question and answers
    print(f"\n{question['question']}")
    if 'direct' in question:
        print(f"Actual answer: {question['direct']}")
    for answer in question['answers']:
        print(f"{answer['choice']}. {answer['text']}")

    # Ask the user whether to keep the question
    choice = input("Do you want to keep this question? (y/n) ").lower()
    while choice not in ('y', 'n', ''):
        print("Invalid choice, please enter 'y' or 'n'.")
        choice = input("Do you want to keep this question? (y/n) ").lower()

    # Add the question to the list of new questions if the user wants to keep it
    if choice == '' or choice == 'y':
        new_questions.append(question)

# Save the new list of questions to the output file
with open(args.output_file, 'w') as f:
    json.dump(new_questions, f, indent=4)

print(f"\nSaved {len(new_questions)} questions to {args.output_file}.")

import json
import random
import openai
from tqdm import tqdm
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import os

random.seed(42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--question', type=str,
    default='original',
    choices=['original', 'rephrased'],
    help="Specify 'original' to process original questions or 'rephrased' to process rephrased questions."
)
parser.add_argument('--new_rephrase',
    action='store_true',
    help='Flag to refine the questions again.'
)
parser.add_argument('--task', type=str,
    choices=[
        'birthdate_day', 'birthdate_month', 'birthdate_year',
        'birthdate_earlier', 'coin_val', 'last_letter_concatenation',
        'last_letter_concatenation4', 'sports', 'date', 'csqa', 'stereo'
    ],
    help='Specify the task file name for processing.'
)
parser.add_argument('--model', type=str,
    default='gpt-4',
    help='Specify the model name of the OpenAI API to use.'
)
parser.add_argument('--onestep', 
    action='store_true',
    help='Flag to use onestep RaR.'
)
args = parser.parse_args()

with open('config.json', 'r') as config_file:
    spec_config = json.load(config_file)

SPEC = ""
if args.task in spec_config:
    SPEC = spec_config[args.task]

openai.api_key = "Your API Key" # put your API key here
model_id = args.model

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def chatgpt_conversation(conversation_log, model_id):
    response = completion_with_backoff(
        model=model_id,
        messages=conversation_log
    )
    response = response.choices[0].message.content.strip()
    return response


def get_result(filename): 
    with open(f'data/{filename}_{model_id}.json', 'r') as f:
        data = json.load(f)
    print(f'data/{filename}_{model_id}.json')

    right, wrong = 0, 0 # gross calculation of the accuracy, will be human-inspected later
    
    for idx, q in tqdm(enumerate(data), total=len(data)):
        answer = q['answer']
        if args.question == 'rephrased':
            assert 'refined_question' in q.keys() and q['refined_question'] != ''

            if 'gpt-4' in args.model:
                messages = [
                    {"role": "user",
                    "content": "(original) {original}\n(rephrased) {rephrased}\n{spec}Use your answer for the rephrased question to answer the original question.".format(
                        original=q['question'],
                        rephrased=q['refined_question'],
                        spec=SPEC
                        )
                    }
                ]
            else:
                messages = [
                    {"role": "user",
                    "content": "(original) {original}\n(revised) {rephrased}\n{spec}Use your answer in the revised question to answer the original question.".format(
                        original=q['question'],
                        rephrased=q['refined_question'],
                        spec=SPEC
                        )
                    }
                ]
        else:
            messages = [
                {"role": "user",
                "content": "{question}\n{spec}".format(
                    question=q['question'],
                    spec=SPEC)
                }
            ]
        response = chatgpt_conversation(messages, model_id)
        
        log_directory = f'log_{model_id}'
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        
        by_word = ['coin_val', 'last_letter_concatenation', 'last_letter_concatenation4', 'birthdate_day', 'birthdate_month', 'birthdate_year', 'birthdate_earlier', 'sports']
        punctuation_marks = {'.', ','}
        normalized_answer = answer.lower()
        quoted_answer1 = '"'+answer.lower()+'"'
        quoted_answer2 = '\''+answer.lower()+'\''
        if normalized_answer in response.lower():
            if args.task in by_word:
                splitted = response.lower().split(' ')
                splitted = [x.strip() for x in splitted]
                if any(normalized_answer + mark in splitted for mark in punctuation_marks) \
                    or normalized_answer in splitted\
                    or any(quoted_answer1 + mark in splitted for mark in punctuation_marks) \
                    or quoted_answer1 in splitted\
                    or any(quoted_answer2 + mark in splitted for mark in punctuation_marks) \
                    or quoted_answer2 in splitted:
                    right += 1
                else:
                    wrong += 1
                    with open(f'log_{model_id}/{filename}_{args.question}_wrong.json', 'a') as f:
                        record = {"question":q["question"], "answer":q["answer"], "response":response}
                        json.dump(record, f)
                        f.write('\n')
            else:
                right += 1
        else:
            wrong += 1
            with open(f'log_{model_id}/{filename}_{args.question}_wrong.json', 'a') as f:
                record = {"question":q["question"], "answer":q["answer"], "response":response}
                json.dump(record, f)
                f.write('\n')

        # document the responses
        with open(f'log_{model_id}/{filename}_{args.question}_response.json', 'a') as f:
            record = {"question":q["question"], "answer":q["answer"], "response":response}
            json.dump(record, f)
            f.write('\n')

        time.sleep(1)
    print("Accuracy: ", right / (right + wrong))


def get_result_multi(filename): 
    with open(f'data/{filename}_{model_id}.json', 'r') as f:
        data = json.load(f)

    right, wrong = 0, 0
    for idx, q in tqdm(enumerate(data), total=len(data)):
        answer = q['answer']
        
        if args.question == 'rephrased':
            assert 'refined_question' in q.keys() and q['refined_question'] != ''
            messages = [
                {"role": "user",
                 "content": (f"(original) {q['question']}\n" + 
                             f"(rephrased) {q['refined_question']}\n" + 
                             "Choices: "+ ' '.join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])) + "\n"
                             "Use your answer for the rephrased question to answer the original question.\n"+ SPEC)
                }
            ]
        else:
            messages = [
                {"role": "user",
                 "content": f"{q['question']}\n" +
                    f"Choices: " + ' '.join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])) + "\n" + SPEC
                }
            ]
        response = chatgpt_conversation(messages, model_id)

        incorrect_choices = [c for c in q['choices'] if answer.lower() not in c.lower()]
        if answer.lower() in response.lower() and all([c.lower() not in response.lower() for c in incorrect_choices]):
            right += 1
        else:
            wrong += 1
            # document the wrong examples
            with open(f'log_{model_id}/{filename}_{args.question}_wrong.json', 'a') as f:
                record = {"question":q["question"], "answer":q["answer"], "response":response}
                json.dump(record, f)
                f.write('\n')

        # document the responses
        with open(f'log_{model_id}/{filename}_{args.question}_response.json', 'a') as f:
            record = {"question":q["question"], "answer":q["answer"], "response":response}
            json.dump(record, f)
            f.write('\n')

        time.sleep(1)

    print("Accuracy: ", right / (right + wrong))


def get_result_stereo(filename): 
    with open(f'data/{filename}_{model_id}.json', 'r') as f:
        data = json.load(f)
        
    data = data
    stereo_num, anti_stereo_num, unrelated_num, undetermined = 0, 0, 0, 0
    
    for idx, q in tqdm(enumerate(data), total=len(data)):
        stereo = q['stereo']
        anti_stereo = q['anti_stereo']
        unrelated = q['unrelated']
        
        if args.question == 'rephrased':
            assert 'refined_question' in q.keys() and q['refined_question'] != ''
            messages = [
                {"role": "user", 
                 "content": "(original)" + q['question'] + "\n"
                    + "(revised)" +  q['refined_question'] + "\n"
                    + f"Choices: A. {q['choices'][0]} B. {q['choices'][1]} C. {q['choices'][2]}\n"
                    + "Use your answer in the revised question to answer the original question.\n"
                    + SPEC
                }
            ]
        elif args.question == 'zero-CoT':
            messages = [
                    {"role": "user", 
                     "content": q['question'] 
                        + f"Choices: A. {q['choices'][0]} B. {q['choices'][1]} C. {q['choices'][2]}\n"
                        + "Let's think step by step."
                    }
            ]
        else:
            messages = [
                {"role": "user", 
                 "content": q['question'] + "\n"
                    + f"Choices: A. {q['choices'][0]} B. {q['choices'][1]} C. {q['choices'][2]}\n"
                    + SPEC
                }
            ]
        response = chatgpt_conversation(messages, model_id)

        if stereo.lower() in response.lower() and anti_stereo.lower() not in response.lower() and unrelated.lower() not in response.lower():
            stereo_num += 1
            with open(f'log_{model_id}/{filename}_{args.question}_stereo.json', 'a') as f:
                record = {"question":q["question"], "answer":q["anti_stereo"], "response":response}
                json.dump(record, f)
                f.write('\n')
        elif anti_stereo.lower() in response.lower() and stereo.lower() not in response.lower() and unrelated.lower() not in response.lower():
            anti_stereo_num += 1
            with open(f'log_{model_id}/{filename}_{args.question}_anti_stereo.json', 'a') as f:
                record = {"question":q["question"], "answer":q["anti_stereo"], "response":response}
                json.dump(record, f)
                f.write('\n')
        elif unrelated.lower() in response.lower() and stereo.lower() not in response.lower() and anti_stereo.lower() not in response.lower():
            unrelated_num += 1
        else:
            undetermined += 1
            with open(f'log_{model_id}/{filename}_{args.question}_undertermined.json', 'a') as f:
                record = {"question":q["question"], "answer":q["anti_stereo"], "response":response}
                json.dump(record, f)
                f.write('\n')

        # document the responses
        with open(f'log_{model_id}/{filename}_{args.question}_response.json', 'a') as f:
            record = {"question":q["question"], "answer":q["anti_stereo"], "response":response}
            json.dump(record, f)
            f.write('\n')

        time.sleep(1)

    print("stereo: ", stereo_num)
    print("anti_stereo: ", anti_stereo_num)
    print("unrelated: ", unrelated_num)
    print("undetermined: ", undetermined)


def refine_question(filename):
    with open(f'data/{filename}_{model_id}.json', 'r') as f:
        data = json.load(f)

    if 'refined_question' in data[0].keys() and data[0]['refined_question'] != '':
        print("Overwriting the refined questions.")
    
    for idx, q in tqdm(enumerate(data), total=len(data)):
        messages = [
            {"role": "user", 
            "content": f'"{q["question"]}"\nGiven the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.'
            }
        ]
        response = chatgpt_conversation(messages, model_id)
        if response[0] == '"' and response[-1] == '"':
            response = response[1:-1]
        
        q['refined_question'] = response
        time.sleep(1)

    with open(f'data/{filename}_{model_id}.json', 'w') as f:
        json.dump(data, f)


def get_result_onestep(filename):
    right, wrong = 0, 0
    with open(f'data/{filename}_{model_id}.json', 'r') as f:
        data = json.load(f)
    log_directory = f'log_{model_id}'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    for idx, q in tqdm(enumerate(data), total=len(data)):
        if 'csqa' in args.task:
            if "gpt-3.5" in model_id:
                messages = [
                    {"role": "user", 
                     "content": f'"{q["question"]}"\nReword and elaborate on the inquiry, then provide an answer. '
                        "Choices: "+ ' '.join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])) + "\n"
                        + SPEC
                    }
                ]    
            else: 
                messages = [
                    {"role": "user", 
                     "content": f'"{q["question"]}"\nRephrase and expand the question, and respond. '
                        "Choices: "+ ' '.join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])) + "\n"
                        + SPEC
                    }
                ]
        else:
            if "gpt-3.5" in model_id:
                messages = [
                    {"role": "user", 
                    "content": f'"{q["question"]}"\nReword and elaborate on the inquiry, then provide an answer. ' + SPEC
                    }
                ]
            else:
                messages = [
                    {"role": "user", 
                    "content": f'"{q["question"]}"\nRephrase and expand the question, and respond. ' + SPEC
                    }
                ]
        response = chatgpt_conversation(messages, model_id)
        answer = q['answer']
        
        by_word = ['coin_val', 'last_letter_concatenation', 'last_letter_concatenation4', 'birthdate_day', 'birthdate_month', 'birthdate_year', 'birthdate_earlier', 'sports']
        punctuation_marks = {'.', ','}
        normalized_answer = answer.lower()
        quoted_answer1 = '"'+answer.lower()+'"'
        quoted_answer2 = '\''+answer.lower()+'\''
        half_quoted_answer1 = '"'+answer.lower()
        half_quoted_answer2 = '\''+answer.lower()
        if normalized_answer in response.lower():
            if args.task in by_word:
                splitted = response.lower().split(' ')
                splitted = [x.strip() for x in splitted]
                if any(normalized_answer + mark in splitted for mark in punctuation_marks) \
                    or normalized_answer in splitted\
                    or any(quoted_answer1 + mark in splitted for mark in punctuation_marks) \
                    or quoted_answer1 in splitted\
                    or any(quoted_answer2 + mark in splitted for mark in punctuation_marks) \
                    or quoted_answer2 in splitted\
                    or any(half_quoted_answer1 + mark in splitted for mark in punctuation_marks) \
                    or half_quoted_answer1 in splitted\
                    or any(half_quoted_answer2 + mark in splitted for mark in punctuation_marks) \
                    or half_quoted_answer2 in splitted:
                    right += 1
                else:
                    wrong += 1
                    with open(f'log_{model_id}/{filename}_{args.question}_wrong.json', 'a') as f:
                        record = {"question":q["question"], "answer":q["answer"], "response":response}
                        json.dump(record, f)
                        f.write('\n')
            else:
                right += 1
        else:
            wrong += 1
            with open(f'log_{model_id}/{filename}_{args.question}_wrong.json', 'a') as f:
                record = {"question":q["question"], "answer":q["answer"], "response":response}
                json.dump(record, f)
                f.write('\n')

        # document the responses
        with open(f'log_{model_id}/{filename}_{args.question}_combine_response.json', 'a') as f:
            record = {"question":q["question"], "answer":q["answer"], "response":response}
            json.dump(record, f)
            f.write('\n')

        time.sleep(1)

    print("Accuracy: ", right / (right + wrong))

def main():

    if args.onestep:
        args.question = 'rephrased'
        get_result_onestep(args.task)
    
    else:
        if args.new_rephrase:
            refine_question(args.task)

        if 'csqa' in args.task:
            get_result_multi(args.task)
        elif args.task == 'stereo':
            get_result_stereo(args.task)
        else:
            get_result(args.task)

if __name__ == "__main__":
    main()

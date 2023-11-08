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
parser.add_argument('--question', type=str, default='original', help='refined, original, or zero-CoT')
parser.add_argument('--new_refine', action='store_true', default=False, help='refine the questions agaian')
parser.add_argument('--task', type=str, default='date_subset', help='task file name.')
parser.add_argument('--model', type=str, default='gpt-4', help='model name.')
args = parser.parse_args()

with open('config.json', 'r') as config_file:
    spec_config = json.load(config_file)

SPEC = ""
if args.task in spec_config:
    SPEC = spec_config[args.task]

openai.api_key = "Your API key" # put your API key here
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
        if args.question == 'refined':
            assert 'refined_question' in q.keys() and q['refined_question'] != ''
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
                "content": "{question}\n{spec}".format(
                    question=q['question'],
                    spec=SPEC)
                }
            ]
        response = chatgpt_conversation(messages, model_id)
        
        log_directory = 'log_{model_id}/{filename}_{args.question}_response.json'
        log_directory_false = 'log_{model_id}/{filename}_{args.question}_wrong.json'
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        if not os.path.exists(log_directory_false):
            os.makedirs(log_directory_false)
        
        by_word = ['coin_val', 'last_letter_concatenation', 'last_letter_concatenation4', 'birthdate_day', 'birthdate_month', 'birthdate_year', 'birthdate_earlier', 'sports']
        if answer.lower() in response.lower():
            if args.task in by_word:
                splitted = response.lower().split(' ')
                splitted = [x.strip() for x in splitted]
                if answer.lower() in splitted or answer.lower()+'.' in splitted or answer.lower()+',' in splitted or '"'+answer.lower()+'"' in splitted or '\''+answer.lower()+'\'' in splitted or '"'+answer.lower()+'".' in splitted or '\''+answer.lower()+'\'.' in splitted:
                    right += 1
                else:
                    wrong += 1
                    with open(log_directory_false, 'a') as f:
                        record = {"question":q["question"], "answer":q["answer"], "response":response}
                        json.dump(record, f)
                        f.write('\n')
            else:
                right += 1
        else:
            wrong += 1
            with open(log_directory_false, 'a') as f:
                record = {"question":q["question"], "answer":q["answer"], "response":response}
                json.dump(record, f)
                f.write('\n')

        # document the responses
        with open(log_directory, 'a') as f:
            record = {"question":q["question"], "answer":q["answer"], "response":response}
            json.dump(record, f)
            f.write('\n')

        time.sleep(1)

    print("Accuracy: ", right / (right + wrong))


def main():    
    get_result(args.task)

if __name__ == "__main__":
    main()
import json
import os
from openai import OpenAI
from utils.api_utils import GPT
from utils.common_utils import save_file, save_json_file, load_json_file
import time
import argparse


def _load_api_config(model: str, path: str):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    if model == 'deepseek-reasoner':
        entry = cfg.get('deepseek-reasoner')
    else:
        entry = cfg.get('default')

    return entry['api_key'], entry['base_url']


def _read_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def create_code_prompt(question):
    prompt = f"#案例:\n{CODE_SAMPLES}\n```question_begin:\n{question}\n```question_end\n```code_begin\n"
    return prompt


def ask_llm(prompt, instruct, model_name):
    messages = [
        {"role": "system", "content": instruct},
        {"role": "user", "content": prompt}
    ]
    model_name = model_name
    response, success = gpt.request_gpt(20, messages, client, model_name)
    res = response.choices[0].message.content   
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, required=True, help='请选择要使用的大模型标识')
    args = parser.parse_args()
    model = args.model

    # 路径配置
    RUN_TIME = time.strftime("%Y%m%d-%H%M%S")
    CODE_DIR = os.path.join(os.path.dirname(__file__), 'output', f'{model}_{RUN_TIME}', 'code')
    PREDICT_PATH = os.path.join(os.path.dirname(__file__), 'output', f'{model}_{RUN_TIME}', 'predict.json')
    INSTRUCT_PATH = os.path.join(os.path.dirname(__file__), 'prompts/code_instruct.txt')
    SAMPLES_PATH = os.path.join(os.path.dirname(__file__), 'prompts/code_samples.txt')
    QUESTION_PATH = os.path.join(os.path.dirname(__file__), 'question.json')
    API_JSON_PATH = os.path.join(os.path.dirname(__file__), 'API.json')

    api_key_use, base_url_use = _load_api_config(model, API_JSON_PATH)
    
    client = OpenAI(api_key=api_key_use, base_url=base_url_use)

    gpt = GPT()

    CODE_INSTRUCT = _read_prompt(INSTRUCT_PATH)
    CODE_SAMPLES = _read_prompt(SAMPLES_PATH)

    save_data = []
    if not os.path.exists(CODE_DIR):
        os.makedirs(CODE_DIR)
    eval_set = load_json_file(QUESTION_PATH)
    try:
        for sample in eval_set:
            id = sample['question_id']
            question = sample['question']
            prompt = create_code_prompt(question)
            print("正在输出第{}个问题的代码".format(id))
            result = ask_llm(prompt, CODE_INSTRUCT, model)
            code = result.split('```code_begin')[1].split('```code_end')[0]
            code_path = f"{CODE_DIR}/id_{sample['question_id']}.py"
            save_file(code_path, code)
            sample = {
                "question_id": sample['question_id'],
                "question": question,
                "code_path": code_path
            }
            save_data.append(sample)
    except Exception as e:
        print(f"出错: {e}")
    finally:
        save_json_file(PREDICT_PATH, save_data)
    
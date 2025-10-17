import json
import os
from openai import OpenAI
from utils.api_utils import request_gpt
from utils.common_utils import save_file, save_json_file, load_json_file, sort_json_by_question_id
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re


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


def _create_code_prompt(question, code_sample):
    prompt = f"#案例:\n{code_sample}\n```question_begin:\n{question}\n```question_end\n```code_begin\n"
    return prompt


def _ask_llm(prompt, instruct, model_name, client):
    messages = [
        {"role": "system", "content": instruct},
        {"role": "user", "content": prompt}
    ]
    response, success = request_gpt(20, messages, client, model_name)
    if success:
        return response.choices[0].message.content
    else:
        raise Exception("API调用失败")


def process_single_question(question_data, code_dir, instruct_path, sample_path, model_name, client):
    question_id = question_data['question_id']
    question = question_data['question']
    print(f"正在输出第{question_id}个问题的代码...")
    instruct = _read_prompt(instruct_path)
    sample = _read_prompt(sample_path)
    prompt = _create_code_prompt(question, sample)
    for attempt in range(3):
        result = _ask_llm(prompt, instruct, model_name, client)
        if '```code_begin' in result and '```code_end' in result:
            code = result.split('```code_begin')[1].split('```code_end')[0]
            code = code.encode('utf-8').decode('utf-8-sig')  # 去除BOM
            code = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u4e00-\u9fa5]', '', code)  # 去除不可见字符
            code_path = f"{code_dir}/id_{question_id}.py"
            save_file(code_path, code)
            break
        else:
            print(f"问题 {question_id} 第{attempt+1}次请求失败: LLM返回内容格式异常")

    return {
        "question_id": question_id,
        "question": question,
        "code_path": f'./code/id_{question_id}.py',
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='deepseek-reasoner', help='请选择要使用的大模型标识')
    parser.add_argument('--threads', type=int, default=32, help='线程数，默认32个')
    args = parser.parse_args()
    model = args.model
    num_threads = args.threads

    # 路径配置
    RUN_TIME = time.strftime("%Y%m%d-%H%M%S")
    CODE_DIR = os.path.join(os.path.dirname(__file__), 'output', f'{model}_{RUN_TIME}', 'code')
    PREDICT_PATH = os.path.join(os.path.dirname(__file__), 'output', f'{model}_{RUN_TIME}', 'predict.json')
    INSTRUCT_PATH = os.path.join(os.path.dirname(__file__), 'prompts/code_instruct.txt')
    SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'prompts/code_sample.py')
    QUESTION_PATH = os.path.join(os.path.dirname(__file__), 'question.json')
    API_JSON_PATH = os.path.join(os.path.dirname(__file__), 'API.json')

    api_key_use, base_url_use = _load_api_config(model, API_JSON_PATH)
    
    client = OpenAI(api_key=api_key_use, base_url=base_url_use)

    if not os.path.exists(CODE_DIR):
        os.makedirs(CODE_DIR)

    save_data = []
    eval_set = load_json_file(QUESTION_PATH)
    
    print(f"开始处理 {len(eval_set)} 个问题，使用 {num_threads} 个线程")
    start_time = time.time()
    
    try:
        # 使用线程池进行并发处理
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_question = {
                executor.submit(
                    process_single_question, 
                    question_data, CODE_DIR, INSTRUCT_PATH, SAMPLE_PATH, model, client
                ): question_data for question_data in eval_set
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_question):
                completed += 1
                question_data = future_to_question[future]
                try:
                    result = future.result()
                    save_data.append(result)
                    print(f"进度: {completed}/{len(eval_set)} - 问题 {question_data['question_id']} 完成")
                except Exception as exc:
                    print(f"问题 {question_data['question_id']} 处理失败: {exc}")
                    
    except KeyboardInterrupt:
        print("\n用户中断，正在保存已完成的结果...")
    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:        
        save_json_file(PREDICT_PATH, sort_json_by_question_id(save_data))
import time

def request_gpt(max_retries, messages, client, model_name):
    """发送GPT请求，带重试机制"""
    for retry in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False
            ), True
        except Exception as e:
            print(f"Request failed: {e}. Retrying ({retry + 1}/{max_retries})...")
            if retry < max_retries - 1:  # 最后一次不等待
                time.sleep(2)
    
    return None, False
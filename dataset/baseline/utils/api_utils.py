import time
# 调用GPT 估计价格
class GPT():
    def __init__(self) -> None:
        self.total_input_tokens, self.total_output_tokens = 0, 0
        self.input_cost_per_milion_token, self.output_cost_per_milion_token = 2, 8
    def request_gpt(self, max_retries, messages, client, model_name):
        retry_count = 0
        success = False
        while retry_count < max_retries:
            # print("---------------------------------------------")
            # print(f"Request GPT: {messages}")
            # print("---------------------------------------------")
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False
                )
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                return response, True
            except Exception as e:
                print(f"Request failed with error: {e}. Retrying...")
                retry_count += 1
                time.sleep(2)
        
        response = []
        return response, success
    
    def cal_cost(self):
        cost = (self.total_input_tokens / 1e6) * self.input_cost_per_milion_token + \
                 (self.total_output_tokens / 1e6) * self.output_cost_per_milion_token
        return cost
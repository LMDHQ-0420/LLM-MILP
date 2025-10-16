from utils.common_utils import sort_api_json_by_question_id, save_json_file

# 文件路径
predict_json_path = 'output/deepseek-reasoner_20251016-204015/predict.json'

# 读取并排序
sorted_data = sort_api_json_by_question_id(predict_json_path)

# 保存回原文件
save_json_file(predict_json_path, sorted_data)

print("排序完成并已保存")
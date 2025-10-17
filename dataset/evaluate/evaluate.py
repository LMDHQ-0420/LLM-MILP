import argparse
import subprocess
import sys
import os
import json
import time
import zipfile
import shutil

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 更安全，直接加载
    return data

def save_json_file(path,data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def parse_parameters():
    parser = argparse.ArgumentParser(description='Manual to this script')
    parser.add_argument('--zip_path', type=str, required=True)
    args = parser.parse_args()
    return args

def calculate_er(code_path):
    """
    param: code_path: Python文件路径
    """
    if not os.path.exists(code_path):
        return 0, "", f"文件不存在: {code_path}"
    if not code_path.endswith('.py'):
        return 0, "", "文件不是.py格式"
    try:
        result = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            return 1, result.stdout, result.stderr
        else:
            return 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 0, "", "执行超时"
    except Exception as e:
        return 0, "", f"执行错误: {str(e)}"

def evaluate(pred_data, pred_path):
    output = list()
    for sample in pred_data:
        question_id = sample['question_id']
        code_path = os.path.join(pred_path, sample['code_path'])
        er, res, status = calculate_er(code_path)
        if er != 1:
            print(f'question_id:{question_id}，代码执行报错:\n{status}')
        else:
            print(f'question_id:{question_id}，代码执行成功')
            status = '执行成功'
        single_output = dict()
        single_output['question_id'] = question_id
        single_output['output'] = status
        output.append(single_output)
    return output

def _sync_unzip(zip_path, extract_to):
    """同步解压ZIP文件"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def save_uploaded_files(zip_path, target_dir):
    """保存并解压ZIP文件（同步版本）"""
    # 创建目标目录
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        if isinstance(e, PermissionError):
            raise ValueError(f"创建目录 {target_dir} 权限不足: {str(e)}")
        raise ValueError(f"创建目录 {target_dir} 失败: {str(e)}")

    # 复制ZIP文件（如果是路径字符串）
    if not isinstance(zip_path, str):
        raise ValueError("zip_path 必须是文件路径字符串")

    zip_filename = os.path.basename(zip_path)
    zip_save_path = os.path.join(target_dir, zip_filename)

    # 同步复制文件（避免大文件内存问题）
    with open(zip_path, 'rb') as src_f:
        with open(zip_save_path, 'wb') as dst_f:
            chunk_size = 1024 * 1024
            while True:
                chunk = src_f.read(chunk_size)
                if not chunk:
                    break
                dst_f.write(chunk)

    # 解压ZIP
    try:
        _sync_unzip(zip_save_path, target_dir)
    except zipfile.BadZipFile as e:
        raise ValueError(f"ZIP文件损坏，无法解压: {str(e)}")
    except Exception as e:
        raise ValueError(f"ZIP解压失败: {str(e)}")

    # 验证必要文件/目录
    code_dir = os.path.join(target_dir, 'code')
    if not os.path.isdir(code_dir):
        raise ValueError(f"ZIP压缩包中未找到 'code' 目录（路径：{code_dir}）")
    
    json_path = os.path.join(target_dir, 'predict.json')
    if not os.path.isfile(json_path):
        raise ValueError(f"ZIP压缩包中未找到 'predict.json' 文件（路径：{json_path}）")

    return code_dir, json_path

def start_eval():
    args = parse_parameters()
    zip_path = args.zip_path

    # 验证文件格式
    if not zip_path.endswith(".zip"):
        print("提交文件不是.zip格式")
        return False, "提交文件不是.zip格式"

    # 临时目录
    UPLOAD_DIR = "./tmp_code_execute_result"
    contestant_id = str(int(time.time()))  # 使用整数时间戳避免小数
    pred_path = os.path.join(UPLOAD_DIR, contestant_id)
    save_path = os.path.join(UPLOAD_DIR, 'output.json')

    try:
        code_dir, json_path = save_uploaded_files(
            zip_path=zip_path,
            target_dir=pred_path
        )
    except ValueError as e:
        print(f"文件处理错误: {e}")
        return False, str(e)

    # 检查预测文件
    if not os.path.exists(json_path):
        print(f"{json_path} 不存在，请检查上传内容")
        return False, "predict.json 不存在"

    # 加载预测数据
    try:
        pred_data = load_json_file(json_path)
    except json.JSONDecodeError as e:
        print(f"{json_path} 格式错误: {str(e)}")
        return False, f"JSON格式错误: {str(e)}"

    # 执行评估
    try:
        res = evaluate(pred_data, pred_path)
        save_json_file(save_path, res)
        print("评估完成")
    except Exception as e:
        print(f"评估出错: {str(e)}")
        return False, str(e)


    try:
        if os.path.exists(pred_path):
            shutil.rmtree(pred_path)
            print(f"成功删除临时目录: {pred_path}")
        else:
            print(f"临时目录不存在: {pred_path}")
    except Exception as e:
        print(f"删除临时目录失败: {e}")

if __name__ == "__main__":
    start_eval()
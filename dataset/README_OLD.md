## 文件结构

- **prompts/**  
  包含 5 个标准示例。  

- **dataset/question.json**  
  数据集文件，包含所有自然语言问题。  

- **dataset/database/**  
  存放 POI 数据库的 schema。  

- **commit/**  
  提交格式示例，选手应该提交一个zip压缩包，压缩包中包含：  
  1. **code/**：存放预测的规划代码文件夹  
  2. **predict.json**：在 `question.json` 的基础上，为每个问题增加预测的规划代码路径  

---

## 环境配置

```bash
# 创建并激活 conda 环境
conda create -n baseline_env python=3.11
conda activate baseline_env 

# 安装依赖
pip install -r requirements.txt

# 安装 scip 规划求解器
conda install -c conda-forge scip

# 运行 Baseline
bash baseline/few_shot.sh
```
# predict.json 示例
```sh
[
  {
    "question_id": "1",
    "question": "我和朋友打算从深圳坐高铁去上海玩三天两晚，预算7000元，计划从2025年6月10日出发，6月12日返回。想找评分4.5以上、价格低于800元每晚的连锁酒店；想去外滩这种高评分景点，门票控制在800元以内；交通方式希望以地铁公交为主。希望游玩性价比高的景点、餐厅与酒店。",
    "code_path": "./code/id_1.py"
  }
]
```
字段说明
- question_id：问题编号，对应 question.json 中的 question_id
- question：自然语言问题文本，对应 question.json 中的 question
- code_path：该问题对应的预测规划代码文件路径

# 代码规范
- 所生成的规划代码中仅允许使用 requirements.txt 中列出的依赖包。
- 不允许使用其他未声明的第三方库，以确保结果可执行初赛评估程序。
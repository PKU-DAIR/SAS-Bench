# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('../')
from main.predictor.llm import Predictor

pred = Predictor(model_from_pretrained='/home/lpc/models/glm-4-9b-chat/')

# %%
ask_content = '''请你现在扮演一位物理学科评分专家, 我们提供了一道题目和一个学生的回答, 请你根据题目分值、评分指南、错因列表，对照参考答案和解题分析,遵循以下规则, 对学生回答进行评分。
规则: 1. 你需要按学生回答的`每个步骤`分别进行评分，并对于有错误的答案从`错因列表`中选取一到多个错因作为打分依据，每个步骤的评估结果以{{'label': '', 'errors': []}}的形式输出。
2. 你需要根据每个步骤的评分结果输出最终的总分。
3. 最终的输出结果为json格式的数据，参考格式为{{'total': '', 'label': '', 'steps': []}}。其中`total`表示总分，`label`表示最终的评分结果，`steps`表示每个步骤的评分结果。
题目: {question}
分值: {total}
错因列表: {error_type}
参考答案: {reference}
解题分析: {analysis}
学生回答: {student_answer}
请你根据以上信息进行评分,并输出评分结果。
'''

ask_content = '''请作为物理学科评分专家，根据以下要求对学生的作答进行专业评估：

【评估任务】
依据题目信息、参考答案及评分标准，对学生的分步解答进行精细化评分，并输出结构化评分结果。

【评估流程】
1. 分步解析：
   - 拆解学生作答的每个解题步骤
   - 对每个步骤独立评估：
     * 判断正误（'label'）
     * 如存在错误，从错因列表中选取1项或多项主因（'errors'）
   - 单步评估格式：{{'label': '', 'errors': []}}

2. 综合评定：
   - 汇总各步骤得分计算总分
   - 给出整体评价（'label'）

3. 结果输出：
   - 采用标准JSON格式：
     {{
       'total': '总分',
       'label': '总体评价',
       'steps': [各步骤评估结果]
     }}

【评估材料】
- 试题内容：{question}
- 题目分值：{total}
- 错因类型：{error_type}
- 标准答案：{reference}
- 解析说明：{analysis}
- 学生作答：{student_answer}

【评分准则】
1. 严格对照标准答案的解题逻辑链
2. 错因标注需精准对应学生错误本质
3. 保持不同步骤间的评分尺度一致性
4. 对于创新解法需额外验证其科学性

【特别说明】
1. 公式错误、单位遗漏等细节问题需单独标注
2. 概念性错误与计算错误需区分处理
3. 部分正确的情况应给予相应步骤分

请按照上述规范完成评分，并输出标准化的评估结果。'''

pred(ask_content, build_message=True)

# %%

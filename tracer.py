from hypothesis import HypothesesSet
import random
import numpy as np
from openai import OpenAI
import os
import openai
os.environ["OPENAI_API_KEY"] = ''
#os.environ["http_proxy"] = "http://localhost:7890"
#os.environ["https_proxy"] = "http://localhost:7890"
import httpx
from httpx_socks import SyncProxyTransport

from typing import List, Optional, Dict, Tuple
import re

#transport = SyncProxyTransport.from_url("socks5://127.0.0.1:7891")

#http_client = httpx.Client(transport=transport)

#client = OpenAI(http_client=http_client)

# http_client = httpx.Client(base_url="https://api.vveai.com/v1")
#
# client = OpenAI(http_client=http_client)

# client = OpenAI(
#     api_key="sk-a957352d623f4f60ab52c5999fe51eb4",
#     base_url="https://api.vveai.com/v1",
# )

openai.base_url = "https://api.vveai.com/v1/"

target_role_map = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist",
}

class StanceTracer:
    """立场追踪器：实现感知追踪、假设生成、传播、权重更新、聚合推理"""
    def __init__(self, target: str, n_hypotheses: int = 4):
        self.target = target  # 目标议题
        self.n_hypotheses = n_hypotheses  # 假设数量
        self.role = target_role_map.get(self.target, "expert")
        # self.client = client

    def parse_text_linguist(self, tweet: str, ) -> dict:
        """按prompt解析文本，提取关键语言学特征特征A"""
        parse_prompt = ("As a linguist, analyze the provided text by examining its linguistic features and their contribution to meaning. Address elements such as grammatical structure, tense and inflection, speech acts, rhetorical devices, and lexical choices. Do nothing else.")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": parse_prompt},
                {"role": "user", "content": f"Tweet: {tweet}"}
            ],
            temperature=0,
            max_tokens=1024
        )
        return {'parse': response.choices[0].message.content.strip()}  # 解析结果

    def parse_text_social(self, tweet: str, ) -> dict:
        """按prompt解析文本，提取关键社会学特征特征A"""
        parse_prompt = ("Analyze the provided text by examining its key elements contained in the quote, such as characters, events, parties, religions, etc. Also explain their relationship with {target} (if exist). Do nothing else.")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": parse_prompt},
                {"role": "user", "content": f"Tweet: {tweet}"}
            ],
            temperature=0,
            max_tokens=1024
        )
        return {'parse': response.choices[0].message.content.strip()}  # 解析结果

    def parse_text_user(self, tweet: str, ) -> dict:
        """按prompt解析文本，提取关键用户社交特征特征A"""
        parse_prompt = ("Analyze the provided text by examining its user features and their contribution to meaning. Focus on the content, hashtags, Internet slang and colloquialisms, emotional tone, implied meaning, and so on. Do nothing else.")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": parse_prompt},
                {"role": "user", "content": f"Tweet: {tweet}"}
            ],
            temperature=0,
            max_tokens=1024
        )
        return {'parse': response.choices[0].message.content.strip()}  # 解析结果

    def generate_thought_linguist(self, tweet: str, parse_result: dict) -> dict:
        """基于文本和解析结果生成思考状态S"""
        thought_state_prompt = ("As a linguist, describe the author's state of mind as evidenced by the language used in this tweet.")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": thought_state_prompt},
                {"role": "user", "content": f"Tweet: {tweet}\nParse: {parse_result['parse']}"}
            ],
            temperature=0,
            max_tokens=1024
        )
        return {"state": response.choices[0].message.content.strip()}  # 思考状态S

    def generate_thought_social(self, tweet: str, parse_result: dict) -> dict:
        """基于文本和解析结果生成思考状态S"""
        thought_state_prompt = (f"As a {self.role}, describe the author's state of mind as evidenced by the language used in this tweet.")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": thought_state_prompt},
                {"role": "user", "content": f"Tweet: {tweet}\nParse: {parse_result['parse']}"}
            ],
            temperature=0,
            max_tokens=1024
        )
        return {"state": response.choices[0].message.content.strip()}  # 思考状态S

    def generate_thought_user(self, tweet: str, parse_result: dict) -> dict:
        """基于文本和解析结果生成思考状态S"""
        thought_state_prompt = ("As a social media veteran, describe the author's state of mind as evidenced by the language used in this tweet.")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": thought_state_prompt},
                {"role": "user", "content": f"Tweet: {tweet}\nParse: {parse_result['parse']}"}
            ],
            temperature=0,
            max_tokens=1024
        )
        return {"state": response.choices[0].message.content.strip()}  # 思考状态S

    def _prompting_for_ordered_list(self, prompt: str) -> List[str]:
        """生成有序假设列表（确保数量为n_hypotheses）"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Generate numbered list without extra comments."},
                {"role": "user", "content": f"{prompt}\n"}
            ],
            temperature=0,
            max_tokens=1024
        )
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        # 提取编号假设（1. ... 2. ...）
        ordered_hypotheses = [line.split(". ", 1)[1] for line in lines if line.startswith(tuple(f"{i}." for i in range(1, self.n_hypotheses + 1)))]
        # 补全数量（若不足）
        while len(ordered_hypotheses) < self.n_hypotheses:
            ordered_hypotheses.append(f"Neutral stance (auto-completed)")
        return ordered_hypotheses[:self.n_hypotheses]

    def initialize(self, tweet: str) -> HypothesesSet:
        """整合文本、解析结果、思考状态S生成立场假设"""
        # 解析文本
        parse_result = self.parse_text_linguist(tweet)  # 解析结果：{"analysis": "..."}

        # 生成思考状态S
        thought_state = self.generate_thought_linguist(tweet, parse_result)  # 思考状态S：{"thought_state": "..."}

        # 构建整合三要素的上下文输入
        context_input = (
            f"<tweet>\n{tweet}\n</tweet>\n\n"  # 原始文本
            f"<parse>\n{parse_result['parse']}\n</parse>\n\n"  # 解析结果
            f"<state>\n{thought_state['state']}\n</state>"  # 思考状态S
        )

        # 生成立场假设的prompt
        n_hypotheses_str = str(self.n_hypotheses)
        belief_query = (
            f"{context_input}\n\n"
            f"From a linguistic perspective, analyze the provided tweet, its parses, and state assessments to generate a numbered list of {n_hypotheses_str} hypotheses about the author's internal reasoning that led to the tweet's stance (favor/against/none) on the {self.target}.")

        # 调用LLM生成有序假设列表
        _hypotheses_list = self._prompting_for_ordered_list(
            prompt=belief_query
        )
        hypotheses_list = [hyp.strip() for hyp in _hypotheses_list if hyp.strip()]  # 清洗假设列表

        # 初始权重均匀分布
        weights = np.ones(len(hypotheses_list)) / len(hypotheses_list)  # 如4个假设各0.25权重

        # 返回假设集
        return HypothesesSet(
            target=self.target,
            contexts={"tweet": tweet, "linguistic_parse": parse_result['parse']},
            hypotheses=hypotheses_list,
            weights=weights,
            thought_state=parse_result['parse'],
            parse=thought_state['state']
        )

    def propagate_to_social(self, ling_hypotheses: HypothesesSet, tweet: str) -> HypothesesSet:
        """第一次传播：语言学→社会领域假设"""
        return self._propagate(
            ling_hypotheses, tweet,
            parse_func=self.parse_text_social,
            thought_func=self.generate_thought_social,
            agent_role=self.role
        )

    def propagate_to_user(self, social_hypotheses: HypothesesSet, tweet: str) -> HypothesesSet:
        """第二次传播：社会领域→用户社交假设"""
        return self._propagate(
            social_hypotheses, tweet,
            parse_func=self.parse_text_user,
            thought_func=self.generate_thought_user,
            agent_role="social media veteran"
        )

    def _propagate(self, prev_hypotheses: HypothesesSet, tweet: str, parse_func, thought_func,
                   agent_role: str) -> HypothesesSet:
        """新假设仅依赖：上一时刻假设 + 当前解析 + 当前状态 + 文本"""
        # 步骤1：按权重概率采样上一时刻假设（仅依赖上一时刻假设）
        sampled_hypotheses = self.resample_hypotheses_by_weight(prev_hypotheses)

        # 步骤2：当前Agent解析文本
        current_parse = parse_func(tweet)  # 当前Agent的解析结果

        # 步骤3：当前Agent生成思考状态
        current_thought = thought_func(tweet, current_parse)  # 当前思考状态S

        # 步骤4：构建传播提示（仅含上一时刻假设、当前解析、当前状态）！！！
        propagation_prompts = [
            f"<current_parse>（{agent_role}）\n{current_parse['parse']}\n</current_parse>\n"
            f"<current_thought_state>\n{current_thought['state']}\n</current_thought_state>\n"
            f"Task: Generate a new hypothesis about the author's stance on {self.target} based on the above. "
            f"Only use information from the previous hypothesis, current parse, and current thought state. "
            f"Do nothing else."
            for hyp in sampled_hypotheses.hypotheses
        ]

        # 生成新假设文本
        new_texts = self._batch_generate(propagation_prompts)

        current_context = {
            "tweet": tweet,  # 文本
            f"{agent_role}_parse": current_parse['parse'],  # 当前解析
            "current_thought": current_thought['state']  # 当前状态
        }

        new_hypotheses = HypothesesSet(
            target=self.target,
            contexts=current_context,  # 仅当前上下文
            hypotheses=new_texts,  # 新假设
            weights=sampled_hypotheses.weights,  # 继承上一时刻采样权重
            thought_state=current_thought['state'],  # 仅当前状态
            parse=current_parse['parse'],  # 仅当前解析
            parent_hypotheses=sampled_hypotheses.hypotheses  # 仅上一时刻假设
        )

        # 逆向贝叶斯更新权重（仅基于当前信息）
        new_hypotheses = self._weigh(new_hypotheses, current_thought)

        return new_hypotheses

    def compute_ess(self, hypotheses: HypothesesSet) -> float:
        """计算有效样本量（ESS）：衡量权重分布均匀性"""
        return 1.0 / np.sum(np.square(hypotheses.weights))  # 标准ESS公式

    # ------------------------------ 逆向贝叶斯权重更新 ------------------------------

    def _weigh(self, hyp_set: HypothesesSet, new_thought: Dict) -> HypothesesSet:
        """逆向贝叶斯权重更新：后验权重 = P(D|H) × P(H) / 归一化常数"""
        likelihood_results = self.prompt_likelihood(hyp_set, new_thought)
        # 直接使用后验权重（已在prompt_likelihood中计算）
        posterior_weights = likelihood_results['posterior_weights']
        hyp_set.update_weights(posterior_weights)
        hyp_set.weight_details = likelihood_results  # 存储完整评估细节
        return hyp_set

    def prompt_likelihood(self, hyp_set: HypothesesSet, new_thought: Dict) -> Dict:
        """
        逆向贝叶斯似然度评估：P(D|H)，即假设H为真时，观察到新思考状态D的概率
        """

        def map_prob_to_score(prob: str) -> float:
            """将概率百分比（如"80%"）转换为0-10的分数"""
            try:
                # 提取数字（支持"80%"或"80"）
                prob_value = float(re.search(r'(\d+)', prob).group(1)) / 100  # 转为0-1
                return prob_value * 10  # 映射到0-10分
            except:
                return 0.5  # 无效响应默认0.5

        # 1. 系统提示：明确任务是计算P(D|H)
        system_prompt = (
            "You MUST distinguish probabilities for different hypotheses. "
            "If hypothesis H directly supports observed state D, P(D|H) = 80-100%. "
            "If H contradicts D, P(D|H) = 0-20%. "
            "If H is neutral to D, P(D|H) = 40-60%. "
            "Output ONLY a percentage (no extra text) and brief reasoning."
        )

        # 2. 观察到的状态D（新思考状态）
        observed_state = new_thought['state']  # 现有状态D


        # 3. 为每个假设H构建评估prompt（H为新生成的假设，作为先验条件）
        prompts = [
            f"<hypothesis H>\n{hyp.hypothesis}\n</hypothesis H>\n"  # 假设H（先验条件）
            f"<observed state D>\n{observed_state}\n</observed state D>\n\n"  # 现有状态D
            "Question: What is the probability P(D|H) that state D would be observed if hypothesis H is true? "
            "Output a percentage (0%-100%) and explain your reasoning."
            for hyp in hyp_set.hypotheses  # 遍历所有假设H
        ]


        # 4. 批量调用LLM评估P(D|H)
        raw_predictions = self._batch_generate(prompts, system_prompt=system_prompt)

        # 5. 解析概率和推理过程
        likelihoods = []  # P(D|H)似然度（0-1）
        reasonings = []
        raw_probs = []  # 原始概率百分比（如"80%"）
        for response in raw_predictions:
            if "Probability:" in response:
                prob_part, reasoning_part = response.split("Probability:", 1)
                prob = reasoning_part.split("\n")[0].strip()  # 提取概率（如"80%"）
                reasoning = reasoning_part.split("Reasoning:", 1)[
                    1].strip() if "Reasoning:" in reasoning_part else "No reasoning"
                raw_probs.append(prob)
                likelihoods.append(map_prob_to_score(prob))  # 转换为0-10分
                reasonings.append(reasoning)
            else:
                raw_probs.append("50%")  # 默认50%
                likelihoods.append(5.0)  # 映射为5分（0.5概率）
                reasonings.append("No response")

        # 6. 计算后验权重：后验 ∝ 似然度（P(D|H)） × 先验权重（P(H)）
        prior_weights = hyp_set.weights  # 先验权重P(H)（采样后的权重分布）
        likelihood_scores = np.array(likelihoods) / 10  # 似然度P(D|H)（归一化到0-1）
        posterior_weights = likelihood_scores * prior_weights  # 后验 ∝ P(D|H) × P(H)
        posterior_weights = posterior_weights / np.sum(posterior_weights)  # 归一化

        return {
            'prompts': prompts,
            'raw_predictions': raw_predictions,
            'raw_probs': raw_probs,  # 原始概率百分比
            'likelihoods': likelihood_scores,  # P(D|H)似然度（0-1）
            'prior_weights': prior_weights,  # 先验权重P(H)
            'posterior_weights': posterior_weights,  # 后验权重P(H|D)
            'reasonings': reasonings
        }

    # ------------------------------ 辅助函数 ------------------------------

    def _batch_generate(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        """批量调用LLM生成响应"""
        system_prompt = system_prompt or "Generate a concise response."
        responses = []
        for prompt in prompts:
            res = openai.chat.completions.create(model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=512
            )
            responses.append(res.choices[0].message.content.strip())
        return responses

    def _prompting_for_ordered_list(self, prompt: str) -> List[str]:
        """生成有序假设列表（确保数量为n_hypotheses）"""
        res = openai.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": "Generate numbered list without extra comments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1024
        )
        response = res.choices[0].message.content.strip()
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        # 提取编号假设（1. ... 2. ...）
        ordered_hypotheses = [line.split(". ", 1)[1] for line in lines if
                              line.startswith(tuple(f"{i}." for i in range(1, self.n_hypotheses + 1)))]
        # 补全数量（若不足）
        while len(ordered_hypotheses) < self.n_hypotheses:
            ordered_hypotheses.append(f"Neutral stance (auto-completed)")
        return ordered_hypotheses[:self.n_hypotheses]

    def resample_hypotheses_by_weight(self, hypotheses: HypothesesSet) -> HypothesesSet:
        """按权重概率采样假设（无条件执行）"""
        # 按权重作为概率采样假设索引
        resampled_idxs = random.choices(
            population=range(len(hypotheses.hypotheses)),  # 假设索引：0, 1, ..., n-1
            weights=hypotheses.weights,  # 采样概率=假设权重
            k=len(hypotheses.hypotheses)  # 采样数量=假设数量
        )

        # 步骤2：提取采样后的假设文本和权重（继承原权重分布）
        texts = [hypotheses.hypotheses[idx].hypothesis for idx in resampled_idxs]  # 按采样索引选择文本
        weights = np.array([hypotheses.weights[idx] for idx in resampled_idxs])  # 继承原权重
        weights = weights / np.sum(weights)  # 归一化确保权重和为1

        # 步骤3：提取父假设和权重细节
        parents = [hypotheses.hypotheses[idx] for idx in resampled_idxs]  # 父假设链
        weight_details = {}
        if hasattr(hypotheses, 'weight_details') and hypotheses.weight_details:
            weight_details['prompts'] = [hypotheses.weight_details['prompts'][idx] for idx in resampled_idxs]
            weight_details['reasonings'] = [hypotheses.weight_details['reasonings'][idx] for idx in resampled_idxs]

        # 步骤4：创建采样后的假设集
        return HypothesesSet(
            target=hypotheses.target,
            contexts=hypotheses.contexts,
            hypotheses=texts,
            weights=weights,  # 权重，采样后的分布
            thought_state=hypotheses.thought_state,
            parse=hypotheses.parse,
            parent_hypotheses=parents,
            weight_details=weight_details
        )

    def aggregate_final_stance(self, tweet ,final_hypotheses: HypothesesSet) -> str:
        """聚合最终假设，生成最终立场推理链（修正属性引用）"""
        # 构建聚合提示
        agg_prompt = (
                "Synthesize the following hypotheses into a final stance (favor/against/none) on the topic, "
                "with a reasoning chain that integrates all hypotheses, prioritizing those with higher weights.\n\n"
                "<final_hypotheses_with_weights>\n"
                + "\n".join([f"Hypothesis {i + 1} (Weight: {hyp.weight:.2f}): {hyp.hypothesis}"
                             for i, hyp in enumerate(final_hypotheses.hypotheses)])  # 修正：遍历hypotheses列表
                + "\n</final_hypotheses_with_weights>\n\n"
                  f"Final Stance on {self.target} (favor/against/none) with reasoning chain:"
                  f"Your answer must be one of the following: \"AGAINST\", \"FAVOR\", \"NONE\". Do not include any additional text, explanations, or clarifications"
        )

        context_input = (
            f"<tweet>\n{tweet}\n</tweet>\n\n"  # 原始文本
            f"<agg_prompt>\n{agg_prompt}\n</agg_prompt>\n\n"  # 解析结果
        )

        # 调用LLM生成最终立场（直接使用client，避免依赖未定义的_call_llm）
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Be concise and decisive. Output only the final stance based tweet and agg_prompt."},
                {"role": "user", "content": context_input}
            ],
            temperature=0,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()

    def run_full_pipeline(self, tweet: str) -> Tuple[str, HypothesesSet]:
        """执行完整立场检测流程：初始化→社会传播→用户传播→结果聚合（修正属性引用）"""
        # 语言学Agent生成初始假设
        ling_hypotheses = self.initialize(tweet)

        # print("\n=== Linguist假设 ===")
        # for i, hyp in enumerate(ling_hypotheses.hypotheses):  # 修正：遍历hypotheses列表
        #     print(f"Hypothesis {i + 1}: {hyp.hypothesis} (Weight: {hyp.weight:.2f})")  # 修正：hyp.text和hyp.weight

        # 传播到社会领域Agent
        social_hypotheses = self.propagate_to_social(ling_hypotheses, tweet)

        # print("\n=== Domain假设 ===")
        # for i, hyp in enumerate(social_hypotheses.hypotheses):  # 修正：遍历hypotheses列表
        #     print(f"Hypothesis {i + 1}: {hyp.hypothesis} (Weight: {hyp.weight:.2f})")  # 修正：hyp.text和hyp.weight

        # 传播到用户社交Agent
        user_hypotheses = self.propagate_to_user(social_hypotheses, tweet)
        # print("\n=== User假设 ===")
        # for i, hyp in enumerate(user_hypotheses.hypotheses):  # 修正：遍历hypotheses列表
            # print(f"Hypothesis {i + 1}: {hyp.hypothesis} (Weight: {hyp.weight:.2f})")  # 修正：hyp.text和hyp.weight

        # 聚合最终立场
        final_stance = self.aggregate_final_stance(tweet, user_hypotheses)
        # print("\n=== 最终立场 ===")
        # print(final_stance)

        return final_stance, user_hypotheses


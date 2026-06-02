import openai
from openai import OpenAI
import pandas as pd
import time
import logging
import argparse

import os
os.environ["OPENAI_API_KEY"] = ''
#os.environ["http_proxy"] = "http://localhost:7890"

#os.environ["https_proxy"] = "https://api.vveai.com/v1"
import httpx
from httpx_socks import SyncProxyTransport
from tracer import StanceTracer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torch.utils.data import Dataset
import csv

# transport = SyncProxyTransport.from_url("https://api.vveai.com/v1")
#
# http_client = httpx.Client(transport=transport)
#
# client = OpenAI(http_client=http_client)


openai.base_url = "https://api.vveai.com/v1/"

# http_client = httpx.Client(base_url="https://api.vveai.com/v1")
#
# client = OpenAI(http_client=http_client)
# assign experts for target
target_role_map = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist"
}

def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

class StanceDataset(Dataset):
    def __init__(self, f: str):
        self.datalist = load_csv_data(f)
        self.data = dict()
        # 用于存储每个 Tweet 的单词数量
        self.word_counts = []
        self.preprocess_data()

    def preprocess_data(self):
        self.data["Tweet"] = []
        self.data["Topic"] = []
        self.data["Stance"] = []
        self.data["Word"] = []

        for i in self.datalist.index:
            row = self.datalist.iloc[i]
            tweet = row["Tweet"]
            self.data["Tweet"].append(tweet)
            self.data["Topic"].append(row["Target"])
            self.data["Stance"].append(row["Stance"])
            # 统计单词数量，简单以空格分割字符串
            word_count = len(tweet.split())
            self.data["Word"].append(word_count)
            self.word_counts.append(word_count)

    def __getitem__(self, index):
        item = dict()
        for k in self.data:
            item[k] = self.data[k][index]
        # 将单词数量添加到返回的样本中
        item["WordCount"] = self.word_counts[index]
        return item

    def __len__(self):
        return len(self.datalist)

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def add_predictions_sequential(dataset):

    results = []  # To store the results
    for data in tqdm(dataset, total=len(dataset), desc='Stance Detecting'):
        tweet = data['Tweet']
        target = data['Topic']

        tracer = StanceTracer(target=target, n_hypotheses=4)

        final_stance, user_hypotheses = tracer.run_full_pipeline(tweet)

        # Construct the result for the current tweet and add it to the results list
        result = {
            'Tweet': tweet,
            'Target': target,
            'Final Judgement': final_stance,
            'hypotheses': user_hypotheses
        }
        results.append(result)

    with open("result_Sem16_1079.txt", "w") as file:
         for number in results:
             file.write(str(number['Final Judgement']) + "\n")

def read_txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取所有行并保存到列表中
            lines = file.readlines()
            # 去除每行末尾的换行符（可选）
            lines = [line.strip() for line in lines]
        return lines
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

def main():
    # Load the data
    dataset = StanceDataset("./data/Sem16/test.csv")

    indices = list(range(0, 1))

    # 创建仅包含第501到第1000个样本的子集数据集
    subset_dataset = SubsetDataset(dataset, indices)

    add_predictions_sequential(subset_dataset)

    content_list = read_txt_to_list('./result_Sem16_1079.txt')
    predictions = []
    labels = []

    # for i in content_list:
    #     if i == 'Against':
    #         j = 0
    #     elif i == 'None':
    #         j = 1
    #     elif i == 'Favor':
    #         j = 2
    #     else:
    #         j = 2
    #     predictions.append(j)

    for i in content_list:
        if i == 'AGAINST':
            j = 0
        elif i == 'NONE':
            j = 1
        elif i == 'FAVOR':
            j = 2
        else:
            j = 2
        predictions.append(j)

    for data in tqdm(subset_dataset, total=len(subset_dataset)):
        if data['Stance'] == 'AGAINST':
            i = 0
        elif data['Stance'] == 'NONE':
            i = 1
        elif data['Stance'] == 'FAVOR':
            i = 2
        labels.append(i)

    # 计算 F1-score
    f1 = f1_score(labels, predictions, average='weighted')  # 对于二分类任务
    accuracy = accuracy_score(labels, predictions)
    print(f"准确率: {accuracy}")
    print(f"F1-score: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--use-tracing', action='store_false', help='whether to run the model with thought tracing')
    parser.add_argument('--tracing-model', type=str, default='gpt-4o-mini', help='Model to use to answer final question.')
    parser.add_argument('--n-hypotheses', type=int, default=4, help='number of hypotheses to generate for each input', )
    parser.add_argument('--target-perceptions', type=str, default='sight', help='target perceptions to test')  # 'sight,hearing,overall',
    parser.add_argument('--use-helper-llm', action='store_true', help='whether to use user helper llm for identifying target agent and labeling actions', )
    parser.add_argument('--existing-traces', default=None, help='path to existing traces')
    parser.add_argument('--input-is-chat', action='store_true', help="whether the input is a chat or not")
    parser.add_argument('--dataset', type=str, default='tomi', help='dataset')
    parser.add_argument('--likelihood-estimate', default="prompting", type=str, choices=['rollout', 'prompting'], help='likelihood estimation method')
    parser.add_argument('--tracer-type', default='tracer', type=str, help='tracer type')

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max_questions_per_type', type=int, default=50)
    parser.add_argument('--tomi-set', type=str, default="paraphrased_tomi", help='ToMi subset to test')
    parser.add_argument('--input_file', type=str, help='Input file to test', default='test')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use to answer final question.')
    parser.add_argument('--use-cot', type=bool, default=False,help='whether to run the model with zero-shot cot',)
    parser.add_argument('--run-id', type=str, default='tracer-first-run', help='Run ID')
    parser.add_argument('--print', action='store_false', help='whether to print the outputs')
    parser.add_argument('--existing-savepoint', default=None, help='path to existing savepoint')
    parser.add_argument('--reasoning-effort', type=str, help='Reasoning effort')
    args = parser.parse_args()
    main()

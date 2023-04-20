import copy
import json
import time
from urllib.request import urlopen
import random
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_creation.client_sample import LLMClient


class ContextPool:
    def __init__(self, which_json='annotation', max_length=-1):
        super().__init__()
        assert which_json in ['annotation', 'generated'], ' which json must belong to annotation or generated.'

        if which_json == 'annotation':
            r = urlopen("https://instruct-pix2pix.eecs.berkeley.edu/human-written-prompts.jsonl")
        else:
            r = urlopen("https://instruct-pix2pix.eecs.berkeley.edu/gpt-generated-prompts.jsonl")

        self.context_dict = [json.loads(l) for l in r]
        if max_length > -1:
            ori_length = len(self.context_dict)
            self.context_length = [len(' '.join(s.values())) for s in self.context_dict]
            self.context_dict = [v for v,length in zip(self.context_dict, self.context_length) if length<max_length]
            print('Screen all the sentences less than {} tokens. Only {}/{} left.'.format(max_length, len(self.context_dict), ori_length))

        # import pdb; pdb.set_trace()
        self.input_edit_templ = "input: {} edit: {} "
        self.input_edit_output_templ = "input: {} edit: {} output: {}"
        self.llm_client = LLMClient()

    def get_n_context(self, num=40):
        # import pdb; pdb.set_trace()
        samples = random.sample(self.context_dict, num)
        # print(max([len(s) for s in samples]))
        input_edit_str = [self.input_edit_templ.format(s['input'], s['edit']) for s in samples]
        input_edit_output_str = [self.input_edit_output_templ.format(s['input'], s['edit'], s['output']) for s in samples]
        input_edit_str = '\n'.join(input_edit_str)
        input_edit_output_str = '\n'.join(input_edit_output_str)
        return input_edit_str, input_edit_output_str

    def query_edit_output(self, input_caption):
        instruction = 'Let us play a game. Following some examples, fill in edit and output.\n'
        input_edit_str, input_edit_output_str = self.get_n_context()
        # print(input_edit_str)
        # import pdb; pdb.set_trace()
        q_input_edit_str = "\n".join([input_edit_output_str, input_caption])
        answer = self.ask_llm(instruction+q_input_edit_str)
        return answer

    # def query_edit(self, input_caption):
    #     # import pdb; pdb.set_trace()
    #     # instruction = 'Instruction: Based on the following examples, finish the \"edit\" sentence.\n'
    #     instruction = 'Let us play a game. Following some examples, fill in edit.\n'
    #     input_edit_str, input_edit_output_str = self.get_n_context()
    #     # print(input_edit_str)
    #     # import pdb; pdb.set_trace()
    #     q_input_edit_str = "\n".join([input_edit_str, input_caption])
    #     answer = self.ask_llm(instruction+q_input_edit_str)
    #     return answer
    #
    # def query_output(self, input_caption):
    #
    #     input_edit_str, input_edit_output_str = self.get_n_context(25)
    #     q_input_edit_output_str = "\n".join([input_edit_output_str, input_caption])
    #     # import pdb; pdb.set_trace()
    #     answer = self.ask_llm(q_input_edit_output_str)
    #     return answer

    def ask_llm(self, prompt):
        request_data = {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 1,
            "top_p": 1,
            "n": 2,
            "stream": False,
            "logprobs": None,
            "stop": "\n"
        }
        try:
            response = self.llm_client.send_request('text-chat-davinci-002', request_data)
            answer = [response['choices'][i]['text'] for i in range(len(response['choices']))]
            return answer
        except Exception as ex:
            print(str(ex))
            return []

def add_answer_to_json(res_dict_i, answer):
    def except_exit():
        res_dict_i['double_edit'] = []
        res_dict_i['double_output'] = []
        return res_dict_i

    try:
        edits, outputs = [], []
        for a_i in answer:
            edit_i, output_i = a_i.split('output:')
            edit_i, output_i = edit_i.strip(" "), output_i.strip(" ")
            if len(edit_i) < 5 or len(output_i) < 5: continue
            edits.append(edit_i)
            outputs.append(output_i)
        res_dict_i['double_edit'], res_dict_i['double_output'] = edits, outputs
        return res_dict_i

    except Exception as ex:
        except_exit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_num', type=int, default=5000)
    parser.add_argument('--begin_num', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    llm_context_pool = ContextPool(max_length=250)
    query_context_pool = ContextPool(which_json='generated')

    t = time.time()
    query_length = len(query_context_pool.context_dict)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'double_edit_{}_{}.json'.format(args.begin_num, args.begin_num+args.shard_num))
    res_dict = []

    for i in tqdm(range(args.begin_num, args.begin_num+args.shard_num)):
        idx = random.randint(0, query_length)
        dict_i = query_context_pool.context_dict[idx]
        input_caption = query_context_pool.input_edit_templ.format(dict_i['output'], '')
        answer = llm_context_pool.query_edit_output(input_caption)

        res_dict_i = copy.deepcopy(dict_i)
        add_answer_to_json(res_dict_i, answer)
        res_dict.append(res_dict_i)

        print(answer)

    with open(save_path, 'w') as f:
        json.dump(res_dict, f)

    print(time.time() - t)

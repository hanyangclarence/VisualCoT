import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
import torch
import random
import openai
from tqdm import tqdm
from transformers import GPT2Tokenizer
import pdb
import pickle
import glob
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import inspect

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(seed=0)

def parse_sentence(raw_result_list):
    output_list = []
    for raw_result in raw_result_list:
        if isinstance(raw_result, str):
            raw_result = raw_result.strip(" ")
            tmp_result_list = raw_result.split(",")
            tmp_output_list = []
            for tmp_result in tmp_result_list:
                tmp_output_list +=tmp_result.split(" and ")
            output_list += tmp_output_list
        elif isinstance(raw_result, list):
            raw_ouput = parse_sentence(raw_result)
            output_list +=raw_ouput
    output_list = [ele.strip() for ele in output_list]
    output_list = [ele for ele in output_list if len(ele)>0]
    return output_list

def parge_obj_name(raw_result_list):
    output_list = []
    for raw_result in raw_result_list:
        if isinstance(raw_result, str):
            raw_result = raw_result.strip(" ")
            tmp_result_list = raw_result.split(",")
            tmp_output_list = []
            for tmp_result in tmp_result_list:
                tmp_output_list +=tmp_result.split(" and ")
            output_list += tmp_output_list
        elif isinstance(raw_result, list):
            raw_ouput = parse_sentence(raw_result)
            output_list +=raw_ouput
    output_list = [ele[2:] if ele.lower().startswith("a ") else ele for ele in output_list]
    output_list = [ele[3:] if ele.lower().startswith("an ") else ele for ele in output_list]
    output_list = [ele[4:] if ele.lower().startswith("the ") else ele for ele in output_list]
    output_list = [ele.strip() for ele in output_list]
    output_list = [ele for ele in output_list if len(ele)>0]
    return output_list


## initial cleaning for reference QA results; Please use vqav2 eval script for the final number
def process_answer(answer):
    answer = answer.replace('.', '').replace(',', '').lower()
    to_be_removed = {'a', 'an', 'the', 'to', ''}
    answer_list = answer.split(' ')
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return ' '.join(answer_list)


class VisualCOT_AOKVQA:
    def __init__(self, args, apikey_list):
        self.args = args
        self.chain_of_thoughts = args.chain_of_thoughts
        ## loading input questions (and answer for reference accuracy computing)
        self.apikey_list = apikey_list
        self.apikey_idx = 0
        openai.api_key = self.apikey_list[self.apikey_idx]

        self.device = torch.device(args.device)

        if args.engine == "opt":
            if args.with_six_gpus:
                self.initialize_opt_6gpu()
            elif args.with_one_gpu:
                self.initialize_opt_small()
            else:
                self.initialize_opt()
        elif args.engine == "llama":
            self.initialize_llama()
        elif args.engine == "bloom":
            from plm.bloom import get_model_and_tokenizer
            self.model, self.tokenizer = get_model_and_tokenizer(name="microsoft/bloom-deepspeed-inference-int8",
                                                                 dtype="int8")
        elif args.engine == "chat":
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # elif args.engine == "codex":
        #     import tiktoken
        #     self.tokenizer = tiktoken.encoding_for_model(self.args.engine_name)
        # elif args.engine == "instruct":
        #     import tiktoken
        #     self.tokenizer = tiktoken.encoding_for_model("davinci-instruct-beta")
        elif args.engine == "chat-test":
            self.initialize_opt_small()
        elif args.engine in ['ada', 'babbage', 'curie', 'davinci', 'gpt3']:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(self.args.engine_name)
            # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        if self.args.use_blip2:
            if not self.args.with_blip2_api:
                blip2_model_name = "pretrain_flant5xl" if self.args.use_v100 else "pretrain_flant5xxl"
                from lavis.models import load_model_and_preprocess
                if args.engine == "chat-test":
                    self.blip2_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
                    self.blip2_model, self.blip2_vis_processors, _ = load_model_and_preprocess(name="blip2_t5",
                                                                                            model_type=blip2_model_name,
                                                                                            is_eval=True,
                                                                                            device=self.blip2_device)
                    import pdb
                    pdb.set_trace()
                else:
                    self.blip2_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
                    self.blip2_model, self.blip2_vis_processors, _ = load_model_and_preprocess(name="blip2_t5",
                                                                                            model_type=blip2_model_name,
                                                                                            is_eval=True,
                                                                                            device=self.blip2_device)
                print("Finish loading BLIP2 model")
            else:
                self.blip2_api = API_URLS = [ "http://localhost:5000/api/generate", ]

        if args.with_clip_verify or args.choice_only:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            model = model.to(self.device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.clip_model, self.clip_processor = model, processor

        if args.oracle_attend:
            with open(self.args.val_sim_file, "rb") as fh:
                self.val_oracle_attend = pickle.load(fh)

        self.temp_question = "What is the person doing?"

    def sleep(self, sleep_time=1.5, switch_key=False):
        if self.args.engine == "codex":
            sleep_time = 0.1
        if switch_key:
            self.apikey_idx += 1
            if self.apikey_idx >= len(self.apikey_list):
                self.apikey_idx = 0
        openai.api_key = self.apikey_list[self.apikey_idx]
        time.sleep(sleep_time)

    def initialize_llama(self):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        self.model = LlamaForCausalLM.from_pretrained(self.args.llama_path,
                                                                    device_map="auto", torch_dtype=torch.float16)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.args.llama_path)

    def initialize_opt_small(self):
        from transformers import OPTForCausalLM
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b",
                                                    device_map="auto", torch_dtype=torch.float16)
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")

    def initialize_opt(self):
        num_device = 8
        import math
        num_layers = math.ceil(64 / 8.0)
        assert torch.cuda.device_count() >= num_device
        opt_device_map = {'model.decoder.embed_tokens': 0,
                          'lm_head': 0,
                          'model.decoder.embed_positions': 0,
                          'model.decoder.final_layer_norm': 0,
                          'model.decoder.layers.0': 0}
        for layer in range(64):
            layer_name = "model.decoder.layers.%s" % (str(layer))
            device = layer // num_layers
            opt_device_map[layer_name] = device
        from transformers import OPTForCausalLM
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-66b",
                                                    device_map=opt_device_map, torch_dtype=torch.float16)
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-66b")

    def initialize_opt_6gpu(self):
        num_device = 6
        import math
        num_layers = math.ceil(64 / 6.0)
        assert torch.cuda.device_count() >= num_device
        opt_device_map = {'model.decoder.embed_tokens': 0,
                          'lm_head': 0,
                          'model.decoder.embed_positions': 0,
                          'model.decoder.final_layer_norm': 0,
                          'model.decoder.layers.0': 0}
        for layer in range(64):
            layer_name = "model.decoder.layers.%s" % (str(layer))
            device = layer // num_layers
            opt_device_map[layer_name] = device
        from transformers import OPTForCausalLM
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-66b",
                                                    device_map=opt_device_map, torch_dtype=torch.float16)
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-66b")

    def decode_scene_graph(self, sg_attr):
        attr_list = []
        for attr in sg_attr:
            attr_list.append(attr[1])

        text = ""
        text += " ".join(attr_list)
        return text

    def sample_inference_scenegraph(self, scenegraph_path: str, img_dir: str, question: str):
        self.given_question = question
        self.img_dir = img_dir

        concept_graph = json.load(open(scenegraph_path))

        attr_list = []
        for attr_id, attr in enumerate(concept_graph):
            tmp_attr = [attr['class'], attr['caption']]
            attr_list.append(tmp_attr)

        attr_list.sort(key=lambda x: x[0], reverse=True)

        answer_list = []
        noticed_attr_list = []
        thoughts = []
        answer_text = ""

        self.current_conversation = []
        rounds = 1 if self.args.all_regional_captions else self.args.rounds
        for i in range(rounds):
            # get the most relevant object idx according to llm
            idx = self.interactive(attr_list)

            # HERE
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            noticed_attr_list.append(attr_list[idx])

            if self.args.debug:
                print(f"{time.time()}\t{inspect.currentframe().f_lineno} ==> before sample_inference ==> noticed_attr_list: {noticed_attr_list}, thoughts: {thoughts}")

            if self.chain_of_thoughts:
                answer_list.append(self.sample_inference([] if self.args.ablation_visual else noticed_attr_list
                                                         , [] if self.args.ablation_reason else thoughts))
            else:
                answer_list.append(self.sample_inference(noticed_attr_list))

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            if idx != None:
                attr_list = attr_list[:idx] + attr_list[idx + 1:]

            thoughts.append(answer_list[-1][2])
            if answer_text == answer_list[-1][0]:
                break
            else:
                answer_text = answer_list[-1][0]
        final_answer = answer_list[-1]
        return final_answer, answer_list

    def pick_example(self, key):
        img_key = int(key.split('<->')[0]) if self.args.set_name != "fvqa" else self.image_dict[key]  # for fvqa
        scene_graph_path = os.path.join(self.sg_attr_dir, str(img_key).zfill(12) + ".json")
        scene_graph_attr = json.load(open(scene_graph_path))
        for attr_id, attr in enumerate(scene_graph_attr[0]):
            if attr['class'] in ['girl', 'boy', 'man', 'woman'] and len(attr['attr']) > 0:
                description = attr['attr'][0]
                self.temp_question = f"What is the {description} {attr['class']} doing?"
                return True
        return False

    def query_blip2_basic(self, image, prompt, use_pred_answer=False):
        if not self.args.with_blip2_api:
            if use_pred_answer:
                output = self.blip2_model.predict_answers({"image": image, "text_input": prompt}, max_len=25)
            else:
                output = self.blip2_model.generate({"image": image, "text_input": prompt})
            if self.args.debug:
                import pdb
                pdb.set_trace()
        else:
            # api only support predict_answers CALL
            output = utils_api.blip_completev2(images=[image ], texts= [prompt], blip_urls=self.blip2_api, num_beams=5, length_penalty=-1.0, encoding_format="PNG",)
            return output

        return output

    def query_blip2_objects(self):
        obj_list = []
        max_obj_num = 10
        while len(obj_list) < max_obj_num:
            if len(obj_list)==0:
                tmp_obj_name_list = self.query_blip2_basic(image=self.current_blip2_image,
                                         prompt="Give me the name of one object, creature, or entity in the image.")
            else:
                tmp_prompt = "Give me the name of one object, creature, or entity in the image besides"
                for tmp_idx, tmp_name in enumerate(obj_list):
                    tmp_prompt = tmp_prompt +" %s"%tmp_name
                    if tmp_idx < len(obj_list) -1:
                        tmp_prompt +=","
                    else:
                        tmp_prompt +="?"
                    tmp_obj_name_list = self.query_blip2_basic(image=self.current_blip2_image, prompt=tmp_prompt)

            tmp_obj_name_list_refine = parge_obj_name(tmp_obj_name_list)
            print(tmp_obj_name_list_refine)

            all_exist_flag = True
            for obj_name in tmp_obj_name_list_refine:
                if obj_name not in obj_list:
                    obj_list.append(obj_name)
                    all_exist_flag = False
            if all_exist_flag:
                break
        obj_list = list(set(obj_list))
        attr_list = [[1.0, obj_name] for obj_name in obj_list] # [[confidence, name]]
        print(attr_list)
        if self.args.debug:
            pdb.set_trace()
        return attr_list

    def query_blip2_global_caption(self, question):
        global_caption = self.query_blip2_basic(image=self.current_blip2_image, prompt="An image of ")[0]
        global_caption_question = self.query_blip2_basic(image=self.current_blip2_image, prompt=f"Question: Please "
                                                   f"look at the picture and answer the following question. "
                                                   f"{question} Answer:", use_pred_answer=True)[0]
        if self.args.debug:
            print(". ".join([global_caption, global_caption_question]))
        return ". ".join([global_caption, global_caption_question])

    def query_blip2_local_caption(self, obj_name, question):
        local_caption_raw = self.query_blip2_basic(image=self.current_blip2_image,
                                                   prompt=f"Question: Look at the {obj_name} in this image. Please give a detailed "
                                                          f"description of the {obj_name} in this image. Answer:", use_pred_answer=True)[0]
        if self.args.engine == "chat":
            self.current_conversation.append({
                'role': 'user',
                'content': f'You will to look at the {obj_name} in the picture and find {local_caption_raw}.'
                           f'To find the answer to {question}, you can ask one question about the {obj_name}. '
                           f'Please tell me the question you want to ask directly.'
            })
            successful = False
            while not successful:
                try:
                    self.sleep()
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=self.current_conversation,
                        max_tokens=40,
                        temperature=0.,
                        stream=False,
                    )
                    successful = True
                except Exception as e:
                    print(e)
                    self.sleep(switch_key=True)
            question_from_chatgpt = response['choices'][0]['message']['content']
        elif self.args.engine == "chat-test":
            self.current_conversation.append({
                'role': 'user',
                'content': f'You will to look at the {obj_name} in the picture and find {local_caption_raw}.'
                           f'To find the answer to {question}, you can ask one question about the {obj_name}. '
                           f'Please tell me the question you want to ask directly.'
            })
            question_from_chatgpt = "Who are you?"
        elif self.args.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct"]:
            prompt = f"I look at the {obj_name} in the picture and find {local_caption_raw}. " \
                     f"To find the answer to {question}, I ask one question about the {obj_name}. " \
                     f"My question is:"
            successful = False
            while not successful:
                try:
                    self.sleep()
                    response = openai.Completion.create(
                        engine=self.args.engine_name,
                        prompt=prompt,
                        max_tokens=41,
                        logprobs=1,
                        temperature=0.,
                        stream=False,
                        stop=["<|endoftext|>", "?", " ?"],
                    )
                    successful = True
                except Exception as e:
                    print(e)
                    self.sleep(switch_key=True)
            question_from_chatgpt = response['choices'][0]['text'].strip()+ "?"
        else:
            question_from_chatgpt = "Empty"
        local_caption_question = self.query_blip2_basic(image=self.current_blip2_image,
                                                        prompt=f"Question: Please look at the {obj_name} and answer the following question."
                                                                             f" {question_from_chatgpt} Answer:", use_pred_answer=True)[0]
        local_caption = ". ".join([local_caption_raw, question_from_chatgpt + " The answer is " + local_caption_question])
        if self.args.debug:
            print(local_caption)
            pdb.set_trace()
        return local_caption

    def query_blip2_thought_match_img(self, thought):
        blip2_answer = self.query_blip2_basic(image=self.current_blip2_image,
                                              prompt=f"Question: Does this sentence match the facts in the picture? Please answer yes or no. "
                                                     f"Sentence: In this picture, {thought} Answer:")[0]
        if self.args.debug:
            print(blip2_answer, thought)
        if blip2_answer == "no":
            correction = self.query_blip2_basic(image=self.current_blip2_image,
                                                prompt=f"Question: Please correct the following sentence according to "
                                                       f"the image. Sentence: {thought}")[0]
            return correction
        else:
            return thought

    def interactive(self, attr_list):
        question = self.given_question

        if self.args.engine == "chat" or self.args.engine == "chat-test":
            system_prompt = "Let's play a game. I have an image and a complex question about it, and you will give me the " \
                     "name of object in the image you want to look at the most. Please follow the format" \
                     " of the following examples and give me an object name directly.\n"
            prompt = "===\n"
        else:
            prompt = 'Please select the object most related to the question.\n===\n'

        # load examples
        example_path = 'select_object_example.json'
        examples = json.load(open(example_path))
        for ni in range(self.args.n_shot):
            example = examples[ni]

            prompt += "Question: %s\n===\nOptions:" % example['question']
            for idx, option in enumerate(example['options']):
                prompt += f" {idx}: {option};"
            prompt += "\n"
            prompt += f"The index of the most related option is {example['options'].index(example['answer'])}.\n\n===\n"

        obj_list = [obj[0] for obj in attr_list]
        if self.args.engine == "chat" or self.args.engine == "chat-test":
            prompt += "Question: %s\n===\nOptions:" % question
            for idx, option in enumerate(obj_list):
                prompt += f" {idx}: {option};"
            prompt += "\n"
        else:
            ValueError("Invalid engine for interactive object selection")
        # elif self.args.use_attributes_to_see:
        #     obj_list = [f"{obj[1]}: {' '.join(obj[2])} {obj[1]}" for obj in attr_list]
        #     prompt += "Question: %s\n===\nOptions: %s\n" % (question, ",\n".join(obj_list))
        # elif self.args.use_caption_to_see:
        #     obj_list = [f"{obj[1]}: {obj[-2]}" for obj in attr_list]
        #     prompt += "Question: %s\n===\nOptions: %s\n" % (question, ", ".join(obj_list))
        # else:
        #     prompt += "Question: %s\n===\nOptions:\n" % (question)
        prompt += "The index of the most related option is"

        if self.args.debug:
            print(f"{time.time()}\t{inspect.currentframe().f_lineno} ==> Construct prompt for selecting objects ==> {prompt}")

        if self.args.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
            obj_idx_list = []
            for obj in obj_list:
                obj_idx_list.append(self.tokenizer.encode(f" {obj}")[0])
            logit_bias = {}
            current_bias = 100
            successful = False
            if self.args.engine == "codex":
                engine_name = "code-davinci-002"
            elif self.args.engine == "instruct":
                engine_name = "davinci-instruct-beta"
            elif self.args.engine == "gpt3":
                engine_name = "text-davinci-001"
            else:
                engine_name = self.args.engine
            while not successful:
                for tok_idx in obj_idx_list:
                    logit_bias[str(tok_idx)] = current_bias
                try:
                    self.sleep()
                    response = openai.Completion.create(
                        engine=engine_name,
                        prompt=prompt,
                        max_tokens=4,
                        logprobs=1,
                        temperature=0.,
                        stream=False,
                        stop=["\n", "<|endoftext|>"],
                        logit_bias=logit_bias
                    )
                    successful = True
                except Exception as e:
                    print(e)
                    self.sleep(switch_key=True)
            result = self.tokenizer.encode(response['choices'][0]['text'])[0]
            if result in obj_idx_list:
                result = obj_idx_list.index(result)
            else:
                result = 0
        elif self.args.engine == "chat":
            successful = False



            # start querying gpt
            while not successful:
                try:
                    self.sleep()
                    client = openai.OpenAI(api_key=self.apikey_list[self.apikey_idx])
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}],
                        max_tokens=5,
                        temperature=0.,
                        stream=False,
                    )
                    # response = openai.ChatCompletion.create(
                    #     model="gpt-3.5-turbo",
                    #     messages=[
                    #         {"role": "system", "content": system_prompt},
                    #         {"role": "user", "content": prompt}],
                    #     max_tokens=5,
                    #     temperature=0.,
                    #     stream=False,
                    #     # stop=["\n", "<|endoftext|>"],
                    #     logit_bias=logit_bias
                    # )
                    successful = True
                except Exception as e:
                    print(e)
                    print(prompt)
                    current_bias = int(0.8 * current_bias)
                    self.sleep(switch_key=True)
            response = response.choices[0].message.content


            # # for debug
            # response = '11.'


            # remove all the non-numeric characters
            try:
                result = ''.join(filter(str.isdigit, response))
                result = int(result)
            except:
                result = 0
                print(f'Warning: The returned object index is not a number: {response}. The selected index is set to 0.')

            if result >= len(obj_list):
                result = 0
                print(f"Warning: The selected object index is out of range. The selected index is set to 0.")

            self.current_conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]

            if self.args.debug:
                print(f"{time.time()}\t{inspect.currentframe().f_lineno} ==> ChatGPT response ==> response: {response['choices'][0]['message']['content']}, result: {result}")

        elif self.args.engine == "chat-test":
            print([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
            self.current_conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Object0"}
            ]
            pdb.set_trace()
            result = 0
        elif self.args.engine == "opt" or self.args.engine == "llama":
            obj_idx_list = []
            for obj in obj_list:
                obj_idx_list.append(self.tokenizer.encode(f" {obj}")[1])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs.input_ids.to(torch.cuda.current_device()), max_length=len(inputs.input_ids[0]) + 5,
                                          return_dict_in_generate=True, output_scores=True)
            scores = outputs['scores']
            scores = scores[0][0][obj_idx_list]
            if self.args.use_attributes_to_see or self.args.use_caption_to_see:
                result_str = self.tokenizer.decode(outputs['sequences'][0][len(inputs.input_ids[0]):]).split("\n")[0].strip()
                result_str = result_str[:-1] if result_str.endswith(".") else result_str
                result = -1
                for obj_id, obj in enumerate(obj_list):
                    if result_str in obj:
                        result = obj_id
                if result == -1:
                    result = scores.argmax().item()
            else:
                result = scores.argmax().item()
        elif self.args.engine == "bloom":
            obj_idx_list = []
            for obj in obj_list:
                obj_idx_list.append(self.tokenizer.encode(f" {obj}")[0])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs.input_ids.to(torch.cuda.current_device()),
                                          max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
            scores = outputs['scores']
            scores = scores[0][0][obj_idx_list]
            result = scores.argmax().item()
        else:
            assert False
        if self.args.debug:
            pdb.set_trace()
        return result

    def make_choices_text(self, choices, answer):
        return f"{', '.join(choices)}.", choices[answer]

    def chat_global_caption(self, scene_graph_attr):
        # TODO: ask gpt to get global caption according to objects and their descriptions in the scene.
        return 'This is a kitchen with microwave, oven, and refrigerator.'

    def sample_inference(self, scene_graph_attr, thoughts_list=None):

        question = self.given_question

        caption_i = self.chat_global_caption(scene_graph_attr)

        default_sg_text = self.decode_scene_graph(scene_graph_attr)

        if self.args.debug:
            print(f"{time.time()}\t{inspect.currentframe().f_lineno} ==> caption_i: {caption_i}, default_sg_text: {default_sg_text}")

        pred_answer_list, pred_prob_list, thought_list, all_thought_list = [], [], [], []


        for repeat in range(self.args.n_ensemble):
            if self.args.engine == "chat" or self.args.engine == "chat-test":
                prompt_before_answer = "Based on the given information, I must guess the most possible answer:"
                system_prompt = "Let's play a game. I have an image and a complex question about it. I will provide you some information about" \
                                " the image in the context, and you will give me the possible answer and reason to the question. You must provide an answer and can not say unclear or unknown. " \
                                "Please follow the format and answer style of the following examples and complete the last example.\n"
                prompt = "===\n"
            else:
                prompt_before_answer = "Answer: The answer is"
                prompt = 'Please answer the question according to the above context.\n===\n'

            ## prompt format following GPT-3 QA API
            example_path = 'openeqa_example.json'
            examples = json.load(open(example_path))
            cur_caption_i = "" if self.args.remove_caption else caption_i
            for ni in range(self.args.n_shot):
                example = examples[ni]

                prompt += 'Context: %s\n===\n' % example['context']
                prompt += 'Question: %s\n%s\nAnswer: %s\nExplanation: %s\n\n===\n' % (example['question'], prompt_before_answer, example['answer'], example['rationale'])

            if thoughts_list is not None and len(thoughts_list) > 0:
                cur_thoughts_list = [th for th in thoughts_list if th != '']
                if len(cur_thoughts_list) > 0:
                    cur_caption_i += " "
                    cur_caption_i += " ".join(cur_thoughts_list)

            choice_text = ""

            if default_sg_text == "":
                prompt += 'Context: %s\n===\n' % cur_caption_i
            else:
                prompt += 'Context: %s %s\n===\n' % (cur_caption_i, default_sg_text)

            if self.chain_of_thoughts:
                prompt += 'Question: %s\n%s\n' % (question, prompt_before_answer)
            else:
                prompt += 'Question: %s\n%s\n' % (question, prompt_before_answer)

            if self.args.debug:
                print(f"{time.time()}\t{inspect.currentframe().f_lineno} ==> Construct prompt for answering in ensemble {repeat} ==> {prompt}")

            response = None
            if self.args.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
                successful = False
                if self.args.engine == "codex":
                    engine_name = "code-davinci-002"
                elif self.args.engine == "instruct":
                    engine_name = "davinci-instruct-beta"
                elif self.args.engine == "gpt3":
                    engine_name = "text-davinci-001"
                else:
                    engine_name = self.args.engine
                while not successful:
                    try:
                        self.sleep()
                        response = openai.Completion.create(
                            engine=engine_name,
                            prompt=prompt,
                            max_tokens=41,
                            logprobs=1,
                            temperature=0.,
                            stream=False,
                            stop=["\n", "<|endoftext|>"]
                        )
                        successful = True
                    except Exception as e:
                        print(e)
                        self.sleep(switch_key=True)
                plist = []
                if self.chain_of_thoughts:
                    for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
                        if response['choices'][0]['logprobs']['tokens'][ii].endswith("."):
                            break
                        plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
                    pred_answer_list.append(process_answer(response['choices'][0]["text"].split(".")[0]))
                    thought = ".".join(response['choices'][0]["text"].split(".")[1:]).strip()
                    pred_prob_list.append(sum(plist))
                else:
                    for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
                        if response['choices'][0]['logprobs']['tokens'][ii] == '\n':
                            break
                        plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
                    pred_answer_list.append(process_answer(response['choices'][0]["text"]))
                    pred_prob_list.append(sum(plist))
            elif self.args.engine == "chat":
                successful = False
                while not successful:
                    try:
                        self.sleep()
                        client = openai.OpenAI(api_key=self.apikey_list[self.apikey_idx])
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        # response = openai.ChatCompletion.create(
                        #     model="gpt-3.5-turbo",
                        #     messages=[
                        #         {"role": "system", "content": system_prompt},
                        #         {"role": "user", "content": prompt}
                        #     ],
                        #     max_tokens=40,
                        #     temperature=0.,
                        #     stream=False,
                        # )
                        successful = True
                    except Exception as e:
                        print(e)
                        print(prompt)
                        self.sleep(switch_key=True)

                result = response.choices[0].message.content
                if self.chain_of_thoughts:
                    pred_answer_list.append(
                        result.split('Explanation:')[0].split('Answer:')[1].strip()
                    )
                    thought = result.split('Explanation:')[1].strip()
                    pred_prob_list.append(0)
                else:
                    pred_answer_list.append(
                        result.split('Explanation:')[0].split('Answer:')[1].strip()
                    )
                    pred_prob_list.append(0)
            elif self.args.engine == "chat-test":
                print([{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}])
                pdb.set_trace()
                pred_answer_list.append("fake answer")
                thought = "This is a fake thought."
                pred_prob_list.append(0)
            elif self.args.engine == "opt" or self.args.engine == "llama" or self.args.engine == "bloom":
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(inputs.input_ids.to(torch.cuda.current_device()), max_length=len(inputs.input_ids[0]) + 40,
                                                  return_dict_in_generate=True, output_scores=True)
                    plist = []
                    result = self.tokenizer.batch_decode(outputs['sequences'][:, len(inputs.input_ids[0]):])[0]
                    if self.chain_of_thoughts:
                        for ii in range(len(inputs.input_ids[0]), len(outputs['sequences'][0])):
                            tok = outputs['sequences'][0][ii]
                            if self.tokenizer.decode([tok]) == '.':
                                break
                            scores = torch.log_softmax(outputs['scores'][ii - len(inputs.input_ids[0])], dim=-1)
                            plist.append(scores[0][tok])
                        thought = ".".join(result.split("\n")[0].split("The answer is")[-1].split(".")[1:]).strip()
                        pred_answer = process_answer(result.split("\n")[0].split("The answer is")[-1].split(".")[0])
                        pred_answer_list.append(pred_answer)
                        pred_prob_list.append(sum(plist))
                    else:
                        for ii in range(len(inputs.input_ids[0]), len(outputs['sequences'][0])):
                            tok = outputs['sequences'][0][ii]
                            if self.tokenizer.decode([tok]) == '\n':
                                break
                            scores = torch.log_softmax(outputs['scores'][ii - len(inputs.input_ids[0])], dim=-1)
                            plist.append(scores[0][tok])
                        pred_answer = process_answer(result.split("\n")[0])
                        pred_answer_list.append(pred_answer)
                        pred_prob_list.append(sum(plist))
            else:
                assert False

            if self.chain_of_thoughts and self.args.with_clip_verify:
                if self.args.use_blip2:
                    tmp_thought_list = thought.split(".")
                    new_tmp_thought_list = []
                    new_tmp_thought_list_all = []
                    for thought in tmp_thought_list:
                        new_tmp_thought_list.append(self.query_blip2_thought_match_img(thought))
                        new_tmp_thought_list_all.append(thought)
                    new_thought = ".".join(new_tmp_thought_list).strip() + "."
                    new_thought_all = ".".join(new_tmp_thought_list_all).strip() + "."
                    if len(new_tmp_thought_list) > 0:
                        thought_list.append(new_thought)
                    else:
                        thought_list.append('')
                    all_thought_list.append(new_thought_all)
                else:
                    with torch.no_grad():
                        # TODO: modify it to dynamically extract features from open-eqa images
                        img_input_list = []
                        img_dir = self.img_dir
                        for img_file in os.listdir(img_dir):
                            img = Image.open(os.path.join(img_dir, img_file)).convert('RGB')
                            img_input_list.append(img)

                        text_input_list = thought.strip().split(".")
                        # remove empty strings
                        text_input_list = [text for text in text_input_list if text != '']

                        inputs = self.clip_processor(text=text_input_list, images=img_input_list, return_tensors="pt", padding=True)
                        for k, v in inputs.items():
                            inputs[k] = v.to(self.device)
                        outputs = self.clip_model(**inputs)
                        img_emb = outputs.image_embeds  # (num_images, embed_dim)
                        thought_emb = outputs.text_embeds  # (num_text, embed_dim)

                        if self.args.debug:
                            print(f"{time.time()}\t{inspect.currentframe().f_lineno} ==> img_emb: {img_emb.shape}, thought_emb: {thought_emb.shape}")
                            print(f'text_input_list: {text_input_list}')

                        img_emb /= img_emb.norm(dim=-1, keepdim=True)
                        thought_emb /= thought_emb.norm(dim=-1, keepdim=True)

                        # TODO: debug the following!!!!
                        sim_cands = img_emb @ thought_emb.T  # (num_images, num_text)
                        # find the image that maximizes the similarity
                        img_id = torch.argmax(torch.sum(sim_cands, dim=1)).item()
                        sim_cands = sim_cands[img_id:img_id + 1, :]  # (1, num_text)
                        sim_thre = self.args.verify_threshold
                        new_tmp_thought_list = []
                        new_tmp_thought_list_all = []
                        for tid in range(sim_cands.shape[1]):
                            sim = sim_cands[0, tid].item()
                            if sim > sim_thre and len(text_input_list[tid]) > 0:
                                new_tmp_thought_list.append(text_input_list[tid])
                            new_tmp_thought_list_all.append(text_input_list[tid])
                        new_thought = ".".join(new_tmp_thought_list).strip() + "."
                        new_thought_all = ".".join(new_tmp_thought_list_all).strip() + "."

                        if len(new_tmp_thought_list) > 0:
                            thought_list.append(new_thought)
                        else:
                            thought_list.append('')
                        all_thought_list.append(new_thought_all)

            elif self.chain_of_thoughts:
                if self.args.random_rationale:
                    assert False
                thought_list.append(thought)
                all_thought_list.append(new_thought)

        maxval = -999.
        for ii in range(len(pred_prob_list)):
            if pred_prob_list[ii] > maxval:
                if self.chain_of_thoughts:
                    thoughts, all_thoughts = thought_list[ii], all_thought_list[ii]
                maxval, pred_answer = pred_prob_list[ii], pred_answer_list[ii]

        if self.chain_of_thoughts:
            return [pred_answer, prompt, thoughts, all_thoughts, float(maxval),
                    [attr[0] for attr in scene_graph_attr]]
        return [pred_answer, prompt, float(maxval), [attr[0] for attr in scene_graph_attr]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey_file', type=str, default="", help='api key; https://openai.com/api/')
    parser.add_argument('--apikey', type=str, default="", help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--engine_name', type=str, default='text-davinci-003', help='api engine; https://openai.com/api/')
    parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl, vinvl_sg, vinvl_ocr')
    parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
    parser.add_argument('--n_ensemble', type=int, default=1, help="number of ensemble")
    parser.add_argument('--rounds', type=int, default=3, help="number of interactive rounds")
    parser.add_argument('--image_id', type=int, default=-1, help="selected image id pick example only")
    parser.add_argument('--iterative_strategy', type=str, default="caption", help="caption or sg")
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--valcaption_file', type=str, default='input_text/vinvl_caption/VinVL_base_val2014.tsv')
    parser.add_argument('--tag_path', type=str, default='input_text/coco_caption_pred_tags')
    parser.add_argument('--concept_caption_path', type=str, default='scene_graph_coco17_caption')
    parser.add_argument('--sg_path', type=str, default='')
    parser.add_argument('--coco_path', type=str, default='coco_annotations')
    parser.add_argument('--similarity_path', type=str, default='coco_clip_new')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--llama_path', type=str, default='/')
    parser.add_argument('--use_blip2', action='store_true')
    parser.add_argument('--choice_only', action='store_true')
    parser.add_argument('--chain_of_thoughts', action='store_true')
    parser.add_argument('--with_six_gpus', action='store_true')
    parser.add_argument('--with_one_gpu', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--random_attend', action='store_true')
    parser.add_argument('--oracle_attend', action='store_true')
    parser.add_argument('--random_caption', action='store_true')
    parser.add_argument('--remove_caption', action='store_true')
    parser.add_argument('--random_rationale', action='store_true')
    parser.add_argument('--oracle_rationale', action='store_true')
    parser.add_argument('--all_regional_captions', action='store_true')
    parser.add_argument('--use_attributes_to_see', action='store_true')
    parser.add_argument('--use_caption_to_see', action='store_true')
    parser.add_argument('--pick_example_mode', action='store_true')
    parser.add_argument('--pick_example_with_question_mode', action='store_true')
    parser.add_argument('--train_sim_metric', type=str, default='rationale')
    parser.add_argument('--train_sim_file', type=str, default='')
    parser.add_argument('--val_sim_file', type=str, default='')
    parser.add_argument('--verify_threshold', type=float, default=0.0)
    parser.add_argument('--start', type=float, default=0.0, help="start point in validation set (0.0-1.0)")
    parser.add_argument('--end', type=float, default=1.0, help="end point in validation set (0.0-1.0)")
    parser.add_argument('--with_clip_verify', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ablation_visual', action='store_true')
    parser.add_argument('--ablation_reason', action='store_true')
    parser.add_argument('--use_v100', action='store_true')
    parser.add_argument('--local_rank', required=False, type=int, help='used by dist launchers')
    parser.add_argument('--raw_image_dir', type=str, default="/path/to/your/coco")
    parser.add_argument('--with_blip2_api', action='store_true')
    parser.add_argument('--set_name', type=str, default='aokvqa')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.apikey_file != "":
        apikey_list = open(args.apikey_file).readlines()
        apikey_list = [line.strip() for line in apikey_list]
    else:
        apikey_list = [args.apikey]

    aokvqa = VisualCOT_AOKVQA(args, apikey_list)

    scenegraph_path = 'tempt_scene_graph.json'
    img_dir = 'img_dir'
    question = 'Where is the air conditioner?'

    answer, answer_list = aokvqa.sample_inference_scenegraph(scenegraph_path, img_dir, question)
    for ans in answer_list:
        print(f'Answer: {ans[0]}, Thought: {ans[2]}')



if __name__ == '__main__':
    main()

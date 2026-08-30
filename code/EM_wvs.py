import pandas as pd
import statistics
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.stats import kstest
from scipy.optimize import LinearConstraint, Bounds, minimize
import random
import json
import pickle
from openai import AzureOpenAI
import re
import os

import matplotlib.pyplot as plt
import argparse

client_4o_0513 = AzureOpenAI(  
    azure_endpoint="https://<endpoint_0513>.openai.azure.com/",
    api_key=os.getenv("AZURE_API_KEY_0513"),
    api_version="2024-12-01-preview",
)

wvs_data = pd.read_csv('../data/WVS_Cross-National_Wave_7_csv_v6_0.csv')

cols_to_keep = [f'Q{i}' for i in [8,11,17,
                                  33, 29,30,
                                  182,184,185,
                                  154,152,153,155]]
cols_to_keep = cols_to_keep + ["A_YEAR", "B_COUNTRY_ALPHA", "G_TOWNSIZE"]
wvs_data_evi = wvs_data[cols_to_keep]

wvs_data_evi.to_csv('wvs_data_evs.csv', index=False)

def calc_autonomy_index(row):
    
    index = []

    for i in [8,11,17]:
        if i == 8 or i == 11:
            if row[f"Q{i}"] == 1:
                index.append(1)
            elif row[f"Q{i}"] == 2:
                index.append(0)
            else:
                index.append(-99)
        elif i == 17:
            if row[f"Q{i}"] == 1:
                index.append(0)
            elif row[f"Q{i}"] == 2:
                index.append(1)
            else:
                index.append(-99)
        

    neg_indices = [i for i, v in enumerate(index) if v < 0]

    if len(neg_indices) == 0:
        return sum(index)/3
    elif len(neg_indices) > 1:
        return -99
    else:
        if neg_indices[0] == 0:
            return 0.102+0.379*index[1]+0.4*index[2]
        elif neg_indices[0] == 1:
            return 0.037+0.364*index[0]+0.356*index[2]
        else:
            return 0.175+0.397*index[0]+0.366*index[1]
        
def calc_equality_index(row):

    index = []

    for i in [33,29,30]:
        ans = row[f"Q{i}"]

        if ans not in [1,2,3,4,5]:
            index.append(-99)
        else:
            if i == 33:
                index.append((ans-1)/4)
            elif i == 29 or i == 30:
                index.append((ans-1)/3)
        

    neg_indices = [i for i, v in enumerate(index) if v < 0]

    if len(neg_indices) == 0:
        return sum(index)/3
    elif len(neg_indices) > 1:
        return -99
    else:
        if neg_indices[0] == 0:
            return 0.042+0.485*index[1]+0.421*index[2]
        elif neg_indices[0] == 1:
            return 0.049+0.404*index[0]+0.447*index[2]
        else:
            return 0.145+0.443*index[1]+0.372*index[0]

def calc_choice_index(row):

    index = []

    for i in [182,184,185]:
        ans = row[f"Q{i}"]

        if ans < 0:
            index.append(-99)
        else:
            index.append((ans-1)/9)
        

    neg_indices = [i for i, v in enumerate(index) if v < 0]

    if len(neg_indices) == 0:
        return sum(index)/3
    elif len(neg_indices) > 1:
        return -99
    else:
        if neg_indices[0] == 0:
            return 0.008+0.434*index[1]+0.439*index[2]
        elif neg_indices[0] == 1:
            return 0.015+0.408*index[0]+0.496*index[2]
        else:
            return 0.069+0.416*index[0]+0.505*index[1]
            
def calc_voice_index(row):

    e003 = row["Q154"]
    e004 = row["Q155"]
        
    if (e003 == 2 and e004 == 4) or (e003 == 4 and e004 == 2):
        i_voice1 = 1
    elif (e003 == 2 and e004 != 4) or (e003 == 4 and e004 != 2):
        i_voice1 = 0.66
    elif (e003 != 2 and e004 == 4) or (e003 != 4 and e004 == 2):
        i_voice1 = 0.33
    elif e003 > -1 and e003 > -1:
        i_voice1 = 0
    else:
        i_voice1 = -99
    

    e001 = row["Q152"]
    e002 = row["Q153"]
        
    if e001 == 3 and e002 != 3:
        i_voice2 = 1
    elif e002 == 3 and e001 != 3:
        i_voice2 = 0.5
    elif (e001>-1) and (e002>-1):
        i_voice2 = 0
    else:
        i_voice2 = -99
        
    if (i_voice1 > -99) and (i_voice2 > -99):
        i_voice2_00 = (i_voice1+i_voice2)/2
    else:
        i_voice2_00 = -99
    
    if i_voice2_00 > -99:
        return i_voice2_00
    elif i_voice1 > -99:
        return 0.656*i_voice1 + 0.136
    elif i_voice2 > -99:
        return 0.613*i_voice2 + 0.141
    else:
        return -99
    

def calc_index(row):
    AUTONOMY = row["Inde_authonomy"]
    EQUALITY = row["Inde_equality"]
    CHOICE   = row["Inde_choice"]
    VOICE    = row["Inde_voice"]

    if (AUTONOMY != -99) and (EQUALITY != -99) and (CHOICE != -99) and (VOICE != -99):
        return (AUTONOMY + EQUALITY + CHOICE + VOICE) / 4
    elif (AUTONOMY != -99) and (EQUALITY == -99) and (CHOICE != -99) and (VOICE != -99):
        return 0.103 + 0.266*AUTONOMY + 0.305*CHOICE + 0.286*VOICE
    elif (AUTONOMY == -99) and (EQUALITY != -99) and (CHOICE != -99) and (VOICE != -99):
        return 0.070 + 0.274*EQUALITY + 0.304*CHOICE + 0.271*VOICE
    elif (AUTONOMY != -99) and (EQUALITY != -99) and (CHOICE == -99) and (VOICE != -99):
        return 0.016 + 0.291*AUTONOMY + 0.310*EQUALITY + 0.288*VOICE
    elif (AUTONOMY != -99) and (EQUALITY != -99) and (CHOICE != -99) and (VOICE == -99):
        return 0.051 + 0.267*AUTONOMY + 0.292*EQUALITY + 0.290*CHOICE
    else:
        return -99

wvs_data_evi["Inde_authonomy"] = wvs_data_evi.apply(calc_autonomy_index, axis=1)
wvs_data_evi["Inde_equality"] = wvs_data_evi.apply(calc_equality_index, axis=1)
wvs_data_evi["Inde_choice"] = wvs_data_evi.apply(calc_choice_index, axis=1)
wvs_data_evi["Inde_voice"] = wvs_data_evi.apply(calc_voice_index, axis=1)

wvs_data_evi["evi_index"] = wvs_data_evi.apply(calc_index, axis=1)


wvs_data_evi["score_scaled"] = (wvs_data_evi["evi_index"].round(2) * 100).astype(int)
target_data = wvs_data_evi[wvs_data_evi["evi_index"]>=0]["score_scaled"].tolist()

wvs_questions = {
    "Qautonomy": """Here is a list of qualities that children can be encouraged to learn at home. Which, if any, do you consider to be especially important? Please choose up to five! 
    
List of qualities:
1=Good manners
2=Independence
3=Hard work
4=Feeling of responsibility
5=Imagination
6=Tolerance and respect for other people
7=Thrift, saving money and things
8=Determination, perseverance
9=Religious faith
10=Not being selfish (unselfishness)
11=Obedience

Please select how the statement describes you and highlight your answer in [] (such as [1, 2, 3, 4, 5], [2, 6, 7, 10, 11], [3, 2, 7, 9, 1], or [5, 11, 6, 4, 3]).""",

    "Q33": """How would you feel about the following statements? Do you agree or disagree with them? 
Please rate each statement on a five-point scale (1 = Agree strongly, 2 = Agree, 3 = Neither agree nor disagree, 4 = Disagree, 5 = Strongly disagree).Select the number that best describes you and highlight it in brackets, e.g. [1], [2], [3], [4], or [5]. When jobs are scarce, men should have more right to a job than women. """,

    "Q29": """How would you feel about the following statements? Do you agree or disagree with them? 
Please rate each statement on a five-point scale (1 = Strongly agree, 2 = Agree, 3 = Disagree, 4 = Disagree strongly). Select the number that best describes you and highlight it in brackets, e.g. [1], [2], [3], or [4]. On the whole, men make better political leaders than women do. """,

    "Q30": """How would you feel about the following statements? Do you agree or disagree with them? 
Please rate each statement on a five-point scale (1 = Strongly agree, 2 = Agree, 3 = Disagree, 4 = Disagree strongly). Select the number that best describes you and highlight it in brackets, e.g. [1], [2], [3], or [4]. A university education is more important for a boy than for a girl """,

    "Q182": """Please tell me for each of the following actions whether you think it can always be justified, never be justified, or something in between.
Please rate each statement on a ten-point scale (1 = Never justifiable, 10 = Always justifiable). Select the number that best describes you and highlight it in brackets, e.g. [1], [4], [6], or [10]. Homosexuality""",

    "Q184": """Please tell me for each of the following actions whether you think it can always be justified, never be justified, or something in between.
Please rate each statement on a ten-point scale (1 = Never justifiable, 10 = Always justifiable). Select the number that best describes you and highlight it in brackets, e.g. [1], [4], [6], or [10]. Abortion""",

    "Q185": """Please tell me for each of the following actions whether you think it can always be justified, never be justified, or something in between.
Please rate each statement on a ten-point scale (1 = Never justifiable, 10 = Always justifiable). Select the number that best describes you and highlight it in brackets, e.g. [1], [4], [6], or [10]. Divorce""",

    "Q152_3": """Which of the following do you consider the most important and the next most important for your country?
Please select one option as your first choice and one option as your second choice, and provide your answer in a single bracket in the format [first choice, second choice].

Options:
1. A high level of economic growth
2. Making sure this country has strong defense forces
3. Seeing that people have more say about how things are done at their jobs and in their communities
4. Trying to make our cities and countryside more beautiful

Example answer: [1, 3]""",


    "Q154_5": """Which of the following do you consider the most important and the next most important for your country?
 Please select one option as your first choice and one option as your second choice, and provide your answer in a single bracket in the format [first choice, second choice].

Options:
1. Maintaining order in the nation
2. Giving people more say in important government decisions
3. Fighting rising prices
4. Protecting freedom of speech

Example answer: [1, 4]""",

}

def play_single_autonomy(
        system_prompt = "Image your are a human.",
        n_choices=1
    ):

    def extract_numbers_from_brackets(s):

        if s.count('[') != 1 or s.count(']') != 1:
            return None
        
        match = re.search(r'\[(.*?)\]', s)
        if not match:
            return None
        
        content = match.group(1)
        numbers = [int(x) for x in re.findall(r'\d+', content)]
        
        return numbers if numbers else None
    
    
    choices_all = [[],[],[]]
    completions_lst = []

    game_instruction_message = wvs_questions["Qautonomy"]
    
    try_count = 20
    while len(choices_all[0]) < n_choices and try_count > 0:
        try_count -= 1
        completion = client_4o_0513.chat.completions.create(
            model='gpt-4o-2024-05-13',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": game_instruction_message}
            ],
            n=n_choices-len(choices_all[0]),
        )

        completion_tokens = completion.usage.prompt_tokens
        prompt_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

        print("Completion tokens:", completion_tokens)
        print("Prompt tokens:", prompt_tokens)
        print("Total tokens:", total_tokens) 

        completions_lst.append(completion.to_dict)

        for s in [choice.message.content for choice in completion.choices]:
            # print(s)
            choice_list = extract_numbers_from_brackets(s)
            if choice_list is not None:
                if len(choice_list) <= 5 and len(choice_list) > 0:
                    for index, element in enumerate([2, 5, 11]):
                        if element in choice_list:
                            choices_all[index].append(1)
                        else:
                            choices_all[index].append(2)

    for index, lst in enumerate(choices_all):
        if len(lst) >= n_choices:
            choices_all[index] = lst[:n_choices]
        else:
            choices_all[index] = choices_all[index] + [-2]*(n_choices - len(lst))
            random.shuffle(choices_all[index])
            # print("autonomy_aaa")


    return {"Q8":choices_all[0],"Q11":choices_all[1],"Q17":choices_all[2]}, completions_lst

def play_single_equality_choice(
        game,
        system_prompt = "Image your are a human.",
        n_choices=1
    ):

    def extract_numbers_from_brackets(s):
        if s.count('[') > 1 or s.count(']') > 1:
            return None
        
        match = re.search(r'\[(\d+)\]', s)
        return int(match.group(1)) if match else None
    
    game_questions = {
        "equality": ["Q33", "Q29", "Q30"],
        "choice": ["Q182", "Q184", "Q185"],
    }

    # 初始化 choices_all
    if game in game_questions:
        choices_all = {q: [] for q in game_questions[game]}
    else:
        print("Invalid game type. Choose 'equality' or 'choice'.")
        return None, None
    
    completions_lst = []
    
    for index, question_index in enumerate(choices_all.keys()):
        try_count = 20
        # print(question_index)
        while len(choices_all[question_index]) < n_choices and try_count > 0:
            try_count -= 1
            # print(try_count)
            completion = client_4o_0513.chat.completions.create(
                model='gpt-4o-2024-05-13',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": wvs_questions[question_index]}
                ],
                n=n_choices-len(choices_all[question_index]),
            )
            
            completions_lst.append(completion.to_dict)

            for s in [choice.message.content for choice in completion.choices]:
                # print(s)
                choice = extract_numbers_from_brackets(s)
                if choice != None:
                    choices_all[question_index].append(choice)

        if len(choices_all[question_index]) >= n_choices:
            choices_all[question_index] = choices_all[question_index][:n_choices]
        else:
            choices_all[question_index] = choices_all[question_index] + [-1]*(n_choices - len(choices_all[question_index]))
            random.shuffle(choices_all[question_index])
            
            # print(game+"_aaa")
    
    return choices_all, completions_lst

def play_single_voice(
        system_prompt = "Image your are a human.",
        n_choices=1
    ): 

    def extract_numbers_from_brackets(s):

        if s.count('[') != 1 or s.count(']') != 1:
            return None
        
        match = re.search(r'\[(.*?)\]', s)
        if not match:
            return None
        
        content = match.group(1)
        numbers = [int(x) for x in re.findall(r'\d+', content)]
        
        return numbers if numbers else None
    
    
    choices_all = [[],[],[],[]]
    
    questions_lst = ["Q152_3", "Q154_5"]
    
    completions_lst = []

    for index, question in enumerate(questions_lst):
        try_count = 20
        while len(choices_all[index*2]) < n_choices and try_count > 0:
            try_count -= 1
            completion = client_4o_0513.chat.completions.create(
                model='gpt-4o-2024-05-13',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": wvs_questions[question]}
                ],
                n=n_choices-len(choices_all[index*2]),
            )

            completions_lst.append(completion.to_dict)

            for s in [choice.message.content for choice in completion.choices]:
                # print(s)
                # print("---")
                choice_list = extract_numbers_from_brackets(s)
                
                if choice_list!=None and len(choice_list) == 2:
                    choices_all[index*2].append(choice_list[0])
                    choices_all[index*2+1].append(choice_list[1])

        for i in range(2):
            if len(choices_all[index*2 + i]) >= n_choices:
                choices_all[index*2 + i] = choices_all[index*2 + i][:n_choices]
            else:
                choices_all[index*2 + i] = choices_all[index*2 + i] + [-1]*(n_choices - len(choices_all[index*2 + i]))
                random.shuffle(choices_all[index*2 + i][:n_choices])
                # print("voice_aaa")
                    

    return {"Q152":choices_all[0],"Q153":choices_all[1],"Q154":choices_all[2],"Q155":choices_all[3]}, completions_lst

def play_dimension(system_prompt = "Image your are a human.",
                   n_choices = 10):
    
    choices_autonomy, completion_au = play_single_autonomy(system_prompt, n_choices=n_choices)
    choices_equality, completion_eq = play_single_equality_choice("equality", system_prompt, n_choices=n_choices)
    choices_choice, completion_ch = play_single_equality_choice("choice", system_prompt, n_choices=n_choices)
    choices_voice, completion_vo = play_single_voice(system_prompt, n_choices=n_choices)

    completions_all = completion_au + completion_eq + completion_ch + completion_vo

    choices_all = {**choices_autonomy, **choices_equality, **choices_choice, **choices_voice}

    # print(choices_all)

    choices_all_df = pd.DataFrame(choices_all)
    # print(choices_all_df)

    choices_all_df["Inde_authonomy"] = choices_all_df.apply(calc_autonomy_index, axis=1)
    choices_all_df["Inde_equality"] = choices_all_df.apply(calc_equality_index, axis=1)
    choices_all_df["Inde_choice"] = choices_all_df.apply(calc_choice_index, axis=1)
    choices_all_df["Inde_voice"] = choices_all_df.apply(calc_voice_index, axis=1)


    choices_all_df["evi_index"] = choices_all_df.apply(calc_index, axis=1)
    choices_all_df["score"] = (choices_all_df["evi_index"].round(2) * 100).astype(int)

    socre_lst = choices_all_df["score"].tolist()
    
    return socre_lst, completions_all, choices_all_df

requirements = '''
## Crafting Requirements

* The ultimate goal is to prompt the chatbot to behave towards the desired behavior under the given scenario, rather than being a precoded helpful assistant chatbot. 
* Craft the system prompt based on understanding the meaning of the desired behavior under the given instruction.
* Do not include information already included in the game instruction. 
* Craft a generalizable system prompt and avoid including any information specific to this particular instruction or directly implying the desired behavior. 
'''

output_format = '''
## Output Format

Directly output the crafted system prompt starting with "You are ...".
'''

def get_crafting_prompt(
    desired_behavior,
    requirements=requirements,
    ):

    return f'''
            ## Goal

            Imagining a chatbot is a human and doing in a cultural values survey, specifically focusing on Emancipative Values. These values emphasize individual freedom, personal autonomy, equality of opportunity, and people’s right to participate in decision-making. Your goal is to craft a system prompt for this chatbot, so that with the crafted system prompt, the chatbot behaves in a certain way. 

            ## Game Instruction

            The chatbot will answer 9 questions regarding to Emancipative Values.
            
            ## Desired Behavior

            For this particular task, please generate a system prompt for the chatbot. With the generated system prompt and the above survey instruction provided, the chatbot’s total score on Emancipative Values should be: {desired_behavior} out of 100.

            {requirements}
            {output_format}
            '''

def get_improving_prompt(
        sampled_behavior,
        desired_behavior,
        requirements=requirements,
    ):
        return f'''
                Using your crafted system prompt, a chatbot scores mostly {sampled_behavior} instead of {desired_behavior}. Do you have any idea how to improve the system prompt?

                {requirements}
                {output_format}
                '''

def get_binned_mode(values, bin_size=20):
    if len(values) == 0:
        return None
    
    bins = np.arange(0, 101, bin_size)  # 0-100
    hist, edges = np.histogram(values, bins=bins)

    max_bin_idx = np.argmax(hist)
    bin_center = (edges[max_bin_idx] + edges[max_bin_idx+1]) / 2

    return int(bin_center)


def get_bin(choice_lst, bin_size=20):

    value = get_binned_mode(choice_lst, bin_size=bin_size)
    return int(value // bin_size)


def craft_system_prompt(
    desired_behavior,
    n_sample_per_learner=10,
    n_improvement=3,
    bin_size=20
):  
    prompts = []
    choices = []
    desired = []
    last_modes = []

    initial_prompt = get_crafting_prompt(desired_behavior)
    messages = [
        {"role": "user", "content": initial_prompt}
    ]

    completions_lst = []

    def craft():
        completion = client_4o_0513.chat.completions.create(
                model='gpt-4o-2024-05-13',
                messages=messages,
                n=1
            )


        prompt = completion.choices[0].message.content

        play_dimension_result = play_dimension(
            system_prompt=prompt, 
            n_choices=n_sample_per_learner
        )
        choice = play_dimension_result[0]

        completions_lst.append(play_dimension_result[1])

        # Get bin-based mode
        last_mode = get_binned_mode(choice, bin_size=bin_size)
        last_modes.append(last_mode)

        if np.std(choice) < 10: 
            prompts.append(prompt)
            choices.append(choice)
            desired.append(desired_behavior)
        
        messages.append({
            "role": "assistant", 
            "content": completion.choices[0].message.content
        })
    
    craft()

    for _ in range(n_improvement):
        last_mode = last_modes[-1]

        if last_mode == get_binned_mode([desired_behavior], bin_size):
            break

        improve_prompt = get_improving_prompt(last_mode, desired_behavior)
        messages.append({"role": "user", "content": improve_prompt})
        craft()

    return prompts, choices, desired, completions_lst


def samples_to_dist(
    samples, 
    weights=None,
):
    if weights is not None: 
        assert len(samples) == len(weights)
    if isinstance(samples[0], list):
        if weights is not None:
            weights = np.array(weights)[:, np.newaxis].repeat(len(samples[0]), axis=1)
            weights = weights.flatten()
        samples = np.array(samples).flatten()
    dist =  np.histogram(
        samples, 
        weights=weights, 
        bins=101, 
        range=(0, 100), 
        density=True
    )[0]
    dist = dist / np.sum(dist)
    return dist

def initialization(
    num_test,
    K,
    bin_size = 20
):
    
    prompts_lst = []
    choices_lst = []
    disired_behaviors_lst = []

    target_dist = samples_to_dist(target_data)

    sampled_desired_behavior = np.random.choice(
                    list(range(len(target_dist))), 
                    size=K, 
                    p=target_dist/np.sum(target_dist),
                    replace=False
                ).tolist()
    
    complementions_lst_all = []

    # sampled_desired_behavior = random.sample(range(0, gamerange[game]), K)
    for disired_behavior in tqdm(sampled_desired_behavior):
        prompts, choices, desired, complementions_lst = craft_system_prompt(
            disired_behavior,
            bin_size=bin_size
        )
        
        complementions_lst_all = complementions_lst_all + complementions_lst


        if len(choices) == 0:
            print("if len(choices) == 0:")
            continue

        prompts_lst.append(prompts[-1])
        choices_lst.append(choices[-1])
        disired_behaviors_lst.append(desired[-1])

    ## save initialization results to a dataframe 
    df = pd.DataFrame({
        'prompt': prompts_lst, # list of all prompts
        'choices': choices_lst, # choices for each prompts
        'desired_behavior': disired_behaviors_lst # the desired behavior used to craft this prompt
    })

    df_unique_rows = df.drop_duplicates(subset=['choices']) 
    # drop duplicates based on choices, keep prompts with different choices, 81 prompts left

    df_unique_rows.to_csv(str(num_test)+"_result/EM_initialization_prompts.csv")
    
    return df_unique_rows, complementions_lst_all

# Method 3: softmax then normalize w distance and soft assign the data points to the prompt
def data_allocation_1(
        num_test,
        num_iter,
        system_prompt_df,
        weights = None
    ):

    if weights is None:
        weights = [1/len(system_prompt_df)]*len(system_prompt_df)

    weights = [1 / (w if w != 0 else 1e-6) for w in weights]


    # get the probability of each choice to have the specific data point, and get the allocation for each data point
    cluster_allocation = {}

    system_prompt_probability_df = system_prompt_df.copy()
    system_prompt_probability_df['choices_mode'] = system_prompt_probability_df['choices'].map(lambda x: statistics.mode(x))
    
    # calculate the w distance between the choices list and each single data point
    new_columns = {
        i: system_prompt_probability_df['choices'].map(lambda x: wasserstein_distance(x, [i]))
        for i in range(101)
    } 

    def softmax(column):
        exp_values = np.exp(column - np.max(column))  
        return exp_values / np.sum(exp_values)
    
    for i in range(101):
        new_columns[i] = softmax(new_columns[i])

    new_columns =  pd.DataFrame(new_columns)
    new_columns_list = new_columns.values.tolist()
    result = []
    for i, row in enumerate(new_columns_list):
        weight = weights[i] #if i < len(weights) else 1  
        result.append([value * weight for value in row])

    new_columns = pd.DataFrame(result, index=new_columns.index, columns=new_columns.columns)
    system_prompt_probability_df = pd.concat([system_prompt_probability_df, new_columns], axis=1)

    for i in range(101):
        if system_prompt_probability_df[i].sum() != 0:
            system_prompt_probability_df[i] = system_prompt_probability_df[i] / system_prompt_probability_df[i].sum()
        # else:
        #     system_prompt_probability_df[i] = [1/len(system_prompt_probability_df)]*len(system_prompt_probability_df)

    # assign the prompt index to the data point with the smallest w distance
    prompt_lst_index = system_prompt_probability_df.index.tolist()
    for data_point in set(target_data):
        all_probability = system_prompt_probability_df[data_point].tolist()
        all_probability = [1 / (w if w != 0 else 1e-6) for w in all_probability]
        closest_index = random.choices(prompt_lst_index, weights=all_probability, k=1)[0]
        cluster_allocation[int(data_point)] = closest_index

    # get the target distribution for each prompt based on its allocation
    prompt_target_dist_dict = {}

    for key, values in cluster_allocation.items():
        # key: data point
        # values: prompt index
        if values not in prompt_target_dist_dict.keys():
            prompt_target_dist_dict[values] = []

        prompt_target_dist_dict[values] = prompt_target_dist_dict[values] + [key]*list(target_data).count(key)

    system_prompt_probability_df.to_csv(str(num_test)+"_result/"+str(num_iter)+'_system_prompt_probability_df.csv')

    return cluster_allocation, prompt_target_dist_dict

def latentvariabel_update_system_prompts(
    system_prompt_df,
    prompt_target_dist_dict,
    binsize=20
):
    
    system_prompt_df_tmp = system_prompt_df.copy()

    complementions_lst_all = []

    for key, value in tqdm(prompt_target_dist_dict.items()):

        # get the base prompt and its choices list
        base_prompt_choice_lst = system_prompt_df_tmp['choices'].loc[key]

        # get the target distribtion based on the allocation in the E step
        target_dist = samples_to_dist(prompt_target_dist_dict[key])
        # the generated distribution is from the choices of a prompt
        generate_dist = samples_to_dist(base_prompt_choice_lst)
        
        choice_mode = statistics.mode(base_prompt_choice_lst)
        desired_behavior = statistics.mode(prompt_target_dist_dict[key])


        if choice_mode == desired_behavior:
            continue
        else:
            updated_prompts, update_prompt_choices, desired, complementions_lst = craft_system_prompt(
                desired_behavior,
                bin_size=binsize,
            )

            complementions_lst_all = complementions_lst_all + complementions_lst

        if len(update_prompt_choices) == 0:
            continue

        w_distance_origin = wasserstein_distance(
                u_values=list(range(len(target_dist))), 
                v_values=list(range(len(generate_dist))),
                u_weights=target_dist,
                v_weights=generate_dist
            )
        
        update_system_choice_dist = samples_to_dist(update_prompt_choices[-1])
        
        w_distance_updated = wasserstein_distance(
                u_values=list(range(len(target_dist))), 
                v_values=list(range(len(update_system_choice_dist))),
                u_weights=target_dist,
                v_weights=update_system_choice_dist
            )
        
        system_prompt_df_tmp.at[key, 'desired_behavior'] = desired_behavior

        if w_distance_updated >= w_distance_origin:
            continue
        else:
            
            system_prompt_df_tmp.at[key, 'prompt'] = updated_prompts[-1]
            system_prompt_df_tmp.at[key, 'choices'] = update_prompt_choices[-1]

    return system_prompt_df_tmp, complementions_lst_all

    


def weight_optimization(
    # target_dist, # [samples]
    choices, # [[choices], [], ...]
    reg=3,
    n_rounds=10,
):
    target_dist = samples_to_dist(target_data)

    def expand_lists_by_repeating(lists):
        max_length = max(len(sublist) for sublist in lists)
        
        expanded_lists = [
            (sublist * (max_length // len(sublist) + 1))[:max_length]
            for sublist in lists
        ]
        return expanded_lists
    
    choices = expand_lists_by_repeating(choices)

    def loss(
        weights,
        choices=choices,
        target_dist=target_dist,
    ):
        generated_dist = samples_to_dist(choices, weights)
        w = wasserstein_distance(
            u_values=list(range(len(target_dist))), 
            v_values=list(range(len(generated_dist))),
            u_weights=target_dist,
            v_weights=generated_dist
        )
        
        generated = np.random.choice(
            a=list(range(len(generated_dist))), 
            p=generated_dist,
            size=1000, 
        )
        k = kstest(target_data, generated).pvalue
        return w * (.1 if k > 0.05 else 1)
    
    def regularization(weights):
        return reg * np.linalg.norm(weights)
    
    K = len(choices)
    bounds =  Bounds([0.]*K, [1.]*K)
    lc = LinearConstraint([[1.]*K], 1, 1)

    for _ in tqdm(range(n_rounds)):
        x0 = np.random.rand(K)
        x0 = x0 / np.sum(x0)
        result = minimize(
            fun=lambda x: loss(x) + regularization(x), 
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=lc,
            tol=1e-6
        )
        if result.fun != np.inf and result.fun > 0.5:
            weights = result.x
            return weights, loss(weights)
    assert False

def em_play(
    num_test,
    K = 20, # number of system prompts to generate
    numIter = 5,
    binsize = 20
):
    # The first step is get the initial system prompt
    print("----------Initilization Begin----------")
    df_unique_rows, complementions_lst_all = initialization(num_test, K = K, bin_size=binsize)
    print("----------Initilization End----------")

    loss = np.inf
    weights1 = None
    weights_lst = []


    complementions_lst_all_return = []

    complementions_lst_all_return = complementions_lst_all_return + complementions_lst_all

    for num_iter in range(numIter):
        # E-step: allocate each data point in the target distribution to the closest system prompt
        cluster_allocation, prompt_target_dist_dict = data_allocation_1(num_test,
                                                                        num_iter,
                                                                        df_unique_rows, 
                                                                        weights = weights1)
        
        # I want to show how different data points have been allocated to different system prompts
        
        def convert_numpy(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        def convert_keys_and_values(data):
            if isinstance(data, dict):
                return {
                    str(key) if isinstance(key, (np.int64, np.int32, np.float64, np.float32)) else key: convert_keys_and_values(value)
                    for key, value in data.items()
                }
            elif isinstance(data, list):
                return [convert_keys_and_values(item) for item in data]
            elif isinstance(data, (np.int64, np.int32)):
                return int(data)
            elif isinstance(data, (np.float64, np.float32)):
                return float(data)
            else:
                return data
        
        with open(str(num_test)+"_result/"+str(num_iter)+"_cluster_allocation.json", "w") as json_file:
            json.dump(cluster_allocation, json_file, indent=4, default=convert_numpy)  
        
        with open(str(num_test)+"_result/"+str(num_iter)+"_prompt_target_dist_dict.json", "w") as json_file:
            json.dump(convert_keys_and_values(prompt_target_dist_dict), json_file, indent=4, default=convert_numpy)  
        
        # M-step: 
        # update the system prompt
        df_unique_rows_update, complementions_lst_all = latentvariabel_update_system_prompts(
                                        df_unique_rows,
                                        prompt_target_dist_dict,
                                        binsize=binsize
                                    )
        
        complementions_lst_all_return = complementions_lst_all_return + complementions_lst_all
        # print(df_unique_rows)

        df_unique_rows_update.to_csv(str(num_test)+"_result/"+str(num_iter)+'_EM_initialization_prompts_updated.csv')

        # update weights of the system prompts
        pool_choices = df_unique_rows_update['choices'].tolist()
        print(f"Iteration {num_iter}: Optimizing weights...")
        weights1, loss_update = weight_optimization(pool_choices)
        weights_lst.append(weights1)
        print(f'Loss: {loss_update}')

        # Check for convergence
        if df_unique_rows.equals(df_unique_rows_update):
            print(f"Iteration {num_iter}: No change in system prompts, stopping early.")

            break
        
        df_unique_rows = df_unique_rows_update

        # Update parameters only if the loss improves
        if loss_update < loss:
            loss = loss_update
        else:
            print(f"Iteration {num_iter}: Loss did not improve, continuing.")

    with open(str(num_test)+'_weights_lst.pkl', 'wb') as f:
        pickle.dump(weights_lst, f)

    with open("complementation_lst.pkl", "wb") as f:
        pickle.dump(complementions_lst_all_return, f)

def main(args):
    for i in range(5): 
        print("---------Run: "+str(i+1)+" Begin---------")
        os.makedirs(str(i+1)+"_result", exist_ok=True)
        em_play(i+1, numIter=args.numIter, K=args.K, binsize=args.binsize)
        print("---------Run: "+str(i+1)+" End---------")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is for EM formalization Experiments")
    parser.add_argument("--numIter", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--binsize", type=int)
    
    args = parser.parse_args()
    main(args)


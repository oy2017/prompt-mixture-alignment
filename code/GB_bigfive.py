import pandas as pd
import statistics
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import kstest
from scipy.optimize import Bounds, minimize
from openai import AzureOpenAI  
import argparse
import ast
import os
import re

client_0513 = AzureOpenAI(
    azure_endpoint="https://<endpoint_0513>.openai.azure.com/",
    api_key=os.getenv("AZURE_API_KEY_0513"),
    api_version="2024-05-01-preview",
)

client_0718 = AzureOpenAI(
    azure_endpoint="https://<endpoint_0718>.openai.azure.com/",
    api_key=os.getenv("AZURE_API_KEY_0718"),
    api_version="2024-05-01-preview",
)

df_all = pd.read_csv('../data/data.csv', sep="\t")
df_all[df_all.columns.difference(["country"])] = df_all[df_all.columns.difference(["country"])].map(int)

changed_qs = ['E2', 'E4', 'E6', 'E8', 'E10', 
              'N2', 'N4', 
              'A1', 'A3', 'A5', 'A7', 
              'C2', 'C4', 'C6', 'C8', 
              'O2', 'O4', 'O6']

df_all_clean = df_all[df_all["E1"] != 0]


def custom_sum(row, dimension):
    result = 0
    for col in df_all_clean.columns:
        if col.startswith(dimension):
            if col in changed_qs:
                result += abs(6 - row[col])
            else:
                result += row[col]
    return result

df_all_with_score = df_all_clean.copy()
df_all_with_score.loc[:, 'score_O'] = df_all_clean.apply(lambda row: custom_sum(row, "O"), axis=1)
df_all_with_score.loc[:, 'score_C'] = df_all_clean.apply(lambda row: custom_sum(row, "C"), axis=1)
df_all_with_score.loc[:, 'score_E'] = df_all_clean.apply(lambda row: custom_sum(row, "E"), axis=1)
df_all_with_score.loc[:, 'score_A'] = df_all_clean.apply(lambda row: custom_sum(row, "A"), axis=1)
df_all_with_score.loc[:, 'score_N'] = df_all_clean.apply(lambda row: custom_sum(row, "N"), axis=1)

df_all_with_score.to_csv('df_all_with_score.csv', index=False)

game_instruction = "The following item was rated on a five-point scale where 1=Disagree, 2=Partially Disagree, 3=Neural, 4=Partially Agree, 5=Agree. Please select how the statement describes you and highlight your answer in [] (such as [1], [2], [3], [4], or [5]). "


ocean_questions = {
                'E1':  'I am the life of the party.', 
                'E2':  'I don\'t talk a lot.', 
                'E3':  'I feel comfortable around people.', 
                'E4':  'I keep in the background.', 
                'E5':  'I start conversations.',
                'E6':  'I have little to say.', 
                'E7':  'I talk to a lot of different people at parties.', 
                'E8':  'I don\'t like to draw attention to myself.',
                'E9':  'I don\'t mind being the center of attention.',
                'E10': 'I am quiet around strangers.',

                'N1':  'I get stressed out easily.',  # nervous
                'N2':  'I am relaxed most of the time.',  # confident
                'N3':  'I worry about things.',   # nervous
                'N4':  'I seldom feel blue.',   # confident
                'N5':  'I am easily disturbed.',   # nervous
                'N6':  'I get upset easily.',    # nervous
                'N7':  'I change my mood a lot.',    # nervous
                'N8':  'I have frequent mood swings.',   # nervous
                'N9':  'I get irritated easily',   # nervous
                'N10': 'I often feel blue.',

                'A1':  'I feel little concern for others.',  # detached
                'A2':  'I am interested in people.',   # friendly
                'A3':  'I insult people.',   # detached
                'A4':  'I sympathize with others feelings.',    # friendly
                'A5':  'I am not interested in other people problems.',   # detached
                'A6':  'I have a soft heart',     # friendly
                'A7':  'I am not really interested in others.',   # detached
                'A8':  'I take time out for others',    # friendly
                'A9':  'I feel others emotions',    # friendly
                'A10': 'I make people feel at ease.',

                'C1': 'I am always prepared.',  # organized
                'C2': 'I leave my belongings around.',  # careless
                'C3': 'I pay attention to details', # organized
                'C4': 'I make a mess of things.',   # careless
                'C5': 'I get chores done right away.',  # organized
                'C6': 'I often forget to put things back in their proper place.',   # careless
                'C7': 'I like order.',  # organized
                'C8': 'I shirk my duties.',  # careless
                'C9': 'I follow a schedule.', # organized
                'C10': 'I am exacting in my work.',

                'O1':  'I have a rich vocabulary.',  # inventive
                'O2':  'I have difficulty understanding abstract ideas.',  # cautious/consistent
                'O3':  'I have a vivid imagination.',   # inventive
                'O4':  'I am not interested in abstract ideas.',   # cautious/consistent
                'O5':  'I have excellent ideas',  # inventive
                'O6':  'I do not have a good imagination.',   # cautious/consistent
                'O7':  'I am quick to understand things.',   # inventive
                'O8':  'I use difficult words.',    # inventive
                'O9':  'I spend time reflecting on things.',    # inventive
                'O10': 'I am full of ideas.'
            }

d_convert = {
          "O": "openness to experience",
          "C": "conscientiousness",
          "E": "extraversion",
          "A": "agreeableness",
          "N": "neuroticism",
    }

requirements = '''
## Crafting Requirements

* The ultimate goal is to prompt the chatbot to behave towards the desired behavior under the given scenario, rather than being a precoded helpful assistant chatbot. 
* Craft the system prompt based on understanding the meaning of the desired behavior under the given question.
* Do not include information already included in the game instruction. 
* Craft a generalizable system prompt and avoid including any information specific to this particular question or directly implying the desired behavior. 
'''

output_format = '''
## Output Format

Directly output the crafted system prompt starting with "You are ...".
'''


def get_crafting_prompt(
    dimension,
    desired_behavior,
    requirements=requirements,
    ):

    return f'''
            ## Goal

            Imaging a chatbot is a human and doing a personality traits test on {d_convert[dimension]}. Your goal is to craft a system prompt for this chatbot, so that with the crafted system prompt, the chatbot behaves in a certain way. 

            ## Game Instruction

            The chatbot will answer 10 questions regarding to {d_convert[dimension]}.
            
            ## Desired Behavior

            For this particular personality trait, please generate a system prompt for the chatbot. With the generated system prompt and the question, the score for {d_convert[dimension]} chatbot should make is: {desired_behavior} out of 50.

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


def play_single_game(
        game, 
        system_prompt, 
        n_choices=10
    ):


    def extract_numbers_from_brackets(s):
        if s is None:
            return None
            
        if s.count('[') > 1 or s.count(']') > 1:
            return None
        
        match = re.search(r'\[(\d+)\]', s)
        return int(match.group(1)) if match else None
    
    choices = []
    game_instruction_message = game_instruction + ocean_questions[game]
        
    while len(choices) < 10:
        completion = client_0513.chat.completions.create(
            # model="gpt-4o-2024-08-06",
            model='gpt-4o',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": game_instruction_message}
            ],
            n=n_choices,
        )

        for s in [choice.message.content for choice in completion.choices]:
            choice = extract_numbers_from_brackets(s)
            if choice != None:
                choices.append(choice)


    return choices[:10], completion.to_dict()

def play_dimension(dimension,
                   system_prompt = "Image your are a human.",
                   n_choices = 10):
    
    completion_each = []
    game_score_dict = {}
    for game in ocean_questions.keys():
        if game.startswith(dimension):
            output = play_single_game(game, system_prompt, n_choices=n_choices)
            game_score_dict[game] = output[0]
            completion_each.append(output[1])
    game_score_df = pd.DataFrame(game_score_dict)

    def custom_sum1(row, dimension):
        result = 0
        for col in game_score_df.columns:
            if col.startswith(dimension):
                if col in changed_qs:
                    result += abs(6 - row[col])
                else:
                    result += row[col]
        return result
    

    game_score_df.loc[:, 'score'] = game_score_df.apply(lambda row: custom_sum1(row, dimension), axis=1)

    socre_lst = game_score_df["score"].tolist()
    
    return socre_lst, completion_each, game_score_df

def craft_system_prompt(
    game,
    desired_behavior,
    n_sample_per_learner=10,
    n_improvement=3
):  
    
    #(game+"   "+str(desired_behavior))
    prompts = []
    choices = []
    desired = []
    last_modes = []

    initial_prompt = get_crafting_prompt(game, desired_behavior)
    messages = [
        {"role": "user", "content": initial_prompt}
    ]

    def craft():
        
        prompt = None
        while prompt == None:
            completion = client_0513.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                n=1,
            )

            prompt = completion.choices[0].message.content

        choice = play_dimension(
            game, 
            system_prompt=prompt, 
            n_choices=n_sample_per_learner
        )[0]

        last_mode = statistics.mode(choice)
        last_modes.append(last_mode)
        if np.std(choice) < 10: 
            # discard if variance is too large
            prompts.append(prompt)
            choices.append(choice)
            desired.append(desired_behavior)
        else:
            print("if np.std(choice) > 10: ")
        
        messages.append({
            "role": "assistant", 
            "content": completion.choices[0].message.content
        })
    
    craft()

    for _ in range(n_improvement):
        last_mode = last_modes[-1]
        if last_mode == desired_behavior: 
            break

        improve_prompt = get_improving_prompt(last_mode, desired_behavior)
        messages.append({"role": "user", "content": improve_prompt})
        craft()
        #print(last_mode)

    # select prompt based on the mode
    
    return prompts, choices, desired

def samples_to_dist(
    samples, 
    weights=None,
):
    if weights is not None: 
        assert len(samples) == len(weights)
    if isinstance(samples[0], list):
        if weights is not None:
            lengths = [len(lst) for lst in samples]
            weights_expanded = [np.repeat(w, l) for w, l in zip(weights, lengths)]
            weights = np.concatenate(weights_expanded)
            
        samples = np.concatenate([np.array(lst) for lst in samples])
    dist =  np.histogram(
        samples, 
        weights=weights, 
        bins=41, 
        range=(10, 51), 
        density=True
    )[0]
    dist = dist / np.sum(dist)
    return dist

def weight_optimization(
    target,
    # target_dist, # [samples]
    choices, # [[choices], [], ...]
    previous_weights = np.array([]), 
    reg=3,
    n_rounds=10,
):
    target_dist = samples_to_dist(target)

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
        k = kstest(target, generated).pvalue
        return w * (.1 if k > 0.05 else 1)
    
    def loss_1(
        weight,
        weights_pre = previous_weights,
        choices=choices,
        target_dist=target_dist,
    ):  
        weights = np.concatenate((weights_pre, weight))
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
        k = kstest(target, generated).pvalue
        return w * (.1 if k > 0.05 else 1)
    
    def regularization(weights):
        return reg * np.linalg.norm(weights)
    
    bounds =  Bounds([0], [1.])

    def l1_normalize(arr):
        return arr / arr.sum()

    for _ in range(n_rounds):
        x0 = np.random.rand(1)
        
        result = minimize(
            fun=lambda x: loss_1(x) + regularization(x), 
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            tol=1e-6
        )
        if result.fun != np.inf and result.fun > 0.5:
            weight = result.x
            weights = np.concatenate((previous_weights, weight))

            weights_normalized = l1_normalize(weights)

            return weights_normalized, loss(weights_normalized)
    assert False

def get_desired_behaviors(
    target_dist,
    choices,
    weights,
    n=10,
):
    desired_behaviors = None
    if len(choices) == 0:
        desired_behaviors = np.random.choice(
            list(range(10, 51)), 
            size=n, 
            p=target_dist/np.sum(target_dist)
        ).tolist()
    else:
        generated_dist = samples_to_dist(choices, weights)
        dist = (target_dist - generated_dist).clip(min=0)
        desired_behaviors = np.random.choice(
            list(range(10, 51)), 
            size=n, 
            p=dist/np.sum(dist)
        ).tolist()
    # desired_behaviors = [int(5 * round(x/5)) for x in desired_behaviors]
    return desired_behaviors

def gb_run(
    dimension,
    n_test,
    max_iter = 200,
):  

    target_data = df_all_with_score["score_"+dimension].tolist()

    target_dist = samples_to_dist(target_data)

    pool_prompts = []
    pool_choices = []
    pool_disired_behaviors = []
    weights = np.array([1])
    weights_lst = [np.array([1])]


    for iter in range(max_iter):
        
        desired_behavior = get_desired_behaviors(
            target_dist, pool_choices, weights,
            n=1
        )

        prompts, choices, desired = craft_system_prompt(dimension, desired_behavior)
        if len(prompts) < 1:
            continue
        else:
            pool_prompts.append(prompts[-1])
            pool_choices.append(choices[-1])
            pool_disired_behaviors.append(desired[-1])
        
        if iter > 0:
            weights = 0.7 * weights
            weights, loss = weight_optimization(target_data, pool_choices, previous_weights=weights)
            weights_lst.append(weights)
        

        if (iter+1)%5 == 0:
            print(f'Iteration {iter+1}/{max_iter} Loss: {loss}')

        result = {
            "prompts": pool_prompts,
            "weights": weights_lst,
            "choices": pool_choices,
            "disired_behaviors": pool_disired_behaviors
        }

        result_df = pd.DataFrame(result)
        result_df.to_csv(dimension+"/"+str(n_test)+'_result.csv')

def main(args):
    print(f"This is the experiments for {args.dimension}.")
    for i in range(5):
        print("---------Run: "+str(i+1)+" Begin---------")
        folder_path = str(args.dimension)
        os.makedirs(folder_path, exist_ok=True)
        gb_run(args.dimension, i+1)
        print("---------Run: "+str(i+1)+" End---------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is for GB formalization Experiments")
    parser.add_argument("--dimension", type=str)
    
    args = parser.parse_args()
    main(args)

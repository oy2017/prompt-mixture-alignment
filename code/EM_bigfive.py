# Load Packages

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
import argparse

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

client_0513 = AzureOpenAI(
    azure_endpoint="https://<your-endpoint-0513>.openai.azure.com/",
    api_key=os.getenv("AZURE_API_KEY_0513"),
    api_version="2024-05-01-preview",
)

client_0718 = AzureOpenAI(
    azure_endpoint="https://<your-endpoint-0718>.openai.azure.com/",
    api_key=os.getenv("AZURE_API_KEY_0718"),
    api_version="2024-05-01-preview",
)

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


def play_single_game(
        game, 
        system_prompt, 
        n_choices=10
    ):


    def extract_numbers_from_brackets(s):
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

            For this particular question, please generate a system prompt for the chatbot. With the generated system prompt and the above question provided, the score for {d_convert[dimension]} chatbot should make is: {desired_behavior} out of 50.

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
            weights = np.array(weights)[:, np.newaxis].repeat(len(samples[0]), axis=1)
            weights = weights.flatten()
        samples = np.array(samples).flatten()
    dist =  np.histogram(
        samples, 
        weights=weights, 
        bins=41, 
        range=(10, 51), 
        density=True
    )[0]
    dist = dist / np.sum(dist)
    return dist

def initialization(
    num_test,
    K,
    dimension,
):  
    target_distribution = df_all_with_score["score_"+dimension].tolist()
    
    prompts_lst = []
    choices_lst = []
    disired_behaviors_lst = []

    target_dist = samples_to_dist(target_distribution)

    sampled_desired_behavior = np.random.choice(
                    list(range(len(target_dist))), 
                    size=K, 
                    p=target_dist/np.sum(target_dist),
                    replace=False
                ).tolist()

    # sampled_desired_behavior = random.sample(range(0, gamerange[game]), K)
    
    for disired_behavior in tqdm(sampled_desired_behavior):
        prompts, choices, desired = craft_system_prompt(
            dimension,
            disired_behavior,
        )

        if len(choices) == 0:
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

    df_unique_rows.to_csv(dimension+"/"+str(num_test)+"_result/EM_initialization_prompts.csv")
    
    return df_unique_rows

# Method 3: softmax then normalize w distance and soft assign the data points to the prompt
def data_allocation_1(
        num_test,
        num_iter,
        system_prompt_df,
        dimension,
        weights = None
    ):

    target_dist = df_all_with_score["score_"+dimension].tolist()

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
        for i in range(5, 51)
    } 

    def softmax(column):
        exp_values = np.exp(column - np.max(column))  
        return exp_values / np.sum(exp_values)
    
    for i in range(5, 51):
        new_columns[i] = softmax(new_columns[i])

    new_columns =  pd.DataFrame(new_columns)
    new_columns_list = new_columns.values.tolist()

    result = []
    for i, row in enumerate(new_columns_list):
        weight = weights[i] #if i < len(weights) else 1  
        result.append([value * weight for value in row])

    new_columns = pd.DataFrame(result, index=new_columns.index, columns=new_columns.columns)
    system_prompt_probability_df = pd.concat([system_prompt_probability_df, new_columns], axis=1)

    for i in range(5, 51):
        if system_prompt_probability_df[i].sum() != 0:
            system_prompt_probability_df[i] = system_prompt_probability_df[i] / system_prompt_probability_df[i].sum()

    # assign the prompt index to the data point with the smallest w distance
    prompt_lst_index = system_prompt_probability_df.index.tolist()
    for data_point in set(target_dist):
        all_probability = system_prompt_probability_df[data_point].tolist()
        all_probability = [1 / (probability if probability != 0 else 1e-6) for probability in all_probability]
        closest_index = random.choices(prompt_lst_index, weights=all_probability, k=1)[0]
        cluster_allocation[int(data_point)] = closest_index

    # get the target distribution for each prompt based on its allocation
    prompt_target_dist_dict = {}

    for key, values in cluster_allocation.items():
        if values not in prompt_target_dist_dict.keys():
            prompt_target_dist_dict[values] = []

        prompt_target_dist_dict[values] = prompt_target_dist_dict[values] + [key]*list(target_dist).count(key)

    system_prompt_probability_df.to_csv(dimension+"/"+str(num_test)+"_result/"+str(num_iter)+'_system_prompt_probability_df.csv')

    return cluster_allocation, prompt_target_dist_dict


def latentvariabel_update_system_prompts(
    dimension,
    system_prompt_df,
    prompt_target_dist_dict
):
    
    system_prompt_df_tmp = system_prompt_df.copy()
    
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
            updated_prompts, update_prompt_choices, desired = craft_system_prompt(
                dimension,
                desired_behavior,
            )

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

    return system_prompt_df_tmp

    
def weight_optimization(
    target,
    choices,
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
    dimension,
    K = 5, # number of system prompts to generate
    numIter = 5,
):
    
    target_distribution = df_all_with_score["score_"+dimension].tolist()
    # The first step is get the initial system prompt
    print("----------Initilization Begin----------")
    df_unique_rows = initialization(num_test, K = K, dimension=dimension)
    print("----------Initilization End----------")

    loss = np.inf
    weights1 = None
    weights_lst = []

    for num_iter in range(numIter):
        # E-step: allocate each data point in the target distribution to the closest system prompt
        cluster_allocation, prompt_target_dist_dict = data_allocation_1(num_test,
                                                                        num_iter,
                                                                        df_unique_rows, 
                                                                        dimension = dimension,
                                                                        weights = weights1)
        
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
        
        with open(dimension+"/"+str(num_test)+"_result/"+str(num_iter)+"_cluster_allocation.json", "w") as json_file:
            json.dump(cluster_allocation, json_file, indent=4, default=convert_numpy)  
        
        with open(dimension+"/"+str(num_test)+"_result/"+str(num_iter)+"_prompt_target_dist_dict.json", "w") as json_file:
            json.dump(convert_keys_and_values(prompt_target_dist_dict), json_file, indent=4, default=convert_numpy)  
        
        # M-step: 
        # update the system prompt
        df_unique_rows_update = latentvariabel_update_system_prompts(
                                        dimension,
                                        df_unique_rows,
                                        prompt_target_dist_dict
                                    )
        df_unique_rows_update.to_csv(dimension+"/"+str(num_test)+"_result/"+str(num_iter)+'_EM_initialization_prompts_updated.csv')

        pool_choices = df_unique_rows_update['choices'].tolist()
        print(f"Iteration {num_iter}: Optimizing weights...")
        weights1, loss_update = weight_optimization(target_distribution, pool_choices)
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

    with open(dimension+"/"+str(num_test)+'_weights_lst.pkl', 'wb') as f:
        pickle.dump(weights_lst, f)

    
def main(args):
    print(f"This is the experiments for {d_convert[args.dimension]}.") 
    
    for i in range(5): 
        print("---------Run: "+str(i+1)+" Begin---------")  
        folder_path = str(args.dimension)+"/"+str(i+1)+"_result" 
        os.makedirs(folder_path, exist_ok=True)  
        em_play(i+1, args.dimension, K = args.K)
        print("---------Run: "+str(i+1)+" End---------")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is for EM formalization Experiments")
    parser.add_argument("--dimension", type=str)
    parser.add_argument("--K", type=int)
    
    args = parser.parse_args()
    main(args)


import pandas as pd
from openai import OpenAI
import statistics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.stats import kstest
from scipy.optimize import LinearConstraint, Bounds, minimize
import random
import json
import pickle
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

# OpenRouter serves the paper's exact model snapshots via the OpenAI SDK.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
GEN_MODEL = "openai/gpt-4o-2024-05-13"
EXTRACT_MODEL = "openai/gpt-4o-mini-2024-07-18"

df_joint = pd.read_csv('../data/joint.csv')

df_joint.rename(columns={
    'dictator': 'Dictator',
    'ultimatum_1': 'Proposer',
    'ultimatum_2': 'Responder',
    'trust_1': 'Investor',
    'trust_3': 'Banker',
    'bomb': 'Bomb',
    'PG': 'Public_Goods',
}, inplace=True)


game2inst = {
    'Dictator': "You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]).",
    'Proposer': "This is a two-player game. You are the Proposer, and the other player is the Responder. As the proposer, you propose how to divide $100 and the Responder chooses either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \nHow much would you like to propose to give to the Responder? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).",
    'Responder': "This is a two-player game. You are the Responder, and the other player is the Proposer. The proposer proposes how to divide $100 and you, as the Responder, choose either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \nAs the Responder, what is the minimal amount in the proposal that you would accept? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).",
    'Investor': "This is a two-player game. You are an Investor and the other player is a Banker. You have $100 to invest and you choose how much of your money to invest with the Banker. The amount you choose to invest will grow by 3x with the Banker. For example, if you invest $10, it will grow to $30 with the Banker. The Banker then decides how much of the money ($0-$30) to return to you, the Investor.\nHow much of the $100 would you like to invest with the Banker? Please give only one concrete choice, and highlight the number with [] (such as [$x]).",
    'Banker': "This is a two-player game. You are a Banker and the other player is an Investor, and the goal for each player is to earn more. The Investor chooses how much of the money (up to $100) to invest with you. The amount the Investor invests will generate a 2x return with you (the current value of investment becomes 3x).  To settle the investment, as the Banker, you get to decide how much of this total amount to return to the Investor and the rest will be kept as your profit.  For example, you can choose to return $0 (therefore the investor will lose their investment), or you can return the entire 3x (initial investment + 2x profit) to the investor, or you can return any amount in between.\nNow, if the investor has invested $50 with you and the current value became $150, how much of the $150 would you like to return to the Investor? Please give only one concrete choice, and highlight the number with [] (such as [$x]).",
    'Public_Goods': "In this public good game, you and 3 others will choose how much to contribute towards a water cleaning project. Each of you gets $20 per round to contribute between $0 and $20. The project has a 50% return rate. Your payoff relies on both your and others' contributions. Everyone benefits from the group's total contribution. Your payoff in each round equals the amount you didn't contribute (endowment - contribution) plus everyone's benefit (total contributions * 50% return rate). Here are two examples to calculate your payoff.\n\nExample one: You contributed $12; total group contributions were $20\n\nYour Payoff = ($20-$12) + $20*50% = $18\n\nExample two: You contributed $12; total group contributions were $30\n\nYour Payoff = ($20-$12) + $30*50% = $23\n\nWe will play a total of 3 rounds, in the first round, how much of the $20 would you like to contribute? Please give a concrete number and highlight it with [] (e.g., [x]).",
    'Bomb': "There are 100 boxes, and one bomb has been randomly placed in 1 of 100 boxes. You can choose to open 0-100 boxes at the same time. If none of the boxes you open has the bomb, you earn points that are equal to the number of boxes you open. If one of the boxes you open has the bomb, you earn zero points.  How many boxes would you open? Please give one concrete number and highlight it with [] (such as [x]).",
}

gamerange = {
    'Dictator': 100,
    'Proposer': 100,
    'Responder': 100,
    'Investor': 100,
    'Banker': 150,
    'Public_Goods': 20,
    'Bomb': 100,

}

requirements = '''
## Crafting Requirements

* The ultimate goal is to prompt the chatbot to behave towards the desired behavior under the given scenario, rather than being a precoded helpful assistant chatbot. 
* Craft the system prompt based on understanding the meaning of the desired behavior under the given game scenario.
* Do not include information already included in the game instructions. 
* Craft a generalizable system prompt and avoid including any information specific to this particular game or directly implying the desired behavior. 
'''

output_format = '''
## Output Format

Directly output the crafted system prompt starting with "You are ...".
'''

def get_crafting_prompt(
    game,
    desired_behavior,
    requirements=requirements,
):
    return f'''
## Goal

Assuming a chatbot is playing games. Your goal is to craft a system prompt for this chatbot, so that with the crafted system prompt, the chatbot behaves in a certain way. 

## Game Instruction

One of the game instructions provided to the chatbot is:
"{game2inst[game]}"

## Desired Behavior

For this particular game, please generate a system prompt for the chatbot. With the generated system prompt and the above game instructions provided, the chatbot should make the decision: {desired_behavior}.

{requirements}
{output_format}
'''

def get_improving_prompt(
    sampled_behavior,
    desired_behavior,
    requirements=requirements,
):
    return f'''
Using your crafted system prompt, a chatbot outputs mostly {sampled_behavior} instead of {desired_behavior}. Do you have any idea how to improve the system prompt?

{requirements}
{output_format}
'''

def _api_call(model, messages, retries=5):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(model=model, messages=messages, n=1)
            if completion.choices[0].message.content:
                return completion
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    return None


def _play_one(game, system_prompt):
    # OpenRouter ignores n>1, so each sample is its own request.
    completion = _api_call(GEN_MODEL, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": game2inst[game]}
    ])
    response = completion.choices[0].message.content

    for _ in range(5):
        completion = _api_call(EXTRACT_MODEL, [
            {"role": "system", "content": "You are a helpful assistant who helps extract the choice in a conversation. With a conversation between a user and a chatbot provided, please extract the chatbot's choice regarding the user's question. "},
            {"role": "user", "content": game2inst[game]},
            {"role": "assistant", "content": response},
            {"role": "user", "content": "Please output one single integer number that stands for the choice without anything else:"}
        ])
        try:
            choice = completion.choices[0].message.content
            choice = ''.join(filter(str.isdigit, choice))
            return int(choice), completion.to_dict()
        except Exception:
            pass
    return None, None


def play(
        game,
        system_prompt="You are a helpful assistant.",
        n_choices=1
    ):
    choices = []
    completions = []
    for _ in range(3):  # retry batches until n_choices collected
        missing = n_choices - len(choices)
        if missing <= 0:
            break
        with ThreadPoolExecutor(min(missing, 10)) as ex:
            futs = [ex.submit(_play_one, game, system_prompt) for _ in range(missing)]
            for f in futs:
                c, comp = f.result()
                if c is not None:
                    choices.append(c)
                    completions.append(comp)
    return choices, completions


def craft_system_prompt(
    game,
    desired_behavior,
    n_sample_per_learner=10,
    n_improvement=3
):  
    prompts = []
    choices = []
    desired = []
    last_modes = []

    initial_prompt = get_crafting_prompt(game, desired_behavior)
    messages = [
        {"role": "user", "content": initial_prompt}
    ]

    def craft():
        completion = _api_call(GEN_MODEL, messages)

        prompt = completion.choices[0].message.content
        choice = play(
            game,
            system_prompt=prompt,
            n_choices=n_sample_per_learner
        )[0]

        if len(choice) == 0:
            last_modes.append(None)
            return

        last_mode = statistics.mode(choice)
        last_modes.append(last_mode)
        if np.std(choice) < 10: 
            # discard if variance is too large
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
        if last_mode == desired_behavior: break

        improve_prompt = get_improving_prompt(last_mode, desired_behavior)
        messages.append({"role": "user", "content": improve_prompt})
        craft()

    return prompts, choices, desired


def samples_to_dist(
    samples, 
    weights=None,
    range_size=200
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
        bins=range_size+1, 
        range=(0, range_size), 
        density=True
    )[0]
    dist = dist / np.sum(dist)
    return dist

def initialization(
    num_test,
    K,
    game_name,
):
    
    target_distribution = df_joint[game_name].dropna().values
    
    prompts_lst = []
    choices_lst = []
    disired_behaviors_lst = []

    target_dist = samples_to_dist(target_distribution, range_size=gamerange[game_name])

    sampled_desired_behavior = np.random.choice(
                    list(range(len(target_dist))), 
                    size=K, 
                    p=target_dist/np.sum(target_dist),
                    replace=False
                ).tolist()

    for disired_behavior in tqdm(sampled_desired_behavior):
        prompts, choices, desired = craft_system_prompt(
            game_name,
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

    df_unique_rows.to_csv(game_name+"/"+str(num_test)+"_result/EM_initialization_prompts.csv")
    
    return df_unique_rows


# Method 3: softmax then normalize w distance and soft assign the data points to the prompt
def data_allocation_1(
        num_test,
        num_iter,
        system_prompt_df,
        game_name,
        weights = None,
        df_joint = df_joint
    ):

    target_dist = df_joint[game_name].dropna().values

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
        for i in range(gamerange[game_name]+1)
    } 

    def softmax(column):
        exp_values = np.exp(column - np.max(column))  
        return exp_values / np.sum(exp_values)
    
    for i in range(gamerange[game_name]+1):
        new_columns[i] = softmax(new_columns[i])

    new_columns =  pd.DataFrame(new_columns)
    new_columns_list = new_columns.values.tolist()
    result = []
    for i, row in enumerate(new_columns_list):
        weight = weights[i] #if i < len(weights) else 1  
        result.append([value * weight for value in row])

    new_columns = pd.DataFrame(result, index=new_columns.index, columns=new_columns.columns)
    system_prompt_probability_df = pd.concat([system_prompt_probability_df, new_columns], axis=1)

    for i in range(gamerange[game_name]+1):
        if system_prompt_probability_df[i].sum() != 0:
            system_prompt_probability_df[i] = system_prompt_probability_df[i] / system_prompt_probability_df[i].sum()
        

    # assign the prompt index to the data point with the smallest w distance
    prompt_lst_index = system_prompt_probability_df.index.tolist()
    for data_point in set(target_dist):
        all_probability = system_prompt_probability_df[data_point].tolist()
        all_probability = [1 / (w if w != 0 else 1e-6) for w in all_probability]
        closest_index = random.choices(prompt_lst_index, weights=all_probability, k=1)[0]
        cluster_allocation[int(data_point)] = closest_index

    # get the target distribution for each prompt based on its allocation
    prompt_target_dist_dict = {}

    for key, values in cluster_allocation.items():
        if values not in prompt_target_dist_dict.keys():
            prompt_target_dist_dict[values] = []

        prompt_target_dist_dict[values] = prompt_target_dist_dict[values] + [key]*list(target_dist).count(key)

    system_prompt_probability_df.to_csv(game_name+"/"+str(num_test)+"_result/"+str(num_iter)+'_system_prompt_probability_df.csv')

    return cluster_allocation, prompt_target_dist_dict

def latentvariabel_update_system_prompts(
    game_name,
    system_prompt_df,
    prompt_target_dist_dict
):
    
    system_prompt_df_tmp = system_prompt_df.copy()
    for key, value in tqdm(prompt_target_dist_dict.items()):

        # get the base prompt and its choices list
        base_prompt_choice_lst = system_prompt_df_tmp['choices'].loc[key]

        # get the target distribtion based on the allocation in the E step
        target_dist = samples_to_dist(prompt_target_dist_dict[key], range_size=gamerange[game_name])
        # the generated distribution is from the choices of a prompt
        generate_dist = samples_to_dist(base_prompt_choice_lst, range_size=gamerange[game_name])
        
        choice_mode = statistics.mode(base_prompt_choice_lst)
        desired_behavior = statistics.mode(prompt_target_dist_dict[key])


        if choice_mode == desired_behavior:
            continue
        else:
            updated_prompts, update_prompt_choices, desired = craft_system_prompt(
                game_name,
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
        
        update_system_choice_dist = samples_to_dist(update_prompt_choices[-1], range_size=gamerange[game_name])
        
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
    choices, # [[choices], [], ...]
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
    game_name,
    K = 50, # number of system prompts to generate
    numIter = 5,
):
    
    target_distribution = df_joint[game_name].dropna().values
    # The first step is get the initial system prompt
    print("----------Initilization Begin----------")
    df_unique_rows = initialization(num_test, K = K, game_name=game_name)
    print("----------Initilization End----------")

    loss = np.inf
    weights1 = None
    weights_lst = []

    for num_iter in range(numIter):
        # E-step: allocate each data point in the target distribution to the closest system prompt
        cluster_allocation, prompt_target_dist_dict = data_allocation_1(num_test,
                                                                        num_iter,
                                                                        df_unique_rows, 
                                                                        game_name = game_name,
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
        
        with open(game_name+"/"+str(num_test)+"_result/"+str(num_iter)+"_cluster_allocation.json", "w") as json_file:
            json.dump(cluster_allocation, json_file, indent=4, default=convert_numpy)  
        
        with open(game_name+"/"+str(num_test)+"_result/"+str(num_iter)+"_prompt_target_dist_dict.json", "w") as json_file:
            json.dump(convert_keys_and_values(prompt_target_dist_dict), json_file, indent=4, default=convert_numpy)  
        
        # M-step: 
        # update the system prompt
        df_unique_rows_update = latentvariabel_update_system_prompts(
                                        game_name,
                                        df_unique_rows,
                                        prompt_target_dist_dict
                                    )
        # print(df_unique_rows)

        df_unique_rows_update.to_csv(game_name+"/"+str(num_test)+"_result/"+str(num_iter)+'_investor_EM_prompts_updated.csv')

        # update weights of the system prompts
        pool_choices = df_unique_rows_update['choices'].tolist()
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


    
    with open(game_name+"/"+str(num_test)+'_weights_lst.pkl', 'wb') as f:
        pickle.dump(weights_lst, f)

    

def main(args):
    print(f"This is the experiments for {args.game}.")
    for i in range(args.runs):
        print("---------Run: "+str(i+1)+" Begin---------")
        os.makedirs(args.game+"/"+str(i+1)+"_result", exist_ok=True)
        em_play(i+1, args.game, args.K)
        print("---------Run: "+str(i+1)+" End---------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is for EM formalization Experiments")
    parser.add_argument("--game", type=str)
    parser.add_argument("--K", type=int)
    parser.add_argument("--runs", type=int, default=5)
    
    args = parser.parse_args()
    main(args)



    


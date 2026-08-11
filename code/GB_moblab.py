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


client_0513 = AzureOpenAI(
    azure_endpoint="https://<endpoint_0513>.openai.azure.com/",
    api_key="AZURE_API_KEY_0513",
    api_version="2024-05-01-preview",
)

client_0718 = AzureOpenAI(
    azure_endpoint="https://<endpoint_0718>.openai.azure.com/",
    api_key="AZURE_API_KEY_0718",
    api_version="2024-05-01-preview",
)

df_joint = pd.read_csv('../joint.csv')

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
    'Banker_50': "This is a two-player game. You are a Banker and the other player is an Investor, and the goal for each player is to earn more. The Investor chooses how much of the money (up to $100) to invest with you. The amount the Investor invests will generate a 2x return with you (the current value of investment becomes 3x).  To settle the investment, as the Banker, you get to decide how much of this total amount to return to the Investor and the rest will be kept as your profit.  For example, you can choose to return $0 (therefore the investor will lose their investment), or you can return the entire 3x (initial investment + 2x profit) to the investor, or you can return any amount in between.\nNow, if the investor has invested $50 with you and the current value became $150, how much of the $150 would you like to return to the Investor? Please give only one concrete choice, and highlight the number with [] (such as [$x]).",
    'Banker_75': "This is a two-player game. You are a Banker and the other player is an Investor, and the goal for each player is to earn more. The Investor chooses how much of the money (up to $100) to invest with you. The amount the Investor invests will generate a 2x return with you (the current value of investment becomes 3x).  To settle the investment, as the Banker, you get to decide how much of this total amount to return to the Investor and the rest will be kept as your profit.  For example, you can choose to return $0 (therefore the investor will lose their investment), or you can return the entire 3x (initial investment + 2x profit) to the investor, or you can return any amount in between.\nNow, if the investor has invested $50 with you and the current value became $150, how much of the $150 would you like to return to the Investor? Please give only one concrete choice, and highlight the number with [] (such as [$x]).",
    'Public_Goods': "In this public good game, you and 3 others will choose how much to contribute towards a water cleaning project. Each of you gets $20 per round to contribute between $0 and $20. The project has a 50% return rate. Your payoff relies on both your and others' contributions. Everyone benefits from the group's total contribution. Your payoff in each round equals the amount you didn't contribute (endowment - contribution) plus everyone's benefit (total contributions * 50% return rate). Here are two examples to calculate your payoff.\n\nExample one: You contributed $12; total group contributions were $20\n\nYour Payoff = ($20-$12) + $20*50% = $18\n\nExample two: You contributed $12; total group contributions were $30\n\nYour Payoff = ($20-$12) + $30*50% = $23\n\nWe will play a total of 3 rounds, in the first round, how much of the $20 would you like to contribute? Please give a concrete number and highlight it with [] (e.g., [x]).",
    'Bomb': "There are 100 boxes, and one bomb has been randomly placed in 1 of 100 boxes. You can choose to open 0-100 boxes at the same time. If none of the boxes you open has the bomb, you earn points that are equal to the number of boxes you open. If one of the boxes you open has the bomb, you earn zero points.  How many boxes would you open? Please give one concrete number and highlight it with [] (such as [x]).",
}

gamerange = {
    'Dictator': 100,
    'Proposer': 100,
    'Responder': 100,
    'Investor': 100,
    'Banker_50': 150,
    'Banker_75': 150,
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

def play(
        game, 
        system_prompt="You are a helpful assistant.", 
        n_choices=10
    ):
    
    completion = client_0513.chat.completions.create(
        # model="gpt-4o-2024-08-06",
        model='gpt-4o',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": game2inst[game]}
        ],
        n=n_choices,
    )

    responses = [choice.message.content for choice in completion.choices]
    choices = []
    completions = []

    for response in responses:
        if response == None:
            print(response)
            continue
        
        
        while True:
            completion = client_0718.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "system", "content": "You are a helpful assistant who helps extract the choice in a conversation. With a conversation between a user and a chatbot provided, please extract the chatbot's choice regarding the user's question. "},
                    {"role": "user", "content": game2inst[game]},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": "Please output one single integer number that stands for the choice without anything else:"}
                ],
            )
            try:
                choice = completion.choices[0].message.content
                choice = ''.join(filter(str.isdigit, choice))
                choice = int(choice)
                choices.append(choice)
                completions.append(completion.to_dict())

                break
            except:
                pass

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
        
        prompt = None
        while prompt == None:
            completion = client_0513.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                n=1,
            )

            prompt = completion.choices[0].message.content

        choice = play(
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
            lengths = [len(lst) for lst in samples]
            weights_expanded = [np.repeat(w, l) for w, l in zip(weights, lengths)]
            weights = np.concatenate(weights_expanded)

        samples = np.concatenate([np.array(lst) for lst in samples])
    dist =  np.histogram(
        samples, 
        weights=weights, 
        bins=range_size+1, 
        range=(0, range_size), 
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
            list(range(len(target_dist))), 
            size=n, 
            p=target_dist/np.sum(target_dist)
        ).tolist()
    else:
        generated_dist = samples_to_dist(choices, weights)
        dist = (target_dist - generated_dist).clip(min=0)
        desired_behaviors = np.random.choice(
            list(range(len(dist))), 
            size=n, 
            p=dist/np.sum(dist)
        ).tolist()
    # desired_behaviors = [int(5 * round(x/5)) for x in desired_behaviors]
    return desired_behaviors

def gb_run(
    game_name,
    n_test,
    max_iter = 200,
):  

    target_data = df_joint[game_name].dropna().values

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

        prompts, choices, desired = craft_system_prompt(game_name, desired_behavior)
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
        result_df.to_csv(game_name+"/"+str(n_test)+'_result.csv')

def main(args):
    print(f"This is the experiments for {args.game}.") 
    for i in range(5): 
        print("---------Run: "+str(i+1)+" Begin---------")  
        folder_path = str(args.dimension)+"/"+str(i+1)+"_result" 
        os.makedirs(folder_path, exist_ok=True) 
        gb_run(args.game, i+1)
        print("---------Run: "+str(i+1)+" End---------")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is for GB formalization Experiments")
    parser.add_argument("--game", type=str)
    
    args = parser.parse_args()
    main(args)
    





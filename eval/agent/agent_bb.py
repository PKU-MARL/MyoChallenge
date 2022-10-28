import os
import time

import torch
from utils import RemoteConnection

time.sleep(60)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8086")
else:
    rc = RemoteConnection("localhost:8086")

flag_completed = None  # this flag will detect then the whole eval is finished
repetition = 0
while not flag_completed:
    flag_trial = None  # this flag will detect the end of an episode/trial
    counter = 0
    repetition += 1
    while not flag_trial:

        if counter == 0:
            print(
                "BB: Trial #"
                + str(repetition)
                + "Start Resetting the environment and get 1st obs"
            )
            obs = rc.reset()
            # LOAD the policy every reset
            file_name = "learned_policy_boading.pkl"
            root_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
            policy_path = root_path + f"/policies/{file_name}"
            policy = torch.jit.load(policy_path)
            print("Baoding Ball agent: policy loaded")

        ################################################
        ### B - HERE it is obtained the action from the model and passed to the remove environment
        with torch.no_grad():
            action = policy(
                torch.as_tensor(obs, dtype=torch.float32, device="cpu")
            ).numpy()
        ################################################

        ## gets info from the environment
        base = rc.act_on_environment(action)
        obs = base["feedback"][0]

        flag_trial = base["feedback"][2]
        flag_completed = base["eval_completed"]

        print(
            f"BAODING ): Agent Feedback iter {counter} -- trial solved: {flag_trial} -- task solved: {flag_completed}"
        )
        print("*" * 100)
        counter += 1

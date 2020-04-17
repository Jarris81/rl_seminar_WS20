import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, isdir, join

if __name__ == "__main__":

    path = "data/"
    all_env = [f for f in listdir(path) if isdir(join(path, f))]

    print(all_env)
    for env in all_env:
        path = join("data/", env)
        all_agents_csv = [f for f in listdir(path) if isfile(join(path, f))]

        agent_names = []
        rewards = []

        for agent_csv in all_agents_csv:
            # extract name of agent
            df = pd.read_csv(join(path, agent_csv))
            agent_name = agent_csv[agent_csv.startswith(env) and len(env):agent_csv.endswith(".csv") and -len(".csv")]
            plt.plot(range(len(df[agent_name])), df[agent_name], label=agent_name)

        plt.legend()
        plt.title(env)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.show()


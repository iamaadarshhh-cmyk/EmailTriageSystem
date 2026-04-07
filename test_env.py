from openenv_wrapper import EmailTriageEnv

env = EmailTriageEnv()

obs, _ = env.reset()
print("Initial Obs:", obs)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)

    print("Action:", action)
    print("Reward:", reward)
    print("Done:", done)

    if done:
        break

env.close()
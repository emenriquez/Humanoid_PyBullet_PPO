from gym.envs.registration import register
register(
    id='HumanoidBasicEnv-v0',
    entry_point='Humanoid_Basic_Env.envs:HumanoidBasicEnv'
)

register(
    id='HumanoidTinyEnv-v0',
    entry_point='Humanoid_Basic_Env.envs:HumanoidTinyEnv'
)
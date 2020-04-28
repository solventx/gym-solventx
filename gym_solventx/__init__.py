from gym.envs.registration import register

try:
    register(
        id='gym_solventx-v0',
        entry_point='gym_solventx.envs:SolventXEnv',
        kwargs={'config_file': '..\\environment_design_config.json','identifier':''},
        max_episode_steps=100,       
    )
except:
    pass
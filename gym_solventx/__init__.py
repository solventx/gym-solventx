from gym.envs.registration import register

try:
    register(
        id='gym_solventx-v0',
        entry_point='gym_solventx.envs:SolventXEnv',
        max_episode_steps=100,
        kwargs={
            'goals_list':       ['Purity', 'Recovery'],
            'DISCRETE_REWARD' : False,}
    )
except:
    pass
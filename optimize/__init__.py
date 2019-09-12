from gym.envs.registration import register

register(
    id='optimize-v0',
    entry_point='optimize.envs:hills',
)

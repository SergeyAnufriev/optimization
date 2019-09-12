from gym.envs.registration import register

register(
    id='optimize-v0',
    entry_point='optimization_1.envs:hills',
)

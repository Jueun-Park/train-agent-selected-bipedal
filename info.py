mode_dict = {
    0: "grass",
    1: "stump",
    2: "stairs",
    3: "pit",
    4: "bipedal-hardcore",
}
total_timesteps = 50000000  # 50M

eval_freq = 3000000 # 3M

lr_schedule = "linear"
# lr_schedule = "constant"

linear_init_lrs = [0.01, 0.02, 0.4, 0.08]
constant_init_lrs = [0.005, 0.01, 0.02, 0.04]

init_lr = linear_init_lrs[0]

num_cpu = 64

expr_nickname = "debugging"

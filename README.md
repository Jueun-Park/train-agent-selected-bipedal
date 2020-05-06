# Train Agents on Selected Bipedal Environment
Training example codes for [gym-selected-bipedal](https://github.com/Jueun-Park/gym-selected-bipedal) using [stable-baselines](https://github.com/hill-a/stable-baselines)

## Usage
```bash
python train.py --base-index 0
```

Please choose the `base-index` value as a key to the `mode_dict`. The `mode_dict` looks like:
```python
mode_dict = {
    0: "grass",
    1: "stump",
    2: "stairs",
    3: "pit",
}
```

## Credits
[mjyoo2](https://github.com/mjyoo2) found hyperparameters

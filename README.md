# Marai
A toned down version of Large Scale Curiosity focused on Super Mario Brothers.
Uses Python3 and Tensorflow

See our blog post here: TBA

See OpenAI's blog on it here: https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/

## Running
You will need to add a copy of a No-Intro Mario ROM in to the main directry with the file name mario.nes

This project includes two runner, run.py and gym_run.py

run.py uses Michael Burge's headless NES emulator, and may work with non No-Intro ROMs. Has minimal dependencies

gym_run.py requires Retro Gym, and requires the ROM to be No-Intro. See https://github.com/openai/retro/blob/master/README.md for set up. The gym runner also has less functionality, as I used it for testing the model initially.

Both runners will save tensorboard files in summary/

## Replays

If you use the non-gym runner with the "record" flag, the runner will record personal bests for each environment. You can then use make_video.py to create a video of any saved run. The video will display a saliency map of the surprisals, as well as action taken. This will take gigabytes of disk space to store.

import os
import numpy as np
from PIL import Image
import argparse
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_RGB2BGR, resize
import glob
from client import NesClient

parser = argparse.ArgumentParser(description="Aggregates replay data into video")
parser.add_argument("--path", type=str, required=True, help="path to directory with replay")
parser.add_argument("--level", type=str, default="1-1", help="level to start playing, defaults to '1-1'")
parser.add_argument("--fps", type=int, default=60, help="fps of video created")
parser.add_argument("--dump_frame", dest="dump_frame", action='store_true')
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--end_frame", type=int)

opts = parser.parse_args()

save_state_path = "headless_levels/level-" + opts.level + ".state"
if not os.path.exists(save_state_path):
    raise Exception("Could not find save file: ", save_state_path)

data_path = opts.path
if not os.path.exists(data_path):
    raise Exception("Could not locate data directory")

def step_observe(env, action, width, height):
    env.send_set_inputs(action)
    env.send_step_frame()
    frame = env.bytes_to_frame(env.send_render_frame()) 
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def data_min_max(directory, base_name, idx_start, idx_end):
    _min = 1
    _max = -1
    for i in range(idx_start, idx_end):
        fullname = directory + base_name + "_" + str(i) + ".npz"
        data = np.load(fullname)
        _max = max(_max, np.amax(data))
        _min = min(_min, np.amin(data))
    return _min, _max

def heatmapify_surprisal(base_image, gradients, surprisal_magnitude):
    surprisal_magnitude = 1
    shape = base_image.size
    agged_grads = np.sum(np.abs(gradients), axis=-1, keepdims=True)

    grads_absmax = max(agged_grads.max(), -agged_grads.min())
    norm_grads = agged_grads * surprisal_magnitude / grads_absmax
    abs_norm_grads = np.abs(norm_grads)

    c255 = np.ones(shape) * 255.0
    heatmap = np.dstack([c255, ((1.0 - abs_norm_grads) * 255) ,c255]).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap, mode='RGB')
    mask_image = Image.fromarray(((1.0 - abs_norm_grads[:,:,0]) * 255).astype(np.uint8), mode='L')
    return Image.composite(base_image, heatmap_img, mask_image).resize((336, 336))

def paste_action(img, action_bin, x, y, x_offset):
    right = action_bin >> 7 & 1
    left  = action_bin >> 6 & 1
    down  = action_bin >> 5 & 1
    up    = action_bin >> 4 & 1
    a     = action_bin >> 0 & 1
    b     = action_bin >> 1 & 1

    if right:
        dir_img = Image.open("assets/Right.bmp")
    elif left:
        dir_img = Image.open("assets/Left.bmp")
    elif down:
        dir_img = Image.open("assets/Down.bmp")
    elif up:
        dir_img = Image.open("assets/Up.bmp")
    else:
        dir_img = Image.open("assets/None.bmp")


    if a:
        a_img = Image.open("assets/A.bmp")
    else:
        a_img = Image.open("assets/None.bmp")
    if b:
        b_img = Image.open("assets/B.bmp")
    else:
        b_img = Image.open("assets/None.bmp")
    
    dir_x = x
    a_x = x + x_offset
    b_x = x + (x_offset * 2)
    
    img.paste(dir_img, (dir_x, y))
    img.paste(a_img, (a_x, y))
    img.paste(b_img, (b_x, y))


def heatmapify_action(base_image, gradients, logit_magnitude):
    clipped_gradients = np.clip(gradients, a_min=0, a_max=None)
    return heatmapify_surprisal(base_image, clipped_gradients, logit_magnitude)
    #logit_magnitude = 1
    #shape = base_image.size
    #agged_grads = np.mean(gradients, axis=-1, keepdims=True)
    #norm_factor = agged_grads.max() - agged_grads.min()
    #offset = 0.5 - np.min(agged_grads)
    #norm_grads = (agged_grads / (norm_factor * 2)) + offset * logit_magnitude


    #ones = np.ones(shape)
    #c255 = (ones * 255).astype(np.uint8)
    #c128 = (ones * 128).astype(np.uint8)
    ##heatmap = np.dstack([((norm_grads) * 255).astype(np.uint8), c255 / 2, c255])
    #derp = ((norm_grads) * 255).astype(np.uint8)
    #print(np.min(derp))
    #print(np.max(derp))
    #heatmap = np.dstack([((norm_grads) * 255).astype(np.uint8), c128, c255])
    #heatmap_img = Image.fromarray(heatmap, mode='RGB')
    #mask_image = Image.fromarray(((1.0 - np.abs(agged_grads / max(agged_grads.max(), -agged_grads.min()))[:,:,0]) * 255).astype(np.uint8), mode='L')

    #return Image.composite(base_image, heatmap_img, mask_image).resize((164, 164))

def generate():
    save_path = data_path + '/save.video'
    states_path = data_path + '/states/'
    
    actions_fh = open(save_path, mode='rb')
    actions = actions_fh.read()[8:]
    
    env = NesClient(opts.level, save_state_path).init_mario()
    num_saved_states = len(glob.glob(states_path + "state_*.bmp"))

    working_dir = "editing/"

    surprisals = np.load(data_path + "/surps.npy")
    neglogps = np.load(data_path + "/action_neglogps.npy")
    surp_grads = np.load(data_path + "/surp_grads.npy")
    action_grads = np.load(data_path + "/action_grads.npy")
    
    max_surprisal = np.amax(surprisals, axis=0)

    max_neglogps = np.amax(neglogps, axis=0) 
    min_neglogps = np.amin(neglogps, axis=0)


    canvas = Image.open("assets/template.bmp")
    
    video_format = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    video = cv2.VideoWriter("video.mp4", fourcc, opts.fps, canvas.size, True)
    end_frame = len(actions)
    if opts.end_frame is not None:
        end_frame = opts.end_frame
    for i in range(0, end_frame):
        is_last = i == (num_saved_states - 1) 
        is_first = i == 0

        action = actions[i]
        frame = step_observe(env, action, 512, 480)
        frame_image = Image.fromarray(frame, "RGB")
        canvas.paste(frame_image, (0, 0))

        if i % 12 == 0:
            state_idx = i // 12
            state = Image.open(states_path + "state_" + str(state_idx) + ".bmp")
            norm_neglogp = (neglogps[state_idx] - min_neglogps) / (max_neglogps - min_neglogps)
            norm_surp = surprisals[state_idx] / max_surprisal
            ds_di = surp_grads[state_idx]
            da_di = action_grads[state_idx]
            if is_first:
                surp_heatmap = Image.new("RGB", (84, 84)) 
            else:
                norm_surp = surprisals[state_idx- 1] / max_surprisal
                ds_di = surp_grads[state_idx]
                surp_heatmap = heatmapify_surprisal(state, ds_di, norm_surp)
            canvas.paste(surp_heatmap, (512, 32))
            paste_action(canvas, action, 530, 420, 120)
#            action_heatmap = heatmapify_action(state, da_di, norm_neglogp)
#            canvas.paste(action_heatmap, (512, 436))
        if i >= opts.start_frame:
            video.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
            if i % 12 == 0 and opts.dump_frame:
                canvas.save(working_dir + "frame_" + str(i) + ".bmp", "BMP")
    video.release()

if __name__ == "__main__":
    generate()

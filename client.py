import os
import itertools
import shutil
import numpy as np
from subprocess import *
from PIL import Image


RENDER_STYLE_NES = 0
RENDER_STYLE_RGB = 1

class VideoRecorder:
    _filename = False
    _num_bytes = 0

    def __init__(self, filename):
        self._filename = open(filename, "wb")
        self._filename.write(bytes([ 0, 0, 0, 0, 0, 0, 0, 0 ]))
    def record_input(self, button_mask):
        self._filename.write(bytes([ button_mask ]))
        self._num_bytes += 1
    def commit(self):
        self._filename.seek(0)
        self._filename.write(u32_to_bytes(self._num_bytes))
    def close_fh(self):
        self._filename.close()

class MetadataRecorder:
    def __init__(self):
        self.action_neglogps = []
        self.action_gradients = []
        self.surprisals = []
        self.surprisal_gradients = []

        self.save_dir_path = None
        self.should_dump = False
        self.len_saved_neglogps = 0

    def add_neglogp(self, neglogp):
        self.action_neglogps.append(neglogp)

    def add_surprisals_and_grads(self, surprisals, surp_grads, action_grads):
        for s, sg, ag in zip(surprisals, surp_grads, action_grads):
            self.surprisals.append(s)
            self.surprisal_gradients.append(sg)
            self.action_gradients.append(ag)
            if len(self.surprisals) == self.len_saved_neglogps:
                if self.save_dir_path is not None:
                    np.save(self.save_dir_path + "/surps", self.surprisals)
                    np.save(self.save_dir_path + "/surp_grads", self.surprisal_gradients)
                    np.save(self.save_dir_path + "/action_grads", self.action_gradients)
                    self.save_dir_path = None
                    self.surprisals = []
                    self.surprisal_gradients = []
                    self.action_gradients = []
                if self.should_dump:
                    self.should_dump = False
                    self.surprisals = []
                    self.surprisal_gradients = []
                    self.action_gradients = []
    
    def prep_save(self, dirpath):
        self.save_dir_path = dirpath
        self.len_saved_neglogps = len(self.action_neglogps)
        np.save(self.save_dir_path + "/action_neglogps", self.action_neglogps)
        self.action_neglogps = []

    def prep_drop(self):
        self.should_dump = True
        self.len_saved_neglogps = len(self.action_neglogps)
        self.action_neglogps = []
        
class NesClient:
    _num_commands = 0
    _num_pending_commands = 0
    _is_single_step = False
    _num_bytes_written = 0
    _num_bytes_read = 0
    _process = None


    def __init__(self, starting_level, save_state_path, debug=False):
        self.save_state_path = save_state_path
        self._process = Popen("./headless", stdin=PIPE, stdout=PIPE, bufsize=0)
        self._is_single_step = debug
        self.recorder = None
        self.metadata_recorder = None

        self.queued_action = 0b00000000 #default move should be index zero
        self.num_stacked_frames = 3
        self.num_skipped_frames = 4
        self.visited_levels = set()
        self.starting_level = starting_level
        self.states_alive = 0
        
        self.best_x = (0, 0)
        self.cur_x = (0, 0)

        self.least_actions = self.actions_to_advance = 9999999

    def enable_recorder(self, idx):
        self.env_idx = idx
        self.tmp_filename = "replays/tmp_env" + str(self.env_idx) + ".video"
        self.recorder = VideoRecorder(self.tmp_filename)
        self.metadata_recorder = MetadataRecorder()

    def should_record(self):
        return self.recorder is not None

    def reset_recorders(self):
        if self.should_record():
            self.recorder.commit()
            self.recorder.close_fh()
            
            tmp_replay_dir = "replays/tmp/env_" + str(self.env_idx)
            if len(self.visited_levels) <= 1 and self.cur_x > self.best_x:
                x_id = str(self.cur_x[0]).rjust(3, "0") + str(self.cur_x[1]).rjust(3, "0")
                cur_x_str = str(self.cur_x[0]) + "_" + str(self.cur_x[1])
                dirpath = "replays/pb/env_" + str(self.env_idx) + "/fail_" + cur_x_str + "/"
                os.makedirs(dirpath)
                self.best_x = self.cur_x
                os.rename(self.tmp_filename, dirpath + "save.video")
                os.rename(tmp_replay_dir, dirpath + "states")
                self.metadata_recorder.prep_save(dirpath)
            elif len(self.visited_levels) > 1 and self.actions_to_advance < self.least_actions:
                dirpath = "replays/pb/env_" + str(self.env_idx) + "/pass_" + str(self.actions_to_advance) + "/"
                os.makedirs(dirpath)
                self.least_actions = self.actions_to_advance
                os.rename(self.tmp_filename, dirpath + "save.video")
                os.rename(tmp_replay_dir, dirpath + "states")
                self.metadata_recorder.prep_save(dirpath)
            if os.path.exists(tmp_replay_dir):
                shutil.rmtree(tmp_replay_dir)
                self.metadata_recorder.prep_drop()
            os.makedirs(tmp_replay_dir)

            self.recorder = VideoRecorder(self.tmp_filename)
    
    def save_neglogp_metadata(self, neglogp):
        self.metadata_recorder.add_neglogp(neglogp)

    def save_surprisals_and_grads_metadata(self, surprisals, surp_grads, action_grads):
        self.metadata_recorder.add_surprisals_and_grads(surprisals, surp_grads, action_grads)

    def init_mario(self):
        self.send_load_rom("mario.nes")
        self.send_load_state(self.save_state_path)
        self.set_lives()
        return self

    def update_pb_info(self):
        if len(self.visited_levels) <= 1:
            screen_x = self.send_cpu_peek(0x071c)
            screen_page = self.send_cpu_peek(0x071a)
            #if self.recorder is not None:
            #    print("env: ", self.env_idx, screen_page, screen_x)
            x = (screen_page, screen_x)
            self.cur_x = max(self.cur_x, x)
        else:
            self.actions_to_advance = min(self.actions_to_advance, self.states_alive) 

    def bytes_to_frame(self, frame_bytes, preprocess_func=None):
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        reshaped = np.reshape(frame, (240, 256, 3))
        if preprocess_func is not None:
            return preprocess_func(reshaped)
        return reshaped

    def set_action(self, action):
        self.queued_action = action
    
    def prepare_advance_state(self):
        self.frames_to_stack = []
        self.start_lives = self.read_lives()
        
    def step_skipped_frame(self):
        self.send_set_inputs(self.queued_action)
        if self.recorder:
            self.recorder.record_input(self.queued_action)
        self.send_step_frame()

    def step_observed_frame(self, preprocess_func):
        self.maybe_reset_state()
        processed_frame = self.bytes_to_frame(self.send_render_frame(), preprocess_func)
        self.frames_to_stack.append(processed_frame)
    
    def maybe_reset_state(self):
        delta_lives = self.start_lives - self.read_lives()
        if delta_lives == 255:
            self.reset_recorders()
            self.send_load_state(self.save_state_path)
            self.set_lives()
            self.set_action(0)
            self.visited_levels = set()
            self.states_alive = 0
            self.cur_x = (0, 0)
            self.actions_to_advance = 9999999

    def get_observation(self):


        if self.read_level() >= self.get_starting_level(self.starting_level):
            self.visited_levels.add(self.read_level())
        stacked_frames = np.dstack(self.frames_to_stack)
        if self.recorder:
            im = Image.fromarray(stacked_frames.astype('uint8'))
            fullname = "replays/tmp/env_" + str(self.env_idx) + "/state_" + str(self.states_alive) + ".bmp"
            im.save(fullname)
            self.states_alive += 1
        return stacked_frames

    def get_starting_level(self, s):
        worldstr, levelstr = s.split('-')
        world = int(worldstr) - 1
        level = int(levelstr) - 1
        return str(world) + "-" + str(level)

    def save_surprisal_saliency(self, arr):
        if self.states_alive > 1:
            fullname = "replays/tmp/env_" + str(self.env_idx) + "/surp_" + str(self.states_alive - 1)
            np.save(fullname, arr)


    def save_action_saliency(self, arr):
        fullname = "replays/tmp/env_" + str(self.env_idx) + "/action_" + str(self.states_alive - 1)
        np.save(fullname, arr)
        
                
    def read_lives(self):
        return self.send_cpu_peek(0x075a)
    
    def set_lives(self):
        self.send_cpu_poke(0x075a, 2)

    def read_level(self):
        world_num = self.send_cpu_peek(0x075f)
        level_num = self.send_cpu_peek(0x075c)
        return str(world_num) + "-" + str(level_num) 

    def wait_pending_commands(self):
        #print("DEBUG - WAITING ON COMMANDS", self._num_pending_commands)
        for _ in range(self._num_pending_commands):
            self.check_synchronization()
        self._num_pending_commands = 0

    def send_load_rom(self, filename):
        #print("COMMAND - LOAD ROM", filename)
        self.write_u8(1) # Command
        self.write_u8(0) # Record TAS file?
        self.write_length_string(filename) # INES file to load
        self.add_pending_command()

    def send_step_frame(self):
        #print("COMMAND - STEP FRAME")
        self.write_u8(2) # Command
        self.add_pending_command()

    def send_render_frame(self, render_style=RENDER_STYLE_RGB):
        #print("COMMAND - RENDER FRAME", render_style)
        self.write_u8(3) # Command
        self.write_u8(render_style)
        num_bytes = 256 * 240 * (3 if render_style == 1 else 1)
        self.wait_pending_commands()
        bs = self._process.stdout.read(num_bytes)
        self.check_synchronization()
        return bs

    def send_render_frame_and_save(self, render_style, filename="output"):
        num_bytes = 256 * 240 * (3 if render_style == 1 else 1)
        bs = self.send_render_frame(render_style)
        img_fh = open(filename + ".ppm", "wb")
        write_ppm(img_fh, 256, 240, bs)

    def send_set_inputs(self, button_mask):
        #print("COMMAND - SET INPUTS", button_mask)
        self.write_u8(4) # Command
        self.write_u8(0) # Controller Id
        self.write_u8(button_mask)
        self.add_pending_command()

    def send_save_state(self, filename):
        #print("COMMAND - SAVE STATE", filename)
        self.write_u8(5) # Command
        self.write_length_string(filename)
        self.add_pending_command()

    def send_load_state(self, filename):
        #print("COMMAND - LOAD STATE", filename)
        self.write_u8(6) # Command
        self.write_length_string(filename)
        self.add_pending_command()

    def send_get_info(self):
        error("Unimplemented")

    def send_step(self):
        #print("COMMAND - STEP")
        self.write_u8(8) # Command
        self.add_pending_command()

    def send_save_tas(self, filename):
        #print("COMMAND - SAVE TAS", filename)
        self.write_u8(9) # Command
        self.write_length_string(filename)
        self.add_pending_command()

    def send_cpu_peek(self, ptr):
        #print("COMMAND - PEEK", ptr)
        self.write_u8(10) # Command
        self.write_u16(ptr)
        result = self.read_u8()
        self.add_pending_command()
        return result


    def send_cpu_poke(self, ptr, value):
        #print("COMMAND - POKE", ptr, value)
        self.write_u8(11) # Command
        self.write_u16(ptr)
        self.write_u8(value)
        self.add_pending_command()

    def checked_write(self, x):
        num_bytes = 0
        fh = self._process.stdin
        if type(x) is bytes:
            num_bytes = len(x)
        elif type(x) is bytearray:
            num_bytes = len(x)
            x = bytes(x)
        elif type(x) == int:
            num_bytes = 1
        else:
            error("Unknown type", type(x), x)
        # print("WRITE", WRITE_NUM_BYTES, " - ", x)
        fh.write(x)
        self._num_bytes_written += num_bytes;

    def write_ppm(self, width, height, bytes):
        # print("DEBUG - WRITE PPM", width, height, len(bytes))
        self.checked_write("P6 {} {} 255 ".format(width, height).encode('utf-8'))
        self.checked_write(bytes)
        # for b in bytes:
        #     checked_write(fh, b)

    def write_length_string(self, string):
        # print("DEBUG - WRITE STRING", string)
        bytes = bytearray(string, "utf-8")
        self.write_u32(len(bytes))
        self.checked_write(bytes)

    def write_u8(self, x):
        self.checked_write(bytes([ x ]))

    def write_u16(self, x):
        self.checked_write(bytes([
            (x >> 0) & 0xff,
            (x >> 8) & 0xff
        ]))

    def write_u32(self, x):
        self.checked_write(bytes([
            (x >> 0) & 0xff,
            (x >> 8) & 0xff,
            (x >> 16) & 0xff,
            (x >> 24) & 0xff,
        ]))

    def read_u8(self):
        self.wait_pending_commands()
        return self._read_u8()

    def _read_u8(self):
        fh = self._process.stdout
        x = fh.read(1);
        # print("PYDEBUG - READ", x)
        return x[0]

    def ignore_bytes(self, num_bytes):
        while num_bytes > 0:
            x = fh.read(num_bytes)
            num_bytes -= len(x)

    def error(*x):
        raise(Exception(x))

    def add_pending_command(self):
        self._num_pending_commands += 1
        if self._is_single_step or self._num_pending_commands > 2000:
            self.wait_pending_commands()

    def check_synchronization(self):
        self._num_commands += 1

        sync_commands = self._read_u8()
        if sync_commands == (self._num_commands % 256):
            pass
        else:
            error("DEBUG - SYNC FAIL", self._num_commands, sync_commands)

def u32_to_bytes(x):
    return bytes([
        (x >> 0) & 0xff,
        (x >> 8) & 0xff,
        (x >> 16) & 0xff,
        (x >> 24) & 0xff,
    ])

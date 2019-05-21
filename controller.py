import random
import numpy as np

class Controller:
    def button_mask(self, up, left, down, right, a, b):
        #A bitmask 0bABCDEFGH, where
        #A = Button A
        #B = Button B
        #C = Select
        #D = Start
        #E = Up
        #F = Down
        #G = Left
        #H = Right

        #CAREFUL: Last significant digit is on the right
        
        #Disable select, start, up, down (I think down actually lets him duck but whatever)
        #illegal moves returns None
        #space is 2^4 = 16 for now
       
        if left and right:
            return None
       

        btn_up = btn_left = btn_down = btn_right = btn_a = btn_b = 0b00000000 

        if up:
            btn_up =    0b00010000
        if left: 
            btn_left =  0b01000000
        if down:
            btn_down =  0b00100000
        if right:
            btn_right = 0b10000000
        if a:
            btn_a =     0b00000001
        if b:
            btn_b =     0b00000010

        return btn_up | btn_left | btn_down | btn_right | btn_a | btn_b
    
    def button_array(self, indexes):
        moves = self.all_moves()
        
        ret = []
        for index in indexes:
            sub_ret = []
            move = moves[index]
            b = bin(move)[2:]
            for i in range(8 - len(b)):
                sub_ret.append(float(0))
            for s in b:
                sub_ret.append(float(s))
            ret.append(sub_ret)
        return ret 

    def all_moves(self):
        return np.array([
            self.button_mask(False, False, False, False, False, False),  # NOOP
            self.button_mask(True,  False, False, False, False, False),  # Up
            self.button_mask(False, False, True,  False, False, False),  # Down
            self.button_mask(False, True,  False, False, False, False),  # Left
            self.button_mask(False, True,  False, False, True,  False),  # Left + A
            self.button_mask(False, True,  False, False, False, True),   # Left + B
            self.button_mask(False, True,  False, False, True,  True),   # Left + A + B
            self.button_mask(False, False, False, True,  False, False),  # Right
            self.button_mask(False, False, False, True,  True,  False),  # Right + A
            self.button_mask(False, False, False, True,  False, True),   # Right + B
            self.button_mask(False, False, False, True,  True,  True),   # Right + A + B
            self.button_mask(False, False, False, False, True,  False),  # A
            self.button_mask(False, False, False, False, False, True),   # B
            self.button_mask(False, False, False, False, True,  True)    # A + B
        ])

    def rand_move(self, num_moves=1):
        ret = []
        for i in range(num_moves):
            roll = random.randint(0, len(self.all_moves()) - 1)
            ret.append(roll)
        return ret


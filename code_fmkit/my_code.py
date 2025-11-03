from fmsignal import *
from fmsignal_vis import *
from fmsignal_demo import *

signal = load_one_demo_signal_pp(device='glove')
trajectory_animation(signal, seg_length=-1)

import os

# Please update the below to your own paths
AMASS_PATH = "./amass/processed"
SMPLH_PATH = "./amass/body_models/smplh"
RIGNET_PATH = './rignet_models'
CUSTOM_PATH = './custom'
MIXAMO_PATH = './mixamo'
MIXAMO_SIMPLIFY_PATH = './mixamo_simplify'
BLEND_FILE = './scene.blend'
BLENDER_PATH = "\"C:/Program Files/Blender Foundation/Blender 2.83/blender.exe\""
TEMP_DIR = './temp'
LOG_DIR = './03_handle'

RADOM_SEED = 2021

os.makedirs(TEMP_DIR, exist_ok=True)


MIXAMO_JOINTS = [ "Hips",   # 0
  "LeftFoot",               # 1
  "Spine1",                 # 2
  "Spine2",                 # 3
  "RightLeg",               # 4
  "RightUpLeg",             # 5
  "RightForeArm",           # 6
  "LeftUpLeg",              # 7
  "RightToeBase",           # 8
  "LeftShoulder",           # 9
  "RightHand",              # 10
  "LeftForeArm",            # 11
  "LeftArm",                # 12
  "RightArm",               # 13
  "RightFoot",              # 14
  "Head",                   # 15
  "LeftHand",               # 16
  "RightShoulder",          # 17
  "LeftLeg",                # 18
  "Spine",                  # 19
  "LeftToeBase",            # 20
  "Neck"]                   # 21

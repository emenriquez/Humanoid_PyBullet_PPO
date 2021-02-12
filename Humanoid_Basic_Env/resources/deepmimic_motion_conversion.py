import json
import math

skillName = ["walk",
        "run",
        "jump",
        "kick",
        "punch",
        "roll",
        "spin",
        "spinkick",
        "backflip",
        "cartwheel",
        "crawl",
        "dance_a",
        "dance_b",
        "getup_facedown",
        "getup_faceup"]
for movement in skillName:
    with open(f"Assets/Motions/humanoid3d_{movement}.txt", 'r') as f:
        skill = json.load(f)

    convertedSkill = {
        'Frames': []
    }

    for frame in skill['Frames']:
        skillFrame = {
            'deltaTime': frame[0],
            'rootPosition':     {'x': frame[1], 'y' : frame[2], 'z': frame[3]},
            'rootRotation':     {'w': frame[4], 'x' : frame[5], 'y': frame[6], 'z': frame[7]},
            'chestRotation':    {'w': frame[8], 'x' : frame[9], 'y': frame[10], 'z': frame[11]},
            'neckRotation':     {'w': frame[12], 'x': frame[13], 'y': frame[14], 'z': frame[15]},
            'rHipRotation':     {'w': frame[16], 'x': frame[17], 'y': frame[18], 'z': frame[19]},
            'lHipRotation':     {'w': frame[30], 'x': frame[31], 'y': frame[32], 'z': frame[33]},

            'rAnkleRotation':   {'w': frame[21], 'x': frame[22], 'y': frame[23], 'z': frame[24]},
            'lAnkleRotation':   {'w': frame[35], 'x': frame[36], 'y':  frame[37], 'z': frame[38]},
            
            'lShoulderRotation': {'w': frame[25], 'x': frame[26], 'y': frame[27], 'z': frame[28]},
            'rShoulderRotation': {'w': frame[39], 'x': frame[40], 'y': frame[41], 'z': frame[42]},

            'rKneeRotation': {'z': frame[20]*(180/math.pi), 'x': 0, 'y': 0},
            'lKneeRotation': {'z': frame[34]*(180/math.pi), 'x': 0, 'y': 0},

            'lElbowRotation': {'z': frame[29]*(180/math.pi), 'x': 0, 'y': 0},
            'rElbowRotation': {'z': frame[43]*(180/math.pi), 'x': 0, 'y': 0},
            }

        convertedSkill['Frames'].append(skillFrame)

    with open(f"Assets/Motions/{movement}.json", 'w') as f:
        json.dump(convertedSkill, f)

import numpy as np
import os
from core import G
import getpath as gp
human = G.app.selectedHuman
Base = "/Users/jeffwang/desktop/Testing_Face/"

#load data
data = np.load('/Users/jeffwang/desktop/Dynamic_Facial/pred_testing.npy')
# the set of action units
action_units = [
'Upper Face AUs/R43',
'Upper Face AUs/L43',
'Upper Face AUs/R5',
'Upper Face AUs/L5',
'Upper Face AUs/4_a',
'Upper Face AUs/4_a',
'Upper Face AUs/R2',
'Upper Face AUs/L2',
'Lip Parting and Jaw Opening/26',
'Miscellaneous AUs/R30',
'Miscellaneous AUs/L30',
'Lower Face AUs/R12',
'Lower Face AUs/L12',
'Lower Face AUs/R20',
'Lower Face AUs/L20',
'Lower Face AUs/28_a',
'Lower Face AUs/28_bottom',
'Miscellaneous AUs/29',
'Lower Face AUs/10',
'Lower Face AUs/16',
'Lower Face AUs/17',
'Lower Face AUs/18',
'Miscellaneous AUs/34',
'Lower Face AUs/9'
]

for i in range(0, len(data)):
    #set value for different action units
    directory = Base + "img_" + str(i) + '/'
    os.mkdir(directory)
    MHScript.setZoom(4.0)
    MHScript.setRotationZ(0)
    MHScript.setPositionX(0)
    for r in range(0, len(data[i])):
        if r == 4:
            AU_modifier = human.getModifier(action_units[r])
            AU_modifier.setValue(data[i][r])
            human.applyAllTargets()
        elif r == 5:
            continue
        else:
            AU_modifier = human.getModifier(action_units[r])
            AU_modifier.setValue(data[i][r])
            human.applyAllTargets()
    mh.redraw()
    imgsavepath = (directory + "right" + ".png")
    mh.grabScreen(G.windowWidth/2-120, G.windowHeight/2-140, 240, 240, imgsavepath)
    # set action units back
    for k in range(0, len(data[i])):
        AU_modifier = human.getModifier(action_units[k])
        AU_modifier.setValue(0)
        human.applyAllTargets()

    for l in range(0, len(data[i])):
        if l == 4:
            continue
        elif l == 5:
            AU_modifier = human.getModifier(action_units[l])
            AU_modifier.setValue(data[i][l])
            human.applyAllTargets()
        else:
            AU_modifier = human.getModifier(action_units[l])
            AU_modifier.setValue(data[i][l])
            human.applyAllTargets()
    mh.redraw()
    imgsavepath = (directory + "left" + ".png")
    mh.grabScreen(G.windowWidth/2-120, G.windowHeight/2-140, 240, 240, imgsavepath)
    # set action units back
    for k in range(0, len(data[i])):
        AU_modifier = human.getModifier(action_units[k])
        AU_modifier.setValue(0)
        human.applyAllTargets()

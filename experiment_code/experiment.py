#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created eon Thu Dec 31 16:29:31 2015
@author: gweindel

Edited by Kenneth Muller
"""
import os #handy system and path functions
from psychopy import core, visual, event, gui, data, monitors
import psychopy.logging as logging
import csv
import numpy as np
from psychopy.constants import *
import ctypes #for parallel port
#import serial needed for USB com
from itertools import permutations
import random
from datetime import datetime


_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)


expName = 'HsMM_contrast'
#expInfo = {'participant':'S','type':'practice,prdm,training,task,test','con':''}
expInfo = {'participant':'S1','type':'test','con':'0.025'}

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()
expInfo['date'] = data.getDateStr()
expInfo['expName'] = expName

n_subj = int(expInfo['participant'].split('S')[-1])-1

conditions = ['acc','spd']
if n_subj % 2 == 0: conditions = ['spd','acc']

# ..... Settings ......

if expInfo['type'] == 'training':
    NtrialsSession = 80 #number of trials per session
    Ntrials = NtrialsChange = [0,20,40,60] #number of trials before feedback
    conditions =  ['spd','acc']
    con_strengths = np.repeat(float(expInfo['con']), NtrialsSession) #Contrast difference between Left and right
    
elif expInfo['type'] == 'test':
    NtrialsSession = 1116 #number of trials per session (352 * 4)
    NtrialsChange = [0,279,558,837]
    Ntrials = [0,139,279,418,558,697,837,976] #number of trials before feedback
    order =  conditions
    con_strengths = np.repeat(float(expInfo['con']), NtrialsSession) #Contrast difference between Left and right

else:
    core.quit()

# ..... Trigger definition ...... For EEG acqusition, TODO later
conditions_to_trigger = {"acc":"101", "spd":"201"}
side_to_trigger = {"left":"99", "right":"199"}
response_to_trigger = {"left":"100", "right":"200"}

#An ExperimentHandler isn't essential but helps with log...
filename = _thisDir + os.sep + u'Data' + os.sep + '%s_%s_%s' %(expInfo['participant'], expInfo['type'], expInfo['date'])
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    dataFileName=filename)
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)

#Definition screen
mon1 = monitors.Monitor('EEG')
resolution = [1024,768]
mon1.setDistance(90) #cm #For EEG acqusition, TODO later
mon1.setWidth(33) #cm
mon1.setSizePix(resolution)
mon1.saveMon() 

#create a window to draw in
win = visual.Window(size=resolution, monitor=mon1, allowGUI=False, units='deg',fullscr=True)#gamma

class KeyResponse:
    def __init__(self):
        self.keys=[]
        self.corr=0
        self.rt=None
        self.clock=None

def send_trigger(code): #For EEG acqusition, TODO later
    ctypes.windll.inpoutx64.Out32(0x4FD8, code)
    core.wait(0.015)
    ctypes.windll.inpoutx64.Out32(0x4FD8, 0)
send_trigger(0)

fixation = visual.GratingStim(win=win,#Fixation Cross
            mask='cross', size=0.4,
            pos=[0,0], sf=0, color='black')
            
square_diode = visual.Rect(win=win,#photo diode square # For EEG acqusition, TODO later
            size=[1,1], pos=[-19.4,-9], fillColor='white')
square_decision = visual.Rect(win=win,#photo diode square # For EEG acqusition, TODO later
            size=[1,1], pos=[-19.4,-9], fillColor='black')

gabor = visual.GratingStim(win,tex="sin", #Gabor patch
            mask="gauss",texRes=256,  pos=[0,0],
            size=2.5, sf=[1.2,0], ori = 0, name='gabor')

pause=visual.TextStim(win, ori=0,height = 0.5, name='pause',
    text=u"Take a break and press any button to resume.")

pause_exp = visual.TextStim(win, ori=0,height = 0.5, name='pause',
    text=u"Paused by the experimenter.")

wrong_key = visual.TextStim(win, ori=0,height = 0.5, name='pause',
    text=u"Wrong key used, left or right expected")

texte_fin=visual.TextStim(win, height = 0.5, ori=0, name='texte_fin',
    text=u"End \n Please wait for the instructions from the experimenter")

instruction_start = visual.TextStim(win, ori=0,height = 0.5, name='instruction_start',
    text='''During the experiment, you will be asked to judge which of two visual gratings (left or right) displays more contrast.
        \n You can provide your response by pressing a button (n for left, m for right).
        \n During this experiment, you will be asked to alternate between a focus on accuracy or on the speed of your responses, which will be indicated at the start of each block.\n''')

instruction_acc  = '''For the following trials, you will have to be as accurate as possible
    '''

instruction_spd  = '''For the following trials, you will have to be as fast as possible
    '''

instructions_dict = {"acc":instruction_acc, "spd":instruction_spd}

RTTextMeanSub = visual.TextStim(win,alignText='left',
                        units='norm',height = 0.1, pos=[0, 0.125],
                        text='Mean : ...',color='white',autoLog = False)

PrecTextMeanSub = visual.TextStim(win,alignText='left',
                        units='norm',height = 0.1, pos=[0, -0.125],
                        text='Mean : ...',color='white',autoLog = False)

conditionText = visual.TextStim(win, alignText='left',
                        units='norm',height = 0.05, pos=[0, 0],
                        text='...',color='white',autoLog = False)

################################### INSTRUCTIONS
while True:
    instruction_start.draw()
    win.flip()
    instr_event= event.getKeys()
    if instr_event:
        break
    if event.getKeys(["escape"]): core.quit()
    instructionPending = True


#############################TRIALS
#Init
trialClock = core.Clock()
counter = 0
bloc_counter = 0
block = 0
rep = KeyResponse()
rep.clock = core.Clock()

#Trial loop
condition = conditions[0]
sumRT = 0
currentRT =0
sumPrec = 0
currentPrec = 0

for trial in range(NtrialsSession):
    event.clearEvents(eventType='keyboard')
    #Defining break and condition switch
    if trial == 0 and expInfo['type'] == 'test':
        send_trigger(254)# For EEG acqusition, TODO later
    if counter > 1 and counter in Ntrials:
        meanRT = sumRT / bloc_counter
        meanPrec = float(sumPrec) / float(bloc_counter)
        block +=1
        bloc_counter = 0
        sumRT = 0
        currentRT =0
        sumPrec = 0
        currentPrec = 0
        pausePending = False
        FBPending = True
        continueRoutine = True
        event.clearEvents(eventType='keyboard')
        while continueRoutine:
            if FBPending :
                if expInfo['type'] != 'prdm':
                    RTTextMeanSub.text= u"Reaction time for decision = %i ms" %(meanRT*1000)
                    RTTextMeanSub.draw()
                    PrecTextMeanSub.text=u"Accuracy for decision = %i %%" %(meanPrec*100)
                    PrecTextMeanSub.draw()
                    win.flip()  
                    if event.getKeys(["escape"]): core.quit()
                    if event.getKeys("space"):#TODO only experimenter can pass this screen
                        FBPending = False
                        event.clearEvents(eventType='keyboard')
                        break
                else:
                    FBPending = False
                    break
            if pausePending:
                pause.draw()
                win.flip()
                if event.getKeys(["escape"]): core.quit()
                if  event.getKeys("space") :
                    bypassPause=0

                    pause
                    win.flip()
                    pausePending = False
                    continueRoutine=False
    if counter == 0 or counter in NtrialsChange:#print instruction
        event.clearEvents(eventType='keyboard')
        
        cList = np.arange(float(expInfo['con']) + 0.01,0.99 - float(expInfo['con']),0.01) # list of all possible mean contrast levels
        cList = np.repeat(cList, 3)

        np.random.shuffle(cList)
        
        if condition == 'acc': condition = 'spd'#alternate condition for next bloc
        elif condition == 'spd': condition = 'acc'
        while True:
            conditionText.text = instructions_dict[condition]
            conditionText.draw()    
            win.flip()
            if event.getKeys("space"):
                win.flip()
                break
#composants boucle
    counter += 1
    bloc_counter += 1
#Fixation
    square_diode.draw()
    win.flip()
    core.wait(0.5) #ISI
    fixation.draw()
    square_diode.draw()
    win.flip()

#Contrast diff definition
    delta_con = con_strengths[counter-1]#contrast difference
    gabor.contrast = .5#average contrast
    corr_ans_con = np.random.choice(['left','right'])
    rep.status = NOT_STARTED
    rep.rt = np.nan
    rep.keys = np.nan
    gabor.status = STARTED
    continueRoutine = True
    waiting_time = np.random.uniform(0.5, 1.25,1)
    core.wait(waiting_time[0]-0.2)#Fix - stim
    
    send_trigger(int(conditions_to_trigger[condition]))#
    core.wait(0.1)#
    send_trigger(int(side_to_trigger[corr_ans_con]))#
    core.wait(0.1)#imposed delay, usefull to see on acquisition screen
    while continueRoutine:
        if event.getKeys(["p"]):#Stops experiment incase of a problem, discard trial
            pause_exp.draw()
            win.flip()
            while True:
                pause_event = event.getKeys("p") #passed by experimentator
                if pause_event:
                    thisExp.addData('direction', np.nan)
                    thisExp.addData('keys',np.nan)
                    thisExp.addData('rt',np.nan)
                    thisExp.addData('condition', condition)
                    thisExp.addData('con_strength', delta_con)
                    thisExp.addData('trial',counter)
                    thisExp.addData('block',block)
                    thisExp.addData('comment', 'paused')
                    send_trigger(333)
                    thisExp.nextEntry()
                    event.clearEvents(eventType='keyboard')
                    conditionText.text = instructions_dict[condition]
                    conditionText.draw()    
                    win.flip()
                    while True:
                        if event.getKeys():
                            win.flip()
                            continueRoutine = False
                            break
                    break
        if gabor.status == STARTED:
            gabor.contrast = cList[0]
            if corr_ans_con == "left":
                gabor.contrast += delta_con 
            else: gabor.contrast -= delta_con
            gabor.pos = [-.6,0]
            gabor.draw()
            # gabor.contrast = .5
            gabor.contrast = cList[0]
            if corr_ans_con == "right":
                gabor.contrast += delta_con 
            else: gabor.contrast -= delta_con 
            gabor.pos = [.6,0]
            gabor.draw()
            square_decision.draw()
            fixation.draw()
            win.flip()
            send_trigger(int(cList[0]*100))
            gabor.status = NOT_STARTED        
            rep.clock.reset()
        if rep.status == NOT_STARTED:
            rep.status = STARTED
            event.clearEvents(eventType='keyboard')
        if event.getKeys(["escape"]): core.quit()
        if rep.status == STARTED:
            theseKeys = []
            if event.getKeys("n"):
                theseKeys.append("left")
                trigger = 100
            elif event.getKeys("m"):
                theseKeys.append("right")
                trigger = 200
            if len(theseKeys) > 0:
                send_trigger(trigger)
                rep.rt = rep.clock.getTime()
                if "escape" in theseKeys:
                    win.close()
                    core.quit()
                rep.keys = theseKeys[0]
                rep.status = NOT_STARTED
                event.clearEvents(eventType='keyboard')
                if rep.keys == corr_ans_con:
                    rep.corr = 1
                else:
                    rep.corr = 0
                currentRT = rep.rt
                currentPrec = rep.corr
                sumPrec = sumPrec + currentPrec
                sumRT = sumRT + currentRT
        #Fin de boucle
                thisExp.addData('direction', corr_ans_con)
                thisExp.addData('keys1',rep.keys)
                thisExp.addData('rt',rep.rt)
                thisExp.addData('condition', condition)
                thisExp.addData('con_strength', delta_con)
                thisExp.addData('mean', cList[0])
                thisExp.addData('trial',counter)
                thisExp.addData('block',block)
                thisExp.addData('comment', '')
                thisExp.addData('timestamp',datetime.now())
                thisExp.nextEntry()
                continueRoutine = False
                cList = cList[1:]
    if event.getKeys(["escape"]):
        win.close
        core.quit()


#routine fin
meanRT = sumRT / bloc_counter
meanPrec = float(sumPrec) / float(bloc_counter)
FBPending = True
continueRoutine = True
while continueRoutine:
    event.clearEvents(eventType='keyboard')
    if FBPending:
        RTTextMeanSub.text= u"Reaction time for decision = %i ms" %(meanRT*1000)
        RTTextMeanSub.draw()
        PrecTextMeanSub.text=u"Accuracy for decision = %i %%" %(meanPrec*100)
        PrecTextMeanSub.draw()
        win.flip()
        if event.getKeys(["escape"]): core.quit()
        if event.getKeys(["space"]):#TODO only experimenter can pass this screen
            FBPending = False
            finPending = True
            event.clearEvents(eventType='keyboard')
            continueRoutine=True
            while continueRoutine:
                if  finPending:
                    core.wait(2.0)
                    texte_fin.draw()
                    win.flip()
                    core.wait(5.0)
                    finPending = False
                if not continueRoutine:
                    break
                continueRoutine = False
                if event.getKeys(["escape"]):
                    core.quit()

win.close()
core.quit()

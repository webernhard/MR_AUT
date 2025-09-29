#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on August 01, 2025, at 14:06
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2025.1.1')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, parallel
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'CreaLying'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'UPN-ID': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['UPN-ID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\eeglabor\\Documents\\_PsychoPy\\2025_CreaLying\\CreaLying_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('debug')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='bw@home', color=[-1,-1,-1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1,-1,-1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('i_key') is None:
        # initialise i_key
        i_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='i_key',
        )
    if deviceManager.getDevice('p_start_key') is None:
        # initialise p_start_key
        p_start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='p_start_key',
        )
    if deviceManager.getDevice('p_item1_key') is None:
        # initialise p_item1_key
        p_item1_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='p_item1_key',
        )
    if deviceManager.getDevice('smry_key') is None:
        # initialise smry_key
        smry_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='smry_key',
        )
    if deviceManager.getDevice('start_key') is None:
        # initialise start_key
        start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_key',
        )
    if deviceManager.getDevice('thx_key') is None:
        # initialise thx_key
        thx_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='thx_key',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "settings" ---
    # Run 'Begin Experiment' code from set_things
    ##  some start time info
    print(data.getDateStr(format="%H:%M:%S.%m"))   #local: set/mark start of current run
    
    ##  enable parallel port for markers  ##
    from psychopy import parallel
    t_mrk = parallel.ParallelPort(address='0x3FF8')
    
    ##  initialise microphone  ##
    deviceManager.addDevice(
        deviceClass='psychopy.hardware.microphone.MicrophoneDevice',
        deviceName='mic',
        index=None,
        exclusive=False,
    )
    
    #fn = os.path.join(_thisDir, 'data', expInfo['UPN-ID'], expName, data.getDateStr(format="%Y-%m-%d_%H%M"))
    custom_fn = '%s_%s_%s_%s' % (os.path.join(_thisDir, 'data', ''), expInfo['UPN-ID'], expName, data.getDateStr(format="%Y-%m-%d_%H%M"))
    #print(f'{fn=}')
    
    wavDirName = custom_fn + '_wav'
    if not os.path.isdir(wavDirName):
        os.makedirs(wavDirName)  # to hold .wav files
    #print("wavDirName"); print(wavDirName)
    
    ##  make microphone object for mic  #
    idea_mic = sound.microphone.Microphone(
        device='mic',
        name='idea_mic',
        #recordingFolder=micRecFolder,
        recordingFolder=wavDirName,
        recordingExt='wav'
    )
    
    ## ### Options for DEBUG/RESARCH-mode ###
    DEBUG = 0 #0=research-mode; 1=debug-mode 
    if DEBUG:
        loopDur = 7 #total time for ideas
        p_loopDur= 7#30 #total time for ideas
        itemDur = 2
        fixDur  = 1
        rfDur   = 2 #ruhe+fixation nach 3 items
    else:
        loopDur = 180 #total time for ideas
        p_loopDur = 60 #total time for ideas
        itemDur = 30
        fixDur  = 10
        rfDur   = 180 #ruhe+fixation nach 3 items
    
    show_rf = 0 #rf= rest + fixation after 3 items/trials
    
    
    ##  quasi-random item shuffle  ##
    #creaLying_cats = {
    #    "Pro": ["stim\items\Pro1.jpg", "stim\items\Pro2.jpg"],
    #    "Self": ["stim\items\Self1.jpg", "stim\items\Self2.jpg"],
    #    "Anti": ["stim\items\Anti1.jpg", "stim\items\Anti2.jpg"]}
    
    itm = 0 #item
    mrk = 1 #marker-code for item
    creaLying_cats = {
        ##cat.   [item,marker]     #short description of list
        "Pro":  [["Pro1",43], ["Pro2",44]],
        "Self": [["Self1",45], ["Self2",46]],
        "Anti": [["Anti1",41], ["Anti2",42]]}
    
    all_items = []      #empty list
    assigned_cat = {}   #empty dict
    for category, items in creaLying_cats.items():
        all_items.extend(items)
        for item, item_mrk in items:
            assigned_cat[item] = category
    
    done = False
    while not done:
        shuffle(all_items)
        valid = True
        for i in range(len(all_items) - 1):
            #if assigned_cat[all_items[i]] == assigned_cat[all_items[i+1]]:
            if assigned_cat[all_items[i][itm]] == assigned_cat[all_items[i+1][itm]]:
                valid = False
                break
        if valid:
            #print(f'{all_items=}')
            done = True
    
    
    # --- Initialize components for Routine "instr" ---
    i_mrk = parallel.ParallelPort(address='0x3FF8')
    i_img = visual.ImageStim(
        win=win,
        name='i_img', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), draggable=False, size=(1.77,1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    i_hint = visual.TextStim(win=win, name='i_hint',
        text='Weiter mit Leertaste',
        font='Arial',
        pos=(0, -.45), draggable=False, height=0.025, wrapWidth=None, ori=0, 
        color='darkgreen', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    i_key = keyboard.Keyboard(deviceName='i_key')
    
    # --- Initialize components for Routine "p_start" ---
    p_start_txt = visual.TextStim(win=win, name='p_start_txt',
        text='Es folgt ein Übungsbeispiel',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    p_start_key = keyboard.Keyboard(deviceName='p_start_key')
    p_start_hint = visual.TextStim(win=win, name='p_start_hint',
        text='Weiter mit Leertaste',
        font='Arial',
        pos=(0, -.45), draggable=False, height=0.025, wrapWidth=None, ori=0, 
        color='darkgreen', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "p_item1" ---
    p_item1_mrk = parallel.ParallelPort(address='0x3FF8')
    p_item1_img = visual.ImageStim(
        win=win,
        name='p_item1_img', 
        image='stim/Probe1.JPG', mask=None, anchor='center',
        ori=0, pos=(0, 0), draggable=False, size=(1.77,1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    p_item1_key = keyboard.Keyboard(deviceName='p_item1_key')
    p_item1_hint = visual.TextStim(win=win, name='p_item1_hint',
        text='Weiter mit Leertaste',
        font='Arial',
        pos=(0, -.45), draggable=False, height=0.025, wrapWidth=None, ori=0, 
        color='darkgreen', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "p_item2" ---
    p_fix_mrk = parallel.ParallelPort(address='0x3FF8')
    p_fixation = visual.TextStim(win=win, name='p_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    p_item2_mrk = parallel.ParallelPort(address='0x3FF8')
    p_item2_img = visual.ImageStim(
        win=win,
        name='p_item2_img', 
        image='stim/Probe2.JPG', mask=None, anchor='center',
        ori=0, pos=(0, 0), draggable=False, size=(1.77,1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "p_idea" ---
    # Run 'Begin Experiment' code from p_code
    import sounddevice as sd
    import soundfile as sf
    import time
    
    ## set audio recording parameters
    sample_rate = 44100  # Sample rate (Hz)
    channels = 1         # Number of audio channels (2 for stereo)
    
    ## get/access the keyboard
    p_kb = keyboard.Keyboard()
    
    p_FZ = visual.TextStim(win=win, name='p_FZ',
        text='?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "smry" ---
    smry_img = visual.ImageStim(
        win=win,
        name='smry_img', 
        image='stim/instr/Instr4.JPG', mask=None, anchor='center',
        ori=0, pos=(0, 0), draggable=False, size=(1.77,1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=0.0)
    smry_key = keyboard.Keyboard(deviceName='smry_key')
    smry_hint = visual.TextStim(win=win, name='smry_hint',
        text='Weiter mit Leertaste',
        font='Arial',
        pos=(0, -.45), draggable=False, height=0.025, wrapWidth=None, ori=0, 
        color='darkgreen', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "start" ---
    start_txt = visual.TextStim(win=win, name='start_txt',
        text='Es folgen die Testdurchgänge',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    start_key = keyboard.Keyboard(deviceName='start_key')
    
    # --- Initialize components for Routine "t_item" ---
    t_fix_mrk = parallel.ParallelPort(address='0x3FF8')
    t_fixation = visual.TextStim(win=win, name='t_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    t_item_mrk = parallel.ParallelPort(address='0x3FF8')
    t_item_img = visual.ImageStim(
        win=win,
        name='t_item_img', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), draggable=False, size=(1.77,1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "t_idea" ---
    # Run 'Begin Experiment' code from t_code
    """
    import sounddevice as sd
    import soundfile as sf
    import time
    
    ## set audio recording parameters
    sample_rate = 44100  # Sample rate (Hz)
    channels = 1         # Number of audio channels (2 for stereo)
    """
    
    
    ## get/access the keyboard
    t_kb = keyboard.Keyboard()
    
    t_FZ = visual.TextStim(win=win, name='t_FZ',
        text='?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "ruhe_fixation" ---
    rf_fixation = visual.TextStim(win=win, name='rf_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "thx" ---
    thx_txt = visual.TextStim(win=win, name='thx_txt',
        text='Vielen Dank,\n\ndieser Teil der Untersuchung ist zu Ende.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    thx_key = keyboard.Keyboard(deviceName='thx_key')
    thx_hint = visual.TextStim(win=win, name='thx_hint',
        text='Beenden mit Enter',
        font='Arial',
        pos=(.25, -.25), draggable=False, height=0.025, wrapWidth=None, ori=0, 
        color=[-0.25,-0.25,-0.25], colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "settings" ---
    # create an object to store info about Routine settings
    settings = data.Routine(
        name='settings',
        components=[],
    )
    settings.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for settings
    settings.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    settings.tStart = globalClock.getTime(format='float')
    settings.status = STARTED
    thisExp.addData('settings.started', settings.tStart)
    settings.maxDuration = None
    # keep track of which components have finished
    settingsComponents = settings.components
    for thisComponent in settings.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "settings" ---
    settings.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=settings,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            settings.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in settings.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "settings" ---
    for thisComponent in settings.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for settings
    settings.tStop = globalClock.getTime(format='float')
    settings.tStopRefresh = tThisFlipGlobal
    thisExp.addData('settings.stopped', settings.tStop)
    thisExp.nextEntry()
    # the Routine "settings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    byp_start = data.TrialHandler2(
        name='byp_start',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(byp_start)  # add the loop to the experiment
    thisByp_start = byp_start.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisByp_start.rgb)
    if thisByp_start != None:
        for paramName in thisByp_start:
            globals()[paramName] = thisByp_start[paramName]
    
    for thisByp_start in byp_start:
        byp_start.status = STARTED
        if hasattr(thisByp_start, 'status'):
            thisByp_start.status = STARTED
        currentLoop = byp_start
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisByp_start.rgb)
        if thisByp_start != None:
            for paramName in thisByp_start:
                globals()[paramName] = thisByp_start[paramName]
        
        # set up handler to look after randomisation of conditions etc
        ins_loop = data.TrialHandler2(
            name='ins_loop',
            nReps=1, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('stim/CreaLying_instr.csv'), 
            seed=None, 
        )
        thisExp.addLoop(ins_loop)  # add the loop to the experiment
        thisIns_loop = ins_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisIns_loop.rgb)
        if thisIns_loop != None:
            for paramName in thisIns_loop:
                globals()[paramName] = thisIns_loop[paramName]
        
        for thisIns_loop in ins_loop:
            ins_loop.status = STARTED
            if hasattr(thisIns_loop, 'status'):
                thisIns_loop.status = STARTED
            currentLoop = ins_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisIns_loop.rgb)
            if thisIns_loop != None:
                for paramName in thisIns_loop:
                    globals()[paramName] = thisIns_loop[paramName]
            
            # --- Prepare to start Routine "instr" ---
            # create an object to store info about Routine instr
            instr = data.Routine(
                name='instr',
                components=[i_mrk, i_img, i_hint, i_key],
            )
            instr.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            i_img.setImage(iSlide)
            # create starting attributes for i_key
            i_key.keys = []
            i_key.rt = []
            _i_key_allKeys = []
            # store start times for instr
            instr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr.tStart = globalClock.getTime(format='float')
            instr.status = STARTED
            thisExp.addData('instr.started', instr.tStart)
            instr.maxDuration = None
            # keep track of which components have finished
            instrComponents = instr.components
            for thisComponent in instr.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr" ---
            instr.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisIns_loop, 'status') and thisIns_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # *i_mrk* updates
                
                # if i_mrk is starting this frame...
                if i_mrk.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    i_mrk.frameNStart = frameN  # exact frame index
                    i_mrk.tStart = t  # local t and not account for scr refresh
                    i_mrk.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(i_mrk, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('i_mrk.started', t)
                    # update status
                    i_mrk.status = STARTED
                    i_mrk.status = STARTED
                    win.callOnFlip(i_mrk.setData, int(iMarker))
                
                # if i_mrk is stopping this frame...
                if i_mrk.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > i_mrk.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        i_mrk.tStop = t  # not accounting for scr refresh
                        i_mrk.tStopRefresh = tThisFlipGlobal  # on global time
                        i_mrk.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.addData('i_mrk.stopped', t)
                        # update status
                        i_mrk.status = FINISHED
                        win.callOnFlip(i_mrk.setData, int(0))
                
                # *i_img* updates
                
                # if i_img is starting this frame...
                if i_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    i_img.frameNStart = frameN  # exact frame index
                    i_img.tStart = t  # local t and not account for scr refresh
                    i_img.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(i_img, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'i_img.started')
                    # update status
                    i_img.status = STARTED
                    i_img.setAutoDraw(True)
                
                # if i_img is active this frame...
                if i_img.status == STARTED:
                    # update params
                    pass
                
                # *i_hint* updates
                
                # if i_hint is starting this frame...
                if i_hint.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    i_hint.frameNStart = frameN  # exact frame index
                    i_hint.tStart = t  # local t and not account for scr refresh
                    i_hint.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(i_hint, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'i_hint.started')
                    # update status
                    i_hint.status = STARTED
                    i_hint.setAutoDraw(True)
                
                # if i_hint is active this frame...
                if i_hint.status == STARTED:
                    # update params
                    pass
                
                # *i_key* updates
                waitOnFlip = False
                
                # if i_key is starting this frame...
                if i_key.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    i_key.frameNStart = frameN  # exact frame index
                    i_key.tStart = t  # local t and not account for scr refresh
                    i_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(i_key, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'i_key.started')
                    # update status
                    i_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(i_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(i_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if i_key.status == STARTED and not waitOnFlip:
                    theseKeys = i_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                    _i_key_allKeys.extend(theseKeys)
                    if len(_i_key_allKeys):
                        i_key.keys = _i_key_allKeys[-1].name  # just the last key pressed
                        i_key.rt = _i_key_allKeys[-1].rt
                        i_key.duration = _i_key_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr" ---
            for thisComponent in instr.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr
            instr.tStop = globalClock.getTime(format='float')
            instr.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr.stopped', instr.tStop)
            if i_mrk.status == STARTED:
                win.callOnFlip(i_mrk.setData, int(0))
            # check responses
            if i_key.keys in ['', [], None]:  # No response was made
                i_key.keys = None
            ins_loop.addData('i_key.keys',i_key.keys)
            if i_key.keys != None:  # we had a response
                ins_loop.addData('i_key.rt', i_key.rt)
                ins_loop.addData('i_key.duration', i_key.duration)
            # the Routine "instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisIns_loop as finished
            if hasattr(thisIns_loop, 'status'):
                thisIns_loop.status = FINISHED
            # if awaiting a pause, pause now
            if ins_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                ins_loop.status = STARTED
        # completed 1 repeats of 'ins_loop'
        ins_loop.status = FINISHED
        
        
        # set up handler to look after randomisation of conditions etc
        ps_byp = data.TrialHandler2(
            name='ps_byp',
            nReps=0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(ps_byp)  # add the loop to the experiment
        thisPs_byp = ps_byp.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPs_byp.rgb)
        if thisPs_byp != None:
            for paramName in thisPs_byp:
                globals()[paramName] = thisPs_byp[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPs_byp in ps_byp:
            ps_byp.status = STARTED
            if hasattr(thisPs_byp, 'status'):
                thisPs_byp.status = STARTED
            currentLoop = ps_byp
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPs_byp.rgb)
            if thisPs_byp != None:
                for paramName in thisPs_byp:
                    globals()[paramName] = thisPs_byp[paramName]
            
            # --- Prepare to start Routine "p_start" ---
            # create an object to store info about Routine p_start
            p_start = data.Routine(
                name='p_start',
                components=[p_start_txt, p_start_key, p_start_hint],
            )
            p_start.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for p_start_key
            p_start_key.keys = []
            p_start_key.rt = []
            _p_start_key_allKeys = []
            # store start times for p_start
            p_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p_start.tStart = globalClock.getTime(format='float')
            p_start.status = STARTED
            thisExp.addData('p_start.started', p_start.tStart)
            p_start.maxDuration = None
            # keep track of which components have finished
            p_startComponents = p_start.components
            for thisComponent in p_start.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p_start" ---
            p_start.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPs_byp, 'status') and thisPs_byp.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *p_start_txt* updates
                
                # if p_start_txt is starting this frame...
                if p_start_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    p_start_txt.frameNStart = frameN  # exact frame index
                    p_start_txt.tStart = t  # local t and not account for scr refresh
                    p_start_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(p_start_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_start_txt.started')
                    # update status
                    p_start_txt.status = STARTED
                    p_start_txt.setAutoDraw(True)
                
                # if p_start_txt is active this frame...
                if p_start_txt.status == STARTED:
                    # update params
                    pass
                
                # *p_start_key* updates
                waitOnFlip = False
                
                # if p_start_key is starting this frame...
                if p_start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    p_start_key.frameNStart = frameN  # exact frame index
                    p_start_key.tStart = t  # local t and not account for scr refresh
                    p_start_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(p_start_key, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_start_key.started')
                    # update status
                    p_start_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(p_start_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(p_start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if p_start_key.status == STARTED and not waitOnFlip:
                    theseKeys = p_start_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                    _p_start_key_allKeys.extend(theseKeys)
                    if len(_p_start_key_allKeys):
                        p_start_key.keys = _p_start_key_allKeys[-1].name  # just the last key pressed
                        p_start_key.rt = _p_start_key_allKeys[-1].rt
                        p_start_key.duration = _p_start_key_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *p_start_hint* updates
                
                # if p_start_hint is starting this frame...
                if p_start_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    p_start_hint.frameNStart = frameN  # exact frame index
                    p_start_hint.tStart = t  # local t and not account for scr refresh
                    p_start_hint.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(p_start_hint, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_start_hint.started')
                    # update status
                    p_start_hint.status = STARTED
                    p_start_hint.setAutoDraw(True)
                
                # if p_start_hint is active this frame...
                if p_start_hint.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p_start,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    p_start.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in p_start.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p_start" ---
            for thisComponent in p_start.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p_start
            p_start.tStop = globalClock.getTime(format='float')
            p_start.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p_start.stopped', p_start.tStop)
            # check responses
            if p_start_key.keys in ['', [], None]:  # No response was made
                p_start_key.keys = None
            ps_byp.addData('p_start_key.keys',p_start_key.keys)
            if p_start_key.keys != None:  # we had a response
                ps_byp.addData('p_start_key.rt', p_start_key.rt)
                ps_byp.addData('p_start_key.duration', p_start_key.duration)
            # the Routine "p_start" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisPs_byp as finished
            if hasattr(thisPs_byp, 'status'):
                thisPs_byp.status = FINISHED
            # if awaiting a pause, pause now
            if ps_byp.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                ps_byp.status = STARTED
            thisExp.nextEntry()
            
        # completed 0 repeats of 'ps_byp'
        ps_byp.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisByp_start as finished
        if hasattr(thisByp_start, 'status'):
            thisByp_start.status = FINISHED
        # if awaiting a pause, pause now
        if byp_start.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            byp_start.status = STARTED
    # completed 1.0 repeats of 'byp_start'
    byp_start.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    pract_byp = data.TrialHandler2(
        name='pract_byp',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(pract_byp)  # add the loop to the experiment
    thisPract_byp = pract_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPract_byp.rgb)
    if thisPract_byp != None:
        for paramName in thisPract_byp:
            globals()[paramName] = thisPract_byp[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPract_byp in pract_byp:
        pract_byp.status = STARTED
        if hasattr(thisPract_byp, 'status'):
            thisPract_byp.status = STARTED
        currentLoop = pract_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPract_byp.rgb)
        if thisPract_byp != None:
            for paramName in thisPract_byp:
                globals()[paramName] = thisPract_byp[paramName]
        
        # --- Prepare to start Routine "p_item1" ---
        # create an object to store info about Routine p_item1
        p_item1 = data.Routine(
            name='p_item1',
            components=[p_item1_mrk, p_item1_img, p_item1_key, p_item1_hint],
        )
        p_item1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for p_item1_key
        p_item1_key.keys = []
        p_item1_key.rt = []
        _p_item1_key_allKeys = []
        # store start times for p_item1
        p_item1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_item1.tStart = globalClock.getTime(format='float')
        p_item1.status = STARTED
        thisExp.addData('p_item1.started', p_item1.tStart)
        p_item1.maxDuration = None
        # keep track of which components have finished
        p_item1Components = p_item1.components
        for thisComponent in p_item1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_item1" ---
        p_item1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPract_byp, 'status') and thisPract_byp.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *p_item1_mrk* updates
            
            # if p_item1_mrk is starting this frame...
            if p_item1_mrk.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_item1_mrk.frameNStart = frameN  # exact frame index
                p_item1_mrk.tStart = t  # local t and not account for scr refresh
                p_item1_mrk.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_item1_mrk, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_item1_mrk.started', t)
                # update status
                p_item1_mrk.status = STARTED
                p_item1_mrk.status = STARTED
                win.callOnFlip(p_item1_mrk.setData, int(20))
            
            # if p_item1_mrk is stopping this frame...
            if p_item1_mrk.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_item1_mrk.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    p_item1_mrk.tStop = t  # not accounting for scr refresh
                    p_item1_mrk.tStopRefresh = tThisFlipGlobal  # on global time
                    p_item1_mrk.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_item1_mrk.stopped', t)
                    # update status
                    p_item1_mrk.status = FINISHED
                    win.callOnFlip(p_item1_mrk.setData, int(0))
            
            # *p_item1_img* updates
            
            # if p_item1_img is starting this frame...
            if p_item1_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_item1_img.frameNStart = frameN  # exact frame index
                p_item1_img.tStart = t  # local t and not account for scr refresh
                p_item1_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_item1_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_item1_img.started')
                # update status
                p_item1_img.status = STARTED
                p_item1_img.setAutoDraw(True)
            
            # if p_item1_img is active this frame...
            if p_item1_img.status == STARTED:
                # update params
                pass
            
            # *p_item1_key* updates
            waitOnFlip = False
            
            # if p_item1_key is starting this frame...
            if p_item1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_item1_key.frameNStart = frameN  # exact frame index
                p_item1_key.tStart = t  # local t and not account for scr refresh
                p_item1_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_item1_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_item1_key.started')
                # update status
                p_item1_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(p_item1_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(p_item1_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if p_item1_key.status == STARTED and not waitOnFlip:
                theseKeys = p_item1_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _p_item1_key_allKeys.extend(theseKeys)
                if len(_p_item1_key_allKeys):
                    p_item1_key.keys = _p_item1_key_allKeys[-1].name  # just the last key pressed
                    p_item1_key.rt = _p_item1_key_allKeys[-1].rt
                    p_item1_key.duration = _p_item1_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *p_item1_hint* updates
            
            # if p_item1_hint is starting this frame...
            if p_item1_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_item1_hint.frameNStart = frameN  # exact frame index
                p_item1_hint.tStart = t  # local t and not account for scr refresh
                p_item1_hint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_item1_hint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_item1_hint.started')
                # update status
                p_item1_hint.status = STARTED
                p_item1_hint.setAutoDraw(True)
            
            # if p_item1_hint is active this frame...
            if p_item1_hint.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=p_item1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_item1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_item1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_item1" ---
        for thisComponent in p_item1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_item1
        p_item1.tStop = globalClock.getTime(format='float')
        p_item1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_item1.stopped', p_item1.tStop)
        if p_item1_mrk.status == STARTED:
            win.callOnFlip(p_item1_mrk.setData, int(0))
        # check responses
        if p_item1_key.keys in ['', [], None]:  # No response was made
            p_item1_key.keys = None
        pract_byp.addData('p_item1_key.keys',p_item1_key.keys)
        if p_item1_key.keys != None:  # we had a response
            pract_byp.addData('p_item1_key.rt', p_item1_key.rt)
            pract_byp.addData('p_item1_key.duration', p_item1_key.duration)
        # the Routine "p_item1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "p_item2" ---
        # create an object to store info about Routine p_item2
        p_item2 = data.Routine(
            name='p_item2',
            components=[p_fix_mrk, p_fixation, p_item2_mrk, p_item2_img],
        )
        p_item2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for p_item2
        p_item2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_item2.tStart = globalClock.getTime(format='float')
        p_item2.status = STARTED
        thisExp.addData('p_item2.started', p_item2.tStart)
        p_item2.maxDuration = None
        # keep track of which components have finished
        p_item2Components = p_item2.components
        for thisComponent in p_item2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_item2" ---
        p_item2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPract_byp, 'status') and thisPract_byp.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *p_fix_mrk* updates
            
            # if p_fix_mrk is starting this frame...
            if p_fix_mrk.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                p_fix_mrk.frameNStart = frameN  # exact frame index
                p_fix_mrk.tStart = t  # local t and not account for scr refresh
                p_fix_mrk.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fix_mrk, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_fix_mrk.started', t)
                # update status
                p_fix_mrk.status = STARTED
                p_fix_mrk.status = STARTED
                win.callOnFlip(p_fix_mrk.setData, int(0))
            
            # if p_fix_mrk is stopping this frame...
            if p_fix_mrk.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fix_mrk.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fix_mrk.tStop = t  # not accounting for scr refresh
                    p_fix_mrk.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fix_mrk.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_fix_mrk.stopped', t)
                    # update status
                    p_fix_mrk.status = FINISHED
                    win.callOnFlip(p_fix_mrk.setData, int(0))
            
            # *p_fixation* updates
            
            # if p_fixation is starting this frame...
            if p_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_fixation.frameNStart = frameN  # exact frame index
                p_fixation.tStart = t  # local t and not account for scr refresh
                p_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation.started')
                # update status
                p_fixation.status = STARTED
                p_fixation.setAutoDraw(True)
            
            # if p_fixation is active this frame...
            if p_fixation.status == STARTED:
                # update params
                pass
            
            # if p_fixation is stopping this frame...
            if p_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fixation.tStartRefresh + 0-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fixation.tStop = t  # not accounting for scr refresh
                    p_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_fixation.stopped')
                    # update status
                    p_fixation.status = FINISHED
                    p_fixation.setAutoDraw(False)
            # *p_item2_mrk* updates
            
            # if p_item2_mrk is starting this frame...
            if p_item2_mrk.status == NOT_STARTED and p_fixation.status==FINISHED:
                # keep track of start time/frame for later
                p_item2_mrk.frameNStart = frameN  # exact frame index
                p_item2_mrk.tStart = t  # local t and not account for scr refresh
                p_item2_mrk.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_item2_mrk, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_item2_mrk.started', t)
                # update status
                p_item2_mrk.status = STARTED
                p_item2_mrk.status = STARTED
                win.callOnFlip(p_item2_mrk.setData, int(20))
            
            # if p_item2_mrk is stopping this frame...
            if p_item2_mrk.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_item2_mrk.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    p_item2_mrk.tStop = t  # not accounting for scr refresh
                    p_item2_mrk.tStopRefresh = tThisFlipGlobal  # on global time
                    p_item2_mrk.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_item2_mrk.stopped', t)
                    # update status
                    p_item2_mrk.status = FINISHED
                    win.callOnFlip(p_item2_mrk.setData, int(0))
            
            # *p_item2_img* updates
            
            # if p_item2_img is starting this frame...
            if p_item2_img.status == NOT_STARTED and p_fixation.status==FINISHED:
                # keep track of start time/frame for later
                p_item2_img.frameNStart = frameN  # exact frame index
                p_item2_img.tStart = t  # local t and not account for scr refresh
                p_item2_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_item2_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_item2_img.started')
                # update status
                p_item2_img.status = STARTED
                p_item2_img.setAutoDraw(True)
            
            # if p_item2_img is active this frame...
            if p_item2_img.status == STARTED:
                # update params
                pass
            
            # if p_item2_img is stopping this frame...
            if p_item2_img.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_item2_img.tStartRefresh + itemDur-frameTolerance:
                    # keep track of stop time/frame for later
                    p_item2_img.tStop = t  # not accounting for scr refresh
                    p_item2_img.tStopRefresh = tThisFlipGlobal  # on global time
                    p_item2_img.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_item2_img.stopped')
                    # update status
                    p_item2_img.status = FINISHED
                    p_item2_img.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=p_item2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_item2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_item2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_item2" ---
        for thisComponent in p_item2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_item2
        p_item2.tStop = globalClock.getTime(format='float')
        p_item2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_item2.stopped', p_item2.tStop)
        if p_fix_mrk.status == STARTED:
            win.callOnFlip(p_fix_mrk.setData, int(0))
        if p_item2_mrk.status == STARTED:
            win.callOnFlip(p_item2_mrk.setData, int(0))
        # the Routine "p_item2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        byp_p_idea_loop = data.TrialHandler2(
            name='byp_p_idea_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(byp_p_idea_loop)  # add the loop to the experiment
        thisByp_p_idea_loop = byp_p_idea_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisByp_p_idea_loop.rgb)
        if thisByp_p_idea_loop != None:
            for paramName in thisByp_p_idea_loop:
                globals()[paramName] = thisByp_p_idea_loop[paramName]
        
        for thisByp_p_idea_loop in byp_p_idea_loop:
            byp_p_idea_loop.status = STARTED
            if hasattr(thisByp_p_idea_loop, 'status'):
                thisByp_p_idea_loop.status = STARTED
            currentLoop = byp_p_idea_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisByp_p_idea_loop.rgb)
            if thisByp_p_idea_loop != None:
                for paramName in thisByp_p_idea_loop:
                    globals()[paramName] = thisByp_p_idea_loop[paramName]
            
            # set up handler to look after randomisation of conditions etc
            p_idea_loop = data.TrialHandler2(
                name='p_idea_loop',
                nReps=1, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(p_idea_loop)  # add the loop to the experiment
            thisP_idea_loop = p_idea_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisP_idea_loop.rgb)
            if thisP_idea_loop != None:
                for paramName in thisP_idea_loop:
                    globals()[paramName] = thisP_idea_loop[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisP_idea_loop in p_idea_loop:
                p_idea_loop.status = STARTED
                if hasattr(thisP_idea_loop, 'status'):
                    thisP_idea_loop.status = STARTED
                currentLoop = p_idea_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisP_idea_loop.rgb)
                if thisP_idea_loop != None:
                    for paramName in thisP_idea_loop:
                        globals()[paramName] = thisP_idea_loop[paramName]
                
                # --- Prepare to start Routine "p_idea" ---
                # create an object to store info about Routine p_idea
                p_idea = data.Routine(
                    name='p_idea',
                    components=[p_FZ],
                )
                p_idea.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from p_code
                ##  flags for spacekey  ##
                #p_spacedown = False
                #p_spaceup = True
                
                p_is_recording = False
                
                ##  Counter for recording enumeration
                p_recording_count = 1
                
                ##  get start-time for 'p-...'-routine
                p_startTime = core.getTime()
                
                # store start times for p_idea
                p_idea.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                p_idea.tStart = globalClock.getTime(format='float')
                p_idea.status = STARTED
                thisExp.addData('p_idea.started', p_idea.tStart)
                p_idea.maxDuration = None
                # keep track of which components have finished
                p_ideaComponents = p_idea.components
                for thisComponent in p_idea.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "p_idea" ---
                p_idea.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisP_idea_loop, 'status') and thisP_idea_loop.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from p_code
                    ##  check the loop-time  ##
                    if core.getTime() > p_startTime + p_loopDur:
                        p_spacedown = False
                        continueRoutine = False # finish p_trial
                    
                    ##  check for keypresses/-states  ##
                    p_response = p_kb.getKeys() # get a list of keys pressed at this instant
                    p_spacebar = p_kb.getState('space')
                    
                    ##  check for 'escape'
                    if p_response:
                        if 'escape' in p_response:
                            continueRoutine = False
                    
                    
                    ##  check the 'spacebar'-key responses  ##
                    ## check for presses
                    if p_spacebar:
                        
                        while p_spacebar: # and core.getTime() < p_startTime + p_loopDur:
                        #while is_recording and core.getTime() < p_startTime + p_loopDur:
                            time.sleep(0.01)                        # Small delay to prevent CPU overuse (from ai)
                            p_spacebar = p_kb.getState('space')
                    
                        ##toggle recording state
                        p_is_recording = not p_is_recording
                        
                        if p_is_recording:
                            p_FZ.color = 'green'
                            win.flip()
                    
                            ## start recording / record audio stream
                            p_audio_data = []
                    
                            #prev. ver: spacekey↓record-on>>spacekey↑-record-off # # # with sd.InputStream(samplerate=sample_rate, channels=channels, callback=lambda indata, frames, time, status: audio_data.append(indata.copy())):
                            p_stream = sd.InputStream(samplerate=sample_rate, channels=channels, callback=lambda indata, frames, time, status: p_audio_data.append(indata.copy()))
                            p_stream.start()
                            
                        # Stop the current recording
                        else:
                            p_FZ.color = 'white'
                            win.flip()
                            
                            p_stream.stop()
                            p_stream.close()
                            if p_audio_data:
                                save_recording(p_audio_data, p_recording_count, wavDirName, sample_rate)
                                p_recording_count += 1
                                p_audio_data = []   # Nach der Speicherung
                            else:
                                print("Recording too short - nothing saved")
                            
                    
                        # Check if we've reached the time limit
                        if core.getTime() > p_startTime + p_loopDur and p_is_recording:
                            # Stop recording if we hit the time limit during a recording
                            p_stream.stop()
                            p_stream.close()
                            if p_audio_data:
                                save_recording(p_audio_data, p_recording_count, recordings_dir, sample_rate)
                                p_recording_count += 1
                                p_audio_data = []   # Nach der Speicherung
                            print("Program time limit reached during recording")
                            
                        time.sleep(0.01)  # Small delay in main loop to prevent CPU overuse
                    
                    
                    ###   save_recording   ###
                    def save_recording(p_audio_data, p_recording_count, recordings_dir, sample_rate):
                        """Helper function to save a recording"""
                        if not p_audio_data:
                            return False
                        
                        # Convert list of numpy arrays into one large numpy array
                        p_recording = np.concatenate(p_audio_data, axis=0)
                        
                        ## prepare the recording
                        # ## Generate filename with enumeration and timestamp
                        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # timestamp = data.getDateStr(format="%Y-%m-%d_%H%M")   #local: set/mark start of current run
                        #audio_file = os.path.join(wavDirName, f"{expInfo['UPN-ID']}_idea{recording_count:02d}_{timestamp}.wav")    #name the recording with 'sub.-ID', timestamp' + 'rec._count'
                        p_audio_file = os.path.join(wavDirName, f"{expInfo['UPN-ID']}_idea{p_recording_count:02d}.wav")    #name the recording with 'sub.-ID', timestamp' + 'rec._count'
                        
                        # Save the recording to a file
                        sf.write(p_audio_file, p_recording, sample_rate)
                        print(f"Recording #{p_recording_count} saved to {p_audio_file}")
                        
                        return True
                    
                    
                    # *p_FZ* updates
                    
                    # if p_FZ is starting this frame...
                    if p_FZ.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        p_FZ.frameNStart = frameN  # exact frame index
                        p_FZ.tStart = t  # local t and not account for scr refresh
                        p_FZ.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(p_FZ, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'p_FZ.started')
                        # update status
                        p_FZ.status = STARTED
                        p_FZ.setAutoDraw(True)
                    
                    # if p_FZ is active this frame...
                    if p_FZ.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=p_idea,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        p_idea.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in p_idea.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "p_idea" ---
                for thisComponent in p_idea.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for p_idea
                p_idea.tStop = globalClock.getTime(format='float')
                p_idea.tStopRefresh = tThisFlipGlobal
                thisExp.addData('p_idea.stopped', p_idea.tStop)
                # the Routine "p_idea" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisP_idea_loop as finished
                if hasattr(thisP_idea_loop, 'status'):
                    thisP_idea_loop.status = FINISHED
                # if awaiting a pause, pause now
                if p_idea_loop.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    p_idea_loop.status = STARTED
                thisExp.nextEntry()
                
            # completed 1 repeats of 'p_idea_loop'
            p_idea_loop.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # mark thisByp_p_idea_loop as finished
            if hasattr(thisByp_p_idea_loop, 'status'):
                thisByp_p_idea_loop.status = FINISHED
            # if awaiting a pause, pause now
            if byp_p_idea_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                byp_p_idea_loop.status = STARTED
        # completed 1.0 repeats of 'byp_p_idea_loop'
        byp_p_idea_loop.status = FINISHED
        
        # mark thisPract_byp as finished
        if hasattr(thisPract_byp, 'status'):
            thisPract_byp.status = FINISHED
        # if awaiting a pause, pause now
        if pract_byp.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pract_byp.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'pract_byp'
    pract_byp.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "smry" ---
    # create an object to store info about Routine smry
    smry = data.Routine(
        name='smry',
        components=[smry_img, smry_key, smry_hint],
    )
    smry.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for smry_key
    smry_key.keys = []
    smry_key.rt = []
    _smry_key_allKeys = []
    # store start times for smry
    smry.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    smry.tStart = globalClock.getTime(format='float')
    smry.status = STARTED
    thisExp.addData('smry.started', smry.tStart)
    smry.maxDuration = None
    # keep track of which components have finished
    smryComponents = smry.components
    for thisComponent in smry.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "smry" ---
    smry.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *smry_img* updates
        
        # if smry_img is starting this frame...
        if smry_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            smry_img.frameNStart = frameN  # exact frame index
            smry_img.tStart = t  # local t and not account for scr refresh
            smry_img.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(smry_img, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'smry_img.started')
            # update status
            smry_img.status = STARTED
            smry_img.setAutoDraw(True)
        
        # if smry_img is active this frame...
        if smry_img.status == STARTED:
            # update params
            pass
        
        # *smry_key* updates
        waitOnFlip = False
        
        # if smry_key is starting this frame...
        if smry_key.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            smry_key.frameNStart = frameN  # exact frame index
            smry_key.tStart = t  # local t and not account for scr refresh
            smry_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(smry_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'smry_key.started')
            # update status
            smry_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(smry_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(smry_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if smry_key.status == STARTED and not waitOnFlip:
            theseKeys = smry_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
            _smry_key_allKeys.extend(theseKeys)
            if len(_smry_key_allKeys):
                smry_key.keys = _smry_key_allKeys[-1].name  # just the last key pressed
                smry_key.rt = _smry_key_allKeys[-1].rt
                smry_key.duration = _smry_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *smry_hint* updates
        
        # if smry_hint is starting this frame...
        if smry_hint.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            smry_hint.frameNStart = frameN  # exact frame index
            smry_hint.tStart = t  # local t and not account for scr refresh
            smry_hint.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(smry_hint, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'smry_hint.started')
            # update status
            smry_hint.status = STARTED
            smry_hint.setAutoDraw(True)
        
        # if smry_hint is active this frame...
        if smry_hint.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=smry,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            smry.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in smry.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "smry" ---
    for thisComponent in smry.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for smry
    smry.tStop = globalClock.getTime(format='float')
    smry.tStopRefresh = tThisFlipGlobal
    thisExp.addData('smry.stopped', smry.tStop)
    # check responses
    if smry_key.keys in ['', [], None]:  # No response was made
        smry_key.keys = None
    thisExp.addData('smry_key.keys',smry_key.keys)
    if smry_key.keys != None:  # we had a response
        thisExp.addData('smry_key.rt', smry_key.rt)
        thisExp.addData('smry_key.duration', smry_key.duration)
    thisExp.nextEntry()
    # the Routine "smry" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "start" ---
    # create an object to store info about Routine start
    start = data.Routine(
        name='start',
        components=[start_txt, start_key],
    )
    start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_key
    start_key.keys = []
    start_key.rt = []
    _start_key_allKeys = []
    # store start times for start
    start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start.tStart = globalClock.getTime(format='float')
    start.status = STARTED
    thisExp.addData('start.started', start.tStart)
    start.maxDuration = None
    # keep track of which components have finished
    startComponents = start.components
    for thisComponent in start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start" ---
    start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_txt* updates
        
        # if start_txt is starting this frame...
        if start_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_txt.frameNStart = frameN  # exact frame index
            start_txt.tStart = t  # local t and not account for scr refresh
            start_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_txt.started')
            # update status
            start_txt.status = STARTED
            start_txt.setAutoDraw(True)
        
        # if start_txt is active this frame...
        if start_txt.status == STARTED:
            # update params
            pass
        
        # *start_key* updates
        waitOnFlip = False
        
        # if start_key is starting this frame...
        if start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_key.frameNStart = frameN  # exact frame index
            start_key.tStart = t  # local t and not account for scr refresh
            start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_key.started')
            # update status
            start_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_key.status == STARTED and not waitOnFlip:
            theseKeys = start_key.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _start_key_allKeys.extend(theseKeys)
            if len(_start_key_allKeys):
                start_key.keys = _start_key_allKeys[-1].name  # just the last key pressed
                start_key.rt = _start_key_allKeys[-1].rt
                start_key.duration = _start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=start,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start" ---
    for thisComponent in start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start
    start.tStop = globalClock.getTime(format='float')
    start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start.stopped', start.tStop)
    # check responses
    if start_key.keys in ['', [], None]:  # No response was made
        start_key.keys = None
    thisExp.addData('start_key.keys',start_key.keys)
    if start_key.keys != None:  # we had a response
        thisExp.addData('start_key.rt', start_key.rt)
        thisExp.addData('start_key.duration', start_key.duration)
    thisExp.nextEntry()
    # the Routine "start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('stim/CreaLying_itemlist.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "t_item" ---
        # create an object to store info about Routine t_item
        t_item = data.Routine(
            name='t_item',
            components=[t_fix_mrk, t_fixation, t_item_mrk, t_item_img],
        )
        t_item.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        t_item_img.setImage('stim\\items\\%s.jpg' % (all_items[trials.thisN][itm]))
        # store start times for t_item
        t_item.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        t_item.tStart = globalClock.getTime(format='float')
        t_item.status = STARTED
        thisExp.addData('t_item.started', t_item.tStart)
        t_item.maxDuration = None
        # keep track of which components have finished
        t_itemComponents = t_item.components
        for thisComponent in t_item.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "t_item" ---
        t_item.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *t_fix_mrk* updates
            
            # if t_fix_mrk is starting this frame...
            if t_fix_mrk.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                t_fix_mrk.frameNStart = frameN  # exact frame index
                t_fix_mrk.tStart = t  # local t and not account for scr refresh
                t_fix_mrk.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_fix_mrk, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('t_fix_mrk.started', t)
                # update status
                t_fix_mrk.status = STARTED
                t_fix_mrk.status = STARTED
                win.callOnFlip(t_fix_mrk.setData, int(10))
            
            # if t_fix_mrk is stopping this frame...
            if t_fix_mrk.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_fix_mrk.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    t_fix_mrk.tStop = t  # not accounting for scr refresh
                    t_fix_mrk.tStopRefresh = tThisFlipGlobal  # on global time
                    t_fix_mrk.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('t_fix_mrk.stopped', t)
                    # update status
                    t_fix_mrk.status = FINISHED
                    win.callOnFlip(t_fix_mrk.setData, int(0))
            
            # *t_fixation* updates
            
            # if t_fixation is starting this frame...
            if t_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_fixation.frameNStart = frameN  # exact frame index
                t_fixation.tStart = t  # local t and not account for scr refresh
                t_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_fixation.started')
                # update status
                t_fixation.status = STARTED
                t_fixation.setAutoDraw(True)
            
            # if t_fixation is active this frame...
            if t_fixation.status == STARTED:
                # update params
                pass
            
            # if t_fixation is stopping this frame...
            if t_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_fixation.tStartRefresh + fixDur-frameTolerance:
                    # keep track of stop time/frame for later
                    t_fixation.tStop = t  # not accounting for scr refresh
                    t_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    t_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_fixation.stopped')
                    # update status
                    t_fixation.status = FINISHED
                    t_fixation.setAutoDraw(False)
            # *t_item_mrk* updates
            
            # if t_item_mrk is starting this frame...
            if t_item_mrk.status == NOT_STARTED and t_fixation.status==FINISHED:
                # keep track of start time/frame for later
                t_item_mrk.frameNStart = frameN  # exact frame index
                t_item_mrk.tStart = t  # local t and not account for scr refresh
                t_item_mrk.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_item_mrk, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('t_item_mrk.started', t)
                # update status
                t_item_mrk.status = STARTED
                t_item_mrk.status = STARTED
                win.callOnFlip(t_item_mrk.setData, int(all_items[trials.thisN][mrk]))
            
            # if t_item_mrk is stopping this frame...
            if t_item_mrk.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_item_mrk.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    t_item_mrk.tStop = t  # not accounting for scr refresh
                    t_item_mrk.tStopRefresh = tThisFlipGlobal  # on global time
                    t_item_mrk.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('t_item_mrk.stopped', t)
                    # update status
                    t_item_mrk.status = FINISHED
                    win.callOnFlip(t_item_mrk.setData, int(0))
            
            # *t_item_img* updates
            
            # if t_item_img is starting this frame...
            if t_item_img.status == NOT_STARTED and t_fixation.status==FINISHED:
                # keep track of start time/frame for later
                t_item_img.frameNStart = frameN  # exact frame index
                t_item_img.tStart = t  # local t and not account for scr refresh
                t_item_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_item_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_item_img.started')
                # update status
                t_item_img.status = STARTED
                t_item_img.setAutoDraw(True)
            
            # if t_item_img is active this frame...
            if t_item_img.status == STARTED:
                # update params
                pass
            
            # if t_item_img is stopping this frame...
            if t_item_img.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_item_img.tStartRefresh + itemDur-frameTolerance:
                    # keep track of stop time/frame for later
                    t_item_img.tStop = t  # not accounting for scr refresh
                    t_item_img.tStopRefresh = tThisFlipGlobal  # on global time
                    t_item_img.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_item_img.stopped')
                    # update status
                    t_item_img.status = FINISHED
                    t_item_img.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=t_item,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                t_item.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in t_item.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "t_item" ---
        for thisComponent in t_item.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for t_item
        t_item.tStop = globalClock.getTime(format='float')
        t_item.tStopRefresh = tThisFlipGlobal
        thisExp.addData('t_item.stopped', t_item.tStop)
        if t_fix_mrk.status == STARTED:
            win.callOnFlip(t_fix_mrk.setData, int(0))
        if t_item_mrk.status == STARTED:
            win.callOnFlip(t_item_mrk.setData, int(0))
        # the Routine "t_item" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        item_loop = data.TrialHandler2(
            name='item_loop',
            nReps=1, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(item_loop)  # add the loop to the experiment
        thisItem_loop = item_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisItem_loop.rgb)
        if thisItem_loop != None:
            for paramName in thisItem_loop:
                globals()[paramName] = thisItem_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisItem_loop in item_loop:
            item_loop.status = STARTED
            if hasattr(thisItem_loop, 'status'):
                thisItem_loop.status = STARTED
            currentLoop = item_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisItem_loop.rgb)
            if thisItem_loop != None:
                for paramName in thisItem_loop:
                    globals()[paramName] = thisItem_loop[paramName]
            
            # --- Prepare to start Routine "t_idea" ---
            # create an object to store info about Routine t_idea
            t_idea = data.Routine(
                name='t_idea',
                components=[t_FZ],
            )
            t_idea.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from t_code
            ##  flags for spacekey  ##
            #t_spacedown = False
            #t_spaceup = True
            
            is_recording = False
            
            ##  Counter for recording enumeration
            recording_count = 1
            
            ##  get start-time for 't-...'-routine
            t_startTime = core.getTime()
            
            t_FZ.color = 'white'
            win.flip()
            
            ##t_mrk: START Nachdenkphase
            t_mrk.setData(50); core.wait(0.1); t_mrk.setData(0)
            
            
            ## Vor dem ersten Aufruf von getKeys() und getState() in "each frame"
            t_kb.clearEvents(eventType='keyboard')  # Lösche alle vorherigen Tasteneingaben  
            # store start times for t_idea
            t_idea.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            t_idea.tStart = globalClock.getTime(format='float')
            t_idea.status = STARTED
            thisExp.addData('t_idea.started', t_idea.tStart)
            t_idea.maxDuration = None
            # keep track of which components have finished
            t_ideaComponents = t_idea.components
            for thisComponent in t_idea.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "t_idea" ---
            t_idea.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisItem_loop, 'status') and thisItem_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from t_code
                ##  check the loop-time  ##
                if core.getTime() > t_startTime + loopDur:
                    ##t_mrk: STOP
                    t_mrk.setData(49); core.wait(0.1); t_mrk.setData(0)
                    save_recording(audio_data, recording_count, wavDirName, sample_rate)
                    is_recording = False
                    continueRoutine = False # finish p_trial
                
                ##  check for keypresses/-states  ##
                t_response = t_kb.getKeys() # get a list of keys pressed at this instant
                t_spacebar = t_kb.getState('space')
                
                ##  check for 'escape'
                if t_response:
                    if 'escape' in t_response:
                        continueRoutine = False
                
                
                ##  check the 'spacebar'-key responses  ##
                ## check for presses
                if t_spacebar:
                    
                    while t_spacebar: # and core.getTime() < t_startTime + t_loopDur:
                    #while is_recording and core.getTime() < t_startTime + t_loopDur:
                        time.sleep(0.01)                        # Small delay to prevent CPU overuse (from ai)
                        t_spacebar = t_kb.getState('space')
                
                    ##toggle recording state
                    is_recording = not is_recording
                    
                    if is_recording:
                        ##t_mrk: START
                        t_mrk.setData(48); core.wait(0.1); t_mrk.setData(0)
                
                        ## start recording / record audio stream
                        audio_data = []
                
                        #prev. ver: spacekey↓record-on>>spacekey↑-record-off # # # with sd.InputStream(samplerate=sample_rate, channels=channels, callback=lambda indata, frames, time, status: audio_data.append(indata.copy())):
                        stream = sd.InputStream(samplerate=sample_rate, channels=channels, callback=lambda indata, frames, time, status: audio_data.append(indata.copy()))
                        stream.start()
                        
                        ##after recording started, flip to green '?'
                        t_FZ.color = 'green'
                        win.flip()
                        
                    # Stop the current recording
                    else:
                        ##before recording stops, bakc to white '?'
                        t_FZ.color = 'white'
                        win.flip()
                        
                        ## stop recording / stop audio stream
                        stream.stop()
                        stream.close()
                        if audio_data:
                            save_recording(audio_data, recording_count, wavDirName, sample_rate)
                            recording_count += 1
                            audio_data = []   # Nach der Speicherung
                        else:
                            print("Recording too short - nothing saved")
                        ##t_mrk: STOP
                        t_mrk.setData(49); core.wait(0.1); t_mrk.setData(0)
                        
                
                    # Check if we've reached the time limit
                    if core.getTime() > t_startTime + loopDur and is_recording:
                        # Stop recording if we hit the time limit during a recording
                        stream.stop()
                        stream.close()
                        if audio_data:
                            save_recording(audio_data, recording_count, wavDirName, sample_rate)
                            recording_count += 1
                            audio_data = []   # Nach der Speicherung
                        ##t_mrk: STOP
                        t_mrk.setData(49); core.wait(0.1); t_mrk.setData(0)
                        
                        print("Program time limit reached during recording")
                        
                    time.sleep(0.01)  # Small delay in main loop to prevent CPU overuse
                
                
                ###   save_recording   ###
                def save_recording(audio_data, recording_count, recordings_dir, sample_rate):
                    """Helper function to save a recording"""
                    if not audio_data:
                        return False
                    
                    # Convert list of numpy arrays into one large numpy array
                    recording = np.concatenate(audio_data, axis=0)
                    
                    ## prepare the recording
                    # ## Generate filename with enumeration and timestamp
                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # timestamp = data.getDateStr(format="%Y-%m-%d_%H%M")   #local: set/mark start of current run
                    #audio_file = os.path.join(wavDirName, f"{expInfo['UPN-ID']}_idea{recording_count:02d}_{timestamp}.wav")    #name the recording with 'sub.-ID', timestamp' + 'rec._count'
                    audio_file = os.path.join(wavDirName, f"{expInfo['UPN-ID']}_{all_items[trials.thisN][0]}_idea{recording_count:02d}.wav")    #name the recording with 'sub.-ID', timestamp' + 'rec._count'
                    
                    # Save the recording to a file
                    sf.write(audio_file, recording, sample_rate)
                    print(f"Recording #{recording_count} saved to {audio_file}")
                    return True
                
                
                #################################################################################################################################################################
                ##    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###   ##
                #################################################################################################################################################################
                
                """
                ## ### check the loop-time ###
                if core.getTime() > t_startTime + loopDur:
                    continueRoutine = False # finish p_trial
                
                ## ### check for keypresses ###
                t_response = t_kb.getKeys() # get a list of keys pressed at this instant
                t_spacebar = t_kb.getState('space')
                
                ## check for 'escape'
                if t_response:
                    if 'escape' in t_response:
                        continueRoutine = False
                
                
                ##  check the 'spacebar'-key responses  ##
                ## check for presses
                if t_spacebar and not t_spacedown:
                    if not t_spacedown:
                        t_spacedown = True
                        t_spaceup = False
                        t_FZ.color = 'green'
                        win.flip()
                    
                    ## prepare the recording
                    # timestamp = data.getDateStr(format="%Y-%m-%d_%H%M")   #local: set/mark start of current run
                    #audio_file = os.path.join(wavDirName, f"{expInfo['UPN-ID']}_idea{recording_count:02d}_{timestamp}.wav")    #name the recording with 'sub.-ID', timestamp' + 'rec._count'
                    audio_file = os.path.join(wavDirName, f"{expInfo['UPN-ID']}_{all_items[trials.thisN][0]}_idea{recording_count:02d}.wav")    #name the recording with 'sub.-ID', timestamp' + 'rec._count'
                    
                    ## start recording
                    audio_data = []
                    
                    # Record audio stream
                    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=lambda indata, frames, time, status: audio_data.append(indata.copy())):
                        ## Continue recording until spacebar is released or loop-time is over
                        while t_spacebar and core.getTime() < t_startTime + loopDur:
                            time.sleep(0.01)  # Small delay to prevent CPU overuse
                            t_spacebar = p_kb.getState('space')
                    
                    if audio_data:
                        ## Convert list of numpy arrays into one large numpy array
                        recording = np.concatenate(audio_data, axis=0)
                        
                        ## Save the recording to a file
                        sf.write(audio_file, recording, sample_rate)
                        recording_count += 1  # Increment the recording counter
                    else:
                        print("Recording too short - nothing saved")
                
                
                ## check for releases
                elif not t_spacebar and not t_spaceup:
                    t_spacedown = False
                    t_spaceup = True
                    t_FZ.color = 'white'
                    win.flip()
                """
                
                
                # *t_FZ* updates
                
                # if t_FZ is starting this frame...
                if t_FZ.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    t_FZ.frameNStart = frameN  # exact frame index
                    t_FZ.tStart = t  # local t and not account for scr refresh
                    t_FZ.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_FZ, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_FZ.started')
                    # update status
                    t_FZ.status = STARTED
                    t_FZ.setAutoDraw(True)
                
                # if t_FZ is active this frame...
                if t_FZ.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=t_idea,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    t_idea.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in t_idea.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "t_idea" ---
            for thisComponent in t_idea.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for t_idea
            t_idea.tStop = globalClock.getTime(format='float')
            t_idea.tStopRefresh = tThisFlipGlobal
            thisExp.addData('t_idea.stopped', t_idea.tStop)
            # the Routine "t_idea" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisItem_loop as finished
            if hasattr(thisItem_loop, 'status'):
                thisItem_loop.status = FINISHED
            # if awaiting a pause, pause now
            if item_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                item_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1 repeats of 'item_loop'
        item_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        rf_loop = data.TrialHandler2(
            name='rf_loop',
            nReps=show_rf, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(rf_loop)  # add the loop to the experiment
        thisRf_loop = rf_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRf_loop.rgb)
        if thisRf_loop != None:
            for paramName in thisRf_loop:
                globals()[paramName] = thisRf_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisRf_loop in rf_loop:
            rf_loop.status = STARTED
            if hasattr(thisRf_loop, 'status'):
                thisRf_loop.status = STARTED
            currentLoop = rf_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisRf_loop.rgb)
            if thisRf_loop != None:
                for paramName in thisRf_loop:
                    globals()[paramName] = thisRf_loop[paramName]
            
            # --- Prepare to start Routine "ruhe_fixation" ---
            # create an object to store info about Routine ruhe_fixation
            ruhe_fixation = data.Routine(
                name='ruhe_fixation',
                components=[rf_fixation],
            )
            ruhe_fixation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for ruhe_fixation
            ruhe_fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ruhe_fixation.tStart = globalClock.getTime(format='float')
            ruhe_fixation.status = STARTED
            thisExp.addData('ruhe_fixation.started', ruhe_fixation.tStart)
            ruhe_fixation.maxDuration = None
            # keep track of which components have finished
            ruhe_fixationComponents = ruhe_fixation.components
            for thisComponent in ruhe_fixation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "ruhe_fixation" ---
            ruhe_fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisRf_loop, 'status') and thisRf_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *rf_fixation* updates
                
                # if rf_fixation is starting this frame...
                if rf_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    rf_fixation.frameNStart = frameN  # exact frame index
                    rf_fixation.tStart = t  # local t and not account for scr refresh
                    rf_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rf_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rf_fixation.started')
                    # update status
                    rf_fixation.status = STARTED
                    rf_fixation.setAutoDraw(True)
                
                # if rf_fixation is active this frame...
                if rf_fixation.status == STARTED:
                    # update params
                    pass
                
                # if rf_fixation is stopping this frame...
                if rf_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > rf_fixation.tStartRefresh + rfDur-frameTolerance:
                        # keep track of stop time/frame for later
                        rf_fixation.tStop = t  # not accounting for scr refresh
                        rf_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        rf_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rf_fixation.stopped')
                        # update status
                        rf_fixation.status = FINISHED
                        rf_fixation.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=ruhe_fixation,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    ruhe_fixation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ruhe_fixation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ruhe_fixation" ---
            for thisComponent in ruhe_fixation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ruhe_fixation
            ruhe_fixation.tStop = globalClock.getTime(format='float')
            ruhe_fixation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('ruhe_fixation.stopped', ruhe_fixation.tStop)
            # the Routine "ruhe_fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisRf_loop as finished
            if hasattr(thisRf_loop, 'status'):
                thisRf_loop.status = FINISHED
            # if awaiting a pause, pause now
            if rf_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                rf_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed show_rf repeats of 'rf_loop'
        rf_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
    # completed 1 repeats of 'trials'
    trials.status = FINISHED
    
    
    # --- Prepare to start Routine "thx" ---
    # create an object to store info about Routine thx
    thx = data.Routine(
        name='thx',
        components=[thx_txt, thx_key, thx_hint],
    )
    thx.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for thx_key
    thx_key.keys = []
    thx_key.rt = []
    _thx_key_allKeys = []
    # store start times for thx
    thx.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thx.tStart = globalClock.getTime(format='float')
    thx.status = STARTED
    thisExp.addData('thx.started', thx.tStart)
    thx.maxDuration = None
    # keep track of which components have finished
    thxComponents = thx.components
    for thisComponent in thx.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thx" ---
    thx.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thx_txt* updates
        
        # if thx_txt is starting this frame...
        if thx_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thx_txt.frameNStart = frameN  # exact frame index
            thx_txt.tStart = t  # local t and not account for scr refresh
            thx_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thx_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thx_txt.started')
            # update status
            thx_txt.status = STARTED
            thx_txt.setAutoDraw(True)
        
        # if thx_txt is active this frame...
        if thx_txt.status == STARTED:
            # update params
            pass
        
        # if thx_txt is stopping this frame...
        if thx_txt.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thx_txt.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                thx_txt.tStop = t  # not accounting for scr refresh
                thx_txt.tStopRefresh = tThisFlipGlobal  # on global time
                thx_txt.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_txt.stopped')
                # update status
                thx_txt.status = FINISHED
                thx_txt.setAutoDraw(False)
        
        # *thx_key* updates
        waitOnFlip = False
        
        # if thx_key is starting this frame...
        if thx_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thx_key.frameNStart = frameN  # exact frame index
            thx_key.tStart = t  # local t and not account for scr refresh
            thx_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thx_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thx_key.started')
            # update status
            thx_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(thx_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(thx_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if thx_key is stopping this frame...
        if thx_key.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thx_key.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                thx_key.tStop = t  # not accounting for scr refresh
                thx_key.tStopRefresh = tThisFlipGlobal  # on global time
                thx_key.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_key.stopped')
                # update status
                thx_key.status = FINISHED
                thx_key.status = FINISHED
        if thx_key.status == STARTED and not waitOnFlip:
            theseKeys = thx_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
            _thx_key_allKeys.extend(theseKeys)
            if len(_thx_key_allKeys):
                thx_key.keys = _thx_key_allKeys[-1].name  # just the last key pressed
                thx_key.rt = _thx_key_allKeys[-1].rt
                thx_key.duration = _thx_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *thx_hint* updates
        
        # if thx_hint is starting this frame...
        if thx_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thx_hint.frameNStart = frameN  # exact frame index
            thx_hint.tStart = t  # local t and not account for scr refresh
            thx_hint.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thx_hint, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thx_hint.started')
            # update status
            thx_hint.status = STARTED
            thx_hint.setAutoDraw(True)
        
        # if thx_hint is active this frame...
        if thx_hint.status == STARTED:
            # update params
            pass
        
        # if thx_hint is stopping this frame...
        if thx_hint.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thx_hint.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                thx_hint.tStop = t  # not accounting for scr refresh
                thx_hint.tStopRefresh = tThisFlipGlobal  # on global time
                thx_hint.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_hint.stopped')
                # update status
                thx_hint.status = FINISHED
                thx_hint.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thx,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thx.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thx.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thx" ---
    for thisComponent in thx.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thx
    thx.tStop = globalClock.getTime(format='float')
    thx.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thx.stopped', thx.tStop)
    # check responses
    if thx_key.keys in ['', [], None]:  # No response was made
        thx_key.keys = None
    thisExp.addData('thx_key.keys',thx_key.keys)
    if thx_key.keys != None:  # we had a response
        thisExp.addData('thx_key.rt', thx_key.rt)
        thisExp.addData('thx_key.duration', thx_key.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thx.maxDurationReached:
        routineTimer.addTime(-thx.maxDuration)
    elif thx.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

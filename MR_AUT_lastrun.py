#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Oktober 02, 2025, at 10:53
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
from psychopy_bids.bids import BIDSBehEvent
from psychopy_bids.bids import BIDSTaskEvent
from psychopy_bids.bids import BIDSError
from psychopy_bids.bids import BIDSHandler

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'MR_AUT'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '999',
    'session': '77',
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
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\weberbe\\Nextcloud\\__PsychoPy\\MR_AUT_test\\MR_AUT_lastrun.py',
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
        logging.console.setLevel('exp')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
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
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='bw@office', color=[-1,-1,-1], colorSpace='rgb',
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
    if deviceManager.getDevice('GOfromMRT_key') is None:
        # initialise GOfromMRT_key
        GOfromMRT_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='GOfromMRT_key',
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
    if deviceManager.getDevice('idea_key') is None:
        # initialise idea_key
        idea_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='idea_key',
        )
    if deviceManager.getDevice('r_end_key') is None:
        # initialise r_end_key
        r_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='r_end_key',
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
    
    wavDirName = '%s_%s_%s_%s_%s' % (os.path.join(_thisDir, 'data', ''), expInfo['participant'], expName, data.getDateStr(format="%Y-%m-%d_%H%M"), '_wav')
    if not os.path.isdir(wavDirName):
        os.makedirs(wavDirName)  # to hold .wav files
    #print(f"{wavDirName=}")
    
    ## ### Options for DEBUG/RESARCH-mode ###
    DEBUG = 1 #0=research-mode; 1=debug-mode 
    if DEBUG: 
        itemDur = 5
    else: 
        itemDur = 15
    
    header_pos_y = .40
    main_pos_y   = 0
    btn_pos_y    = -.25
    
    
    # --- Initialize components for Routine "wait4scanner" ---
    GOfromMRT_txt = visual.TextStim(win=win, name='GOfromMRT_txt',
        text="...awaiting clearance ('5') from the MRI-Scanner...",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    GOfromMRT_key = keyboard.Keyboard(deviceName='GOfromMRT_key')
    
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
    
    # --- Initialize components for Routine "t_item" ---
    t_fix_mrk = parallel.ParallelPort(address='0x3FF8')
    t_fixation = visual.TextStim(win=win, name='t_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    t_item_mrk = parallel.ParallelPort(address='0x3FF8')
    AUTitem = visual.TextStim(win=win, name='AUTitem',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    idea_key = keyboard.Keyboard(deviceName='idea_key')
    idea_mse = event.Mouse(win=win)
    x, y = [None, None]
    idea_mse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "t_idea" ---
    # Run 'Begin Experiment' code from t_code
    import sounddevice as sd
    import soundfile as sf
    import time
    
    ## set audio recording parameters
    sample_rate = 44100  # Sample rate (Hz)
    channels = 1         # Number of audio channels (2 for stereo)
    
    ## get/access the keyboard
    t_kb = keyboard.Keyboard()
    
    ### safe stop stream  ##
    def safe_stop_stream(stream):                              # lukas
        ###Stop and close an input stream safely.###           # lukas
        if stream is not None:                                 # lukas
            try:                                               # lukas
                stream.stop()                                  # lukas: safe stop
            finally:                                           # lukas
                stream.close()                                 # lukas: safe close
        return None                                            # lukas: clear reference
    
    
    ###   save_recording   ###
    def save_recording(audio_data, recording_count, recordings_dir, sample_rate):
        ###Helper function to save a recording###
        if not audio_data:
            print("No audio data to save.")
            return False
        
        # Convert list of numpy arrays into one large numpy array
        recording = np.concatenate(audio_data, axis=0)
        
        ## prepare the recording
        # ## Generate filename with enumeration and timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # timestamp = data.getDateStr(format="%Y-%m-%d_%H%M")   #local: set/mark start of current run
        #audio_file = os.path.join(wavDirName, f"{expInfo['Subject-ID']}_idea{recording_count:02d}_{timestamp}.wav")    #name the recording with 'sub.-ID', timestamp' + 'rec._count'
        #audio_file = os.path.join(wavDirName, f"{expInfo['Subject-ID']}_{RITitems[trials.thisN][itm]}_idea{recording_count:02d}.wav")    #name the recording with 'sub.-ID', 'rec._count'
        #audio_file = os.path.join(wavDirName, f"{expInfo['Subject-ID']}_{MR_AUT_items}_idea{recording_count:02d}.wav")    #name the recording with 'sub.-ID', 'rec._count'
        audio_file = os.path.join(wavDirName, f"{expInfo['participant']}_{MR_AUTitem}.wav")    #name the recording with 'participant'/'sub.-ID'
        
        try:
            # Save the recording to a file
            sf.write(audio_file, recording, sample_rate)
            #print(f"Recording #{recording_count} saved to {audio_file}")
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False
            
        return True
    t_FZ = visual.TextStim(win=win, name='t_FZ',
        text='?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='green', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "AUT_rating" ---
    r_header_txt = visual.TextStim(win=win, name='r_header_txt',
        text='Wie kreativ findest Du Deine Antwort',
        font='Arial',
        pos=(0, header_pos_y), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    VASrating = visual.Slider(win=win, name='VASrating',
        startValue=5, size=(1.0, 0.03), pos=(0, main_pos_y), units=win.units,
        labels=None, ticks=[0,1,2,3,4,5,6,7,8,9,10], granularity=0,
        style=['slider'], styleTweaks=(), opacity=1,
        labelColor='black', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.7035,
        flip=False, ori=0, depth=-2, readOnly=False)
    r_end_key = keyboard.Keyboard(deviceName='r_end_key')
    r1_mouse = event.Mouse(win=win)
    x, y = [None, None]
    r1_mouse.mouseClock = core.Clock()
    l_label = visual.TextStim(win=win, name='l_label',
        text='gar nicht kreativ',
        font='Arial',
        pos=(-.5, main_pos_y - .075), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    r_label = visual.TextStim(win=win, name='r_label',
        text='sehr kreativ',
        font='Arial',
        pos=(.5, main_pos_y - .075), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    r1_btn = visual.Rect(
        win=win, name='r1_btn',
        width=(0.125, 0.0475)[0], height=(0.125, 0.0475)[1],
        ori=0, pos=(0, btn_pos_y), draggable=False, anchor='center',
        lineWidth=1,
        colorSpace='rgb', lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1, depth=-7.0, interpolate=True)
    rating_hint = visual.TextStim(win=win, name='rating_hint',
        text='Weiter',
        font='Arial',
        pos=(0, btn_pos_y), draggable=False, height=0.025, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "insight2" ---
    qu_header2_txt = visual.TextStim(win=win, name='qu_header2_txt',
        text='Hatten Sie ein Gefühl der Einsicht ("Ahhh!"-Moment)?',
        font='Arial',
        pos=(0, header_pos_y), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    moi = visual.Slider(win=win, name='moi',
        startValue=None, size=(0.5, 0.025), pos=(0, 0.1), units=win.units,
        labels=["Nein", "Weiß nicht", "Ja"],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='darkgreen', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.025,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    iihs_txt = visual.TextStim(win=win, name='iihs_txt',
        text='Wenn JA, wie stark war dieses Gefühl:',
        font='Arial',
        pos=(0, -0.09), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    iihs = visual.Slider(win=win, name='iihs',
        startValue=None, size=(1.0, 0.025), pos=(0, -0.15), units=win.units,
        labels=["sehr\nschwach", "sehr\nstark"], ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='darkgreen', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.025,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    insight_btn = visual.Rect(
        win=win, name='insight_btn',
        width=(0.15, 0.045)[0], height=(0.15, 0.045)[1],
        ori=0.0, pos=(0, -.4), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    insight_btn_txt = visual.TextStim(win=win, name='insight_btn_txt',
        text='Weiter',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    insight_mse = event.Mouse(win=win)
    x, y = [None, None]
    insight_mse.mouseClock = core.Clock()
    
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
    wait4scanner_byp = data.TrialHandler2(
        name='wait4scanner_byp',
        nReps=0.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(wait4scanner_byp)  # add the loop to the experiment
    thisWait4scanner_byp = wait4scanner_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisWait4scanner_byp.rgb)
    if thisWait4scanner_byp != None:
        for paramName in thisWait4scanner_byp:
            globals()[paramName] = thisWait4scanner_byp[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisWait4scanner_byp in wait4scanner_byp:
        wait4scanner_byp.status = STARTED
        if hasattr(thisWait4scanner_byp, 'status'):
            thisWait4scanner_byp.status = STARTED
        currentLoop = wait4scanner_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisWait4scanner_byp.rgb)
        if thisWait4scanner_byp != None:
            for paramName in thisWait4scanner_byp:
                globals()[paramName] = thisWait4scanner_byp[paramName]
        
        # --- Prepare to start Routine "wait4scanner" ---
        # create an object to store info about Routine wait4scanner
        wait4scanner = data.Routine(
            name='wait4scanner',
            components=[GOfromMRT_txt, GOfromMRT_key],
        )
        wait4scanner.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for GOfromMRT_key
        GOfromMRT_key.keys = []
        GOfromMRT_key.rt = []
        _GOfromMRT_key_allKeys = []
        # store start times for wait4scanner
        wait4scanner.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wait4scanner.tStart = globalClock.getTime(format='float')
        wait4scanner.status = STARTED
        thisExp.addData('wait4scanner.started', wait4scanner.tStart)
        wait4scanner.maxDuration = None
        # keep track of which components have finished
        wait4scannerComponents = wait4scanner.components
        for thisComponent in wait4scanner.components:
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
        
        # --- Run Routine "wait4scanner" ---
        wait4scanner.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisWait4scanner_byp, 'status') and thisWait4scanner_byp.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *GOfromMRT_txt* updates
            
            # if GOfromMRT_txt is starting this frame...
            if GOfromMRT_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                GOfromMRT_txt.frameNStart = frameN  # exact frame index
                GOfromMRT_txt.tStart = t  # local t and not account for scr refresh
                GOfromMRT_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(GOfromMRT_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'GOfromMRT_txt.started')
                # update status
                GOfromMRT_txt.status = STARTED
                GOfromMRT_txt.setAutoDraw(True)
            
            # if GOfromMRT_txt is active this frame...
            if GOfromMRT_txt.status == STARTED:
                # update params
                GOfromMRT_txt.setOpacity(sin(3*t)+0.25, log=False)
            
            # *GOfromMRT_key* updates
            waitOnFlip = False
            
            # if GOfromMRT_key is starting this frame...
            if GOfromMRT_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                GOfromMRT_key.frameNStart = frameN  # exact frame index
                GOfromMRT_key.tStart = t  # local t and not account for scr refresh
                GOfromMRT_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(GOfromMRT_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'GOfromMRT_key.started')
                # update status
                GOfromMRT_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(GOfromMRT_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(GOfromMRT_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if GOfromMRT_key.status == STARTED and not waitOnFlip:
                theseKeys = GOfromMRT_key.getKeys(keyList=['5'], ignoreKeys=["escape"], waitRelease=False)
                _GOfromMRT_key_allKeys.extend(theseKeys)
                if len(_GOfromMRT_key_allKeys):
                    GOfromMRT_key.keys = _GOfromMRT_key_allKeys[-1].name  # just the last key pressed
                    GOfromMRT_key.rt = _GOfromMRT_key_allKeys[-1].rt
                    GOfromMRT_key.duration = _GOfromMRT_key_allKeys[-1].duration
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
                    currentRoutine=wait4scanner,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                wait4scanner.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wait4scanner.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wait4scanner" ---
        for thisComponent in wait4scanner.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wait4scanner
        wait4scanner.tStop = globalClock.getTime(format='float')
        wait4scanner.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wait4scanner.stopped', wait4scanner.tStop)
        # check responses
        if GOfromMRT_key.keys in ['', [], None]:  # No response was made
            GOfromMRT_key.keys = None
        wait4scanner_byp.addData('GOfromMRT_key.keys',GOfromMRT_key.keys)
        if GOfromMRT_key.keys != None:  # we had a response
            wait4scanner_byp.addData('GOfromMRT_key.rt', GOfromMRT_key.rt)
            wait4scanner_byp.addData('GOfromMRT_key.duration', GOfromMRT_key.duration)
        # the Routine "wait4scanner" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisWait4scanner_byp as finished
        if hasattr(thisWait4scanner_byp, 'status'):
            thisWait4scanner_byp.status = FINISHED
        # if awaiting a pause, pause now
        if wait4scanner_byp.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            wait4scanner_byp.status = STARTED
        thisExp.nextEntry()
        
    # completed 0.0 repeats of 'wait4scanner_byp'
    wait4scanner_byp.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    byp_start = data.TrialHandler2(
        name='byp_start',
        nReps=0.0, 
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
    # completed 0.0 repeats of 'byp_start'
    byp_start.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'stim/MR_AUT_items.csv', 
        selection='1:5'
    )
    , 
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
            components=[t_fix_mrk, t_fixation, t_item_mrk, AUTitem, idea_key, idea_mse],
        )
        t_item.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from fix_code
        rand_dur = np.random.uniform(3, 9)
        trials.addData('rand_dur', rand_dur)
        
        AUTitem.setText(MR_AUTitem)
        # create starting attributes for idea_key
        idea_key.keys = []
        idea_key.rt = []
        _idea_key_allKeys = []
        # setup some python lists for storing info about the idea_mse
        idea_mse.x = []
        idea_mse.y = []
        idea_mse.leftButton = []
        idea_mse.midButton = []
        idea_mse.rightButton = []
        idea_mse.time = []
        gotValidClick = False  # until a click is received
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
                if tThisFlipGlobal > t_fixation.tStartRefresh + rand_dur-frameTolerance:
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
                win.callOnFlip(t_item_mrk.setData, int(MR_AUTitem_mrk))
            
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
            
            # *AUTitem* updates
            
            # if AUTitem is starting this frame...
            if AUTitem.status == NOT_STARTED and t_fixation.status==FINISHED:
                # keep track of start time/frame for later
                AUTitem.frameNStart = frameN  # exact frame index
                AUTitem.tStart = t  # local t and not account for scr refresh
                AUTitem.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(AUTitem, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'AUTitem.started')
                # update status
                AUTitem.status = STARTED
                AUTitem.setAutoDraw(True)
            
            # if AUTitem is active this frame...
            if AUTitem.status == STARTED:
                # update params
                pass
            
            # if AUTitem is stopping this frame...
            if AUTitem.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > AUTitem.tStartRefresh + itemDur-frameTolerance:
                    # keep track of stop time/frame for later
                    AUTitem.tStop = t  # not accounting for scr refresh
                    AUTitem.tStopRefresh = tThisFlipGlobal  # on global time
                    AUTitem.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'AUTitem.stopped')
                    # update status
                    AUTitem.status = FINISHED
                    AUTitem.setAutoDraw(False)
            
            # *idea_key* updates
            waitOnFlip = False
            
            # if idea_key is starting this frame...
            if idea_key.status == NOT_STARTED and AUTitem.status==STARTED:
                # keep track of start time/frame for later
                idea_key.frameNStart = frameN  # exact frame index
                idea_key.tStart = t  # local t and not account for scr refresh
                idea_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(idea_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'idea_key.started')
                # update status
                idea_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(idea_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(idea_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if idea_key is stopping this frame...
            if idea_key.status == STARTED:
                if bool(AUTitem.status==FINISHED):
                    # keep track of stop time/frame for later
                    idea_key.tStop = t  # not accounting for scr refresh
                    idea_key.tStopRefresh = tThisFlipGlobal  # on global time
                    idea_key.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'idea_key.stopped')
                    # update status
                    idea_key.status = FINISHED
                    idea_key.status = FINISHED
            if idea_key.status == STARTED and not waitOnFlip:
                theseKeys = idea_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _idea_key_allKeys.extend(theseKeys)
                if len(_idea_key_allKeys):
                    idea_key.keys = _idea_key_allKeys[-1].name  # just the last key pressed
                    idea_key.rt = _idea_key_allKeys[-1].rt
                    idea_key.duration = _idea_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # *idea_mse* updates
            
            # if idea_mse is starting this frame...
            if idea_mse.status == NOT_STARTED and AUTitem.status==STARTED:
                # keep track of start time/frame for later
                idea_mse.frameNStart = frameN  # exact frame index
                idea_mse.tStart = t  # local t and not account for scr refresh
                idea_mse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(idea_mse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('idea_mse.started', t)
                # update status
                idea_mse.status = STARTED
                idea_mse.mouseClock.reset()
                prevButtonState = idea_mse.getPressed()  # if button is down already this ISN'T a new click
            
            # if idea_mse is stopping this frame...
            if idea_mse.status == STARTED:
                if bool(AUTitem.status==FINISHED):
                    # keep track of stop time/frame for later
                    idea_mse.tStop = t  # not accounting for scr refresh
                    idea_mse.tStopRefresh = tThisFlipGlobal  # on global time
                    idea_mse.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('idea_mse.stopped', t)
                    # update status
                    idea_mse.status = FINISHED
            if idea_mse.status == STARTED:  # only update if started and not finished!
                buttons = idea_mse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = idea_mse.getPos()
                        idea_mse.x.append(x)
                        idea_mse.y.append(y)
                        buttons = idea_mse.getPressed()
                        idea_mse.leftButton.append(buttons[0])
                        idea_mse.midButton.append(buttons[1])
                        idea_mse.rightButton.append(buttons[2])
                        idea_mse.time.append(idea_mse.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
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
        # check responses
        if idea_key.keys in ['', [], None]:  # No response was made
            idea_key.keys = None
        trials.addData('idea_key.keys',idea_key.keys)
        if idea_key.keys != None:  # we had a response
            trials.addData('idea_key.rt', idea_key.rt)
            trials.addData('idea_key.duration', idea_key.duration)
        # store data for trials (TrialHandler)
        trials.addData('idea_mse.x', idea_mse.x)
        trials.addData('idea_mse.y', idea_mse.y)
        trials.addData('idea_mse.leftButton', idea_mse.leftButton)
        trials.addData('idea_mse.midButton', idea_mse.midButton)
        trials.addData('idea_mse.rightButton', idea_mse.rightButton)
        trials.addData('idea_mse.time', idea_mse.time)
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
            ##  recording flag  ##
            is_recording = False
            recording_started = False
            
            ##  Counter for recording enumeration
            recording_count = 1
            
            ##  get start-time for 't-...'-routine
            #t_startTime = core.getTime()
            
            ##t_mrk: START - white '?' && ##t_mrk: 3min idea time START
            t_mrk.setData(49); core.wait(0.1); t_mrk.setData(0)
            
            ## Vor dem ersten Aufruf von getKeys() und getState() in "each frame"
            t_kb.clearEvents(eventType='keyboard')  # Lösche alle vorherigen Tasteneingaben
            
            audio_data = []
            stream = None
            
            ################################################################################
            
            """
            ##  flags for spacekey  ##
            #t_spacedown = False
            #t_spaceup = True
            
            is_recording = False
            
            ##  Counter for recording enumeration
            recording_count = 1
            
            ##  get start-time for 't-...'-routine
            t_startTime = core.getTime()
            
            #t_FZ.color = 'white'
            win.flip()
            
            ##t_mrk: START Nachdenkphase
            t_mrk.setData(50); core.wait(0.1); t_mrk.setData(0)
            
            
            ## Vor dem ersten Aufruf von getKeys() und getState() in "each frame"
            t_kb.clearEvents(eventType='keyboard')  # Lösche alle vorherigen Tasteneingaben  
            """
            
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
                # Initialize recording on first call
                if not recording_started:
                    recording_started = True
                    
                    # Display blue question mark
                    t_FZ.color = 'blue'
                    t_FZ.text = '?'
                    win.flip()
                    
                    # Start audio recording
                    audio_data = []
                    try:
                        stream = sd.InputStream(
                            samplerate=sample_rate,
                            channels=channels,
                            callback=lambda indata, frames, time, status: audio_data.append(indata.copy())
                        )
                        stream.start()
                        
                        # Send start marker
                        t_mrk.setData(48); core.wait(0.1); t_mrk.setData(0)
                        
                    except Exception as e:
                        print(f"Recording failed to start: {e}")
                        continueRoutine = False
                
                # Check for exit keys
                keys = t_kb.getKeys()
                if keys and any(k.name == 'escape' for k in keys):
                    safe_stop_stream(stream)
                    continueRoutine = False
                
                # Check for spacebar to stop
                if t_kb.getState('space'):
                    # Wait for release
                    while t_kb.getState('space'):
                        core.wait(0.01)
                    
                    # Stop and save
                    safe_stop_stream(stream)
                    
                    if audio_data:
                        #save_recording(audio_data, recording_count, wavDirName, sample_rate,
                        #             expInfo['UPN-ID'], RITitems[trials.thisN][itm])
                        save_recording(audio_data, recording_count, wavDirName, sample_rate)
                        #             expInfo['UPN-ID'], RITitems[trials.thisN][itm])
                    
                    # Send stop marker
                    t_mrk.setData(47); core.wait(0.1); t_mrk.setData(0)
                    
                    # Clear screen and finish
                    t_FZ.text = ''
                    win.flip()
                    continueRoutine = False
                
                core.wait(0.01)
                
                #################################################################################################################################################################
                ##    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###   ##
                #################################################################################################################################################################
                
                """
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
                    ###Helper function to save a recording###
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
                """
                
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
            # Run 'End Routine' code from t_code
            ##  show mouse  ##
            win.mouseVisible = True
            
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
        
        # --- Prepare to start Routine "AUT_rating" ---
        # create an object to store info about Routine AUT_rating
        AUT_rating = data.Routine(
            name='AUT_rating',
            components=[r_header_txt, VASrating, r_end_key, r1_mouse, l_label, r_label, r1_btn, rating_hint],
        )
        AUT_rating.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        VASrating.reset()
        # create starting attributes for r_end_key
        r_end_key.keys = []
        r_end_key.rt = []
        _r_end_key_allKeys = []
        # setup some python lists for storing info about the r1_mouse
        r1_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # store start times for AUT_rating
        AUT_rating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        AUT_rating.tStart = globalClock.getTime(format='float')
        AUT_rating.status = STARTED
        thisExp.addData('AUT_rating.started', AUT_rating.tStart)
        AUT_rating.maxDuration = None
        # keep track of which components have finished
        AUT_ratingComponents = AUT_rating.components
        for thisComponent in AUT_rating.components:
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
        
        # --- Run Routine "AUT_rating" ---
        AUT_rating.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *r_header_txt* updates
            
            # if r_header_txt is starting this frame...
            if r_header_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                r_header_txt.frameNStart = frameN  # exact frame index
                r_header_txt.tStart = t  # local t and not account for scr refresh
                r_header_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(r_header_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'r_header_txt.started')
                # update status
                r_header_txt.status = STARTED
                r_header_txt.setAutoDraw(True)
            
            # if r_header_txt is active this frame...
            if r_header_txt.status == STARTED:
                # update params
                pass
            
            # *VASrating* updates
            
            # if VASrating is starting this frame...
            if VASrating.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                VASrating.frameNStart = frameN  # exact frame index
                VASrating.tStart = t  # local t and not account for scr refresh
                VASrating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(VASrating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'VASrating.started')
                # update status
                VASrating.status = STARTED
                VASrating.setAutoDraw(True)
            
            # if VASrating is active this frame...
            if VASrating.status == STARTED:
                # update params
                pass
            
            # *r_end_key* updates
            waitOnFlip = False
            
            # if r_end_key is starting this frame...
            if r_end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                r_end_key.frameNStart = frameN  # exact frame index
                r_end_key.tStart = t  # local t and not account for scr refresh
                r_end_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(r_end_key, 'tStartRefresh')  # time at next scr refresh
                # update status
                r_end_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(r_end_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(r_end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if r_end_key.status == STARTED and not waitOnFlip:
                theseKeys = r_end_key.getKeys(keyList=['w','return'], ignoreKeys=["escape"], waitRelease=False)
                _r_end_key_allKeys.extend(theseKeys)
                if len(_r_end_key_allKeys):
                    r_end_key.keys = _r_end_key_allKeys[-1].name  # just the last key pressed
                    r_end_key.rt = _r_end_key_allKeys[-1].rt
                    r_end_key.duration = _r_end_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # *r1_mouse* updates
            
            # if r1_mouse is starting this frame...
            if r1_mouse.status == NOT_STARTED and VASrating.getMouseResponses() != None:
                # keep track of start time/frame for later
                r1_mouse.frameNStart = frameN  # exact frame index
                r1_mouse.tStart = t  # local t and not account for scr refresh
                r1_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(r1_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('r1_mouse.started', t)
                # update status
                r1_mouse.status = STARTED
                r1_mouse.mouseClock.reset()
                prevButtonState = r1_mouse.getPressed()  # if button is down already this ISN'T a new click
            if r1_mouse.status == STARTED:  # only update if started and not finished!
                buttons = r1_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(r1_btn, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(r1_mouse):
                                gotValidClick = True
                                r1_mouse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            r1_mouse.clicked_name.append(None)
                        if gotValidClick:  
                            continueRoutine = False  # end routine on response
            
            # *l_label* updates
            
            # if l_label is starting this frame...
            if l_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                l_label.frameNStart = frameN  # exact frame index
                l_label.tStart = t  # local t and not account for scr refresh
                l_label.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(l_label, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'l_label.started')
                # update status
                l_label.status = STARTED
                l_label.setAutoDraw(True)
            
            # if l_label is active this frame...
            if l_label.status == STARTED:
                # update params
                pass
            
            # *r_label* updates
            
            # if r_label is starting this frame...
            if r_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                r_label.frameNStart = frameN  # exact frame index
                r_label.tStart = t  # local t and not account for scr refresh
                r_label.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(r_label, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'r_label.started')
                # update status
                r_label.status = STARTED
                r_label.setAutoDraw(True)
            
            # if r_label is active this frame...
            if r_label.status == STARTED:
                # update params
                pass
            
            # *r1_btn* updates
            
            # if r1_btn is starting this frame...
            if r1_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                r1_btn.frameNStart = frameN  # exact frame index
                r1_btn.tStart = t  # local t and not account for scr refresh
                r1_btn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(r1_btn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'r1_btn.started')
                # update status
                r1_btn.status = STARTED
                r1_btn.setAutoDraw(True)
            
            # if r1_btn is active this frame...
            if r1_btn.status == STARTED:
                # update params
                pass
            
            # *rating_hint* updates
            
            # if rating_hint is starting this frame...
            if rating_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rating_hint.frameNStart = frameN  # exact frame index
                rating_hint.tStart = t  # local t and not account for scr refresh
                rating_hint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rating_hint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rating_hint.started')
                # update status
                rating_hint.status = STARTED
                rating_hint.setAutoDraw(True)
            
            # if rating_hint is active this frame...
            if rating_hint.status == STARTED:
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
                    currentRoutine=AUT_rating,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                AUT_rating.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in AUT_rating.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "AUT_rating" ---
        for thisComponent in AUT_rating.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for AUT_rating
        AUT_rating.tStop = globalClock.getTime(format='float')
        AUT_rating.tStopRefresh = tThisFlipGlobal
        thisExp.addData('AUT_rating.stopped', AUT_rating.tStop)
        # Run 'End Routine' code from AUTrating_code
        ##  marker: AUTrating STOP  ##
        t_mrk.setData(57); core.wait(0.1); t_mrk.setData(0)
        
        trials.addData('VASrating.response', VASrating.getRating())
        trials.addData('VASrating.rt', VASrating.getRT())
        # check responses
        if r_end_key.keys in ['', [], None]:  # No response was made
            r_end_key.keys = None
        trials.addData('r_end_key.keys',r_end_key.keys)
        if r_end_key.keys != None:  # we had a response
            trials.addData('r_end_key.rt', r_end_key.rt)
            trials.addData('r_end_key.duration', r_end_key.duration)
        # store data for trials (TrialHandler)
        x, y = r1_mouse.getPos()
        buttons = r1_mouse.getPressed()
        if sum(buttons):
            # check if the mouse was inside our 'clickable' objects
            gotValidClick = False
            clickableList = environmenttools.getFromNames(r1_btn, namespace=locals())
            for obj in clickableList:
                # is this object clicked on?
                if obj.contains(r1_mouse):
                    gotValidClick = True
                    r1_mouse.clicked_name.append(obj.name)
            if not gotValidClick:
                r1_mouse.clicked_name.append(None)
        trials.addData('r1_mouse.x', x)
        trials.addData('r1_mouse.y', y)
        trials.addData('r1_mouse.leftButton', buttons[0])
        trials.addData('r1_mouse.midButton', buttons[1])
        trials.addData('r1_mouse.rightButton', buttons[2])
        if len(r1_mouse.clicked_name):
            trials.addData('r1_mouse.clicked_name', r1_mouse.clicked_name[0])
        # the Routine "AUT_rating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "insight2" ---
        # create an object to store info about Routine insight2
        insight2 = data.Routine(
            name='insight2',
            components=[qu_header2_txt, moi, iihs_txt, iihs, insight_btn, insight_btn_txt, insight_mse],
        )
        insight2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        moi.reset()
        iihs.reset()
        # setup some python lists for storing info about the insight_mse
        insight_mse.x = []
        insight_mse.y = []
        insight_mse.leftButton = []
        insight_mse.midButton = []
        insight_mse.rightButton = []
        insight_mse.time = []
        insight_mse.clicked_name = []
        gotValidClick = False  # until a click is received
        # store start times for insight2
        insight2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        insight2.tStart = globalClock.getTime(format='float')
        insight2.status = STARTED
        thisExp.addData('insight2.started', insight2.tStart)
        insight2.maxDuration = None
        # keep track of which components have finished
        insight2Components = insight2.components
        for thisComponent in insight2.components:
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
        
        # --- Run Routine "insight2" ---
        insight2.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *qu_header2_txt* updates
            
            # if qu_header2_txt is starting this frame...
            if qu_header2_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                qu_header2_txt.frameNStart = frameN  # exact frame index
                qu_header2_txt.tStart = t  # local t and not account for scr refresh
                qu_header2_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(qu_header2_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'qu_header2_txt.started')
                # update status
                qu_header2_txt.status = STARTED
                qu_header2_txt.setAutoDraw(True)
            
            # if qu_header2_txt is active this frame...
            if qu_header2_txt.status == STARTED:
                # update params
                pass
            
            # *moi* updates
            
            # if moi is starting this frame...
            if moi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moi.frameNStart = frameN  # exact frame index
                moi.tStart = t  # local t and not account for scr refresh
                moi.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moi, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moi.started')
                # update status
                moi.status = STARTED
                moi.setAutoDraw(True)
            
            # if moi is active this frame...
            if moi.status == STARTED:
                # update params
                pass
            
            # *iihs_txt* updates
            
            # if iihs_txt is starting this frame...
            if iihs_txt.status == NOT_STARTED and moi.getRating() == 'Ja':
                # keep track of start time/frame for later
                iihs_txt.frameNStart = frameN  # exact frame index
                iihs_txt.tStart = t  # local t and not account for scr refresh
                iihs_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(iihs_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'iihs_txt.started')
                # update status
                iihs_txt.status = STARTED
                iihs_txt.setAutoDraw(True)
            
            # if iihs_txt is active this frame...
            if iihs_txt.status == STARTED:
                # update params
                pass
            
            # *iihs* updates
            
            # if iihs is starting this frame...
            if iihs.status == NOT_STARTED and moi.getRating() == 'Ja':
                # keep track of start time/frame for later
                iihs.frameNStart = frameN  # exact frame index
                iihs.tStart = t  # local t and not account for scr refresh
                iihs.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(iihs, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'iihs.started')
                # update status
                iihs.status = STARTED
                iihs.setAutoDraw(True)
            
            # if iihs is active this frame...
            if iihs.status == STARTED:
                # update params
                pass
            
            # *insight_btn* updates
            
            # if insight_btn is starting this frame...
            if insight_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                insight_btn.frameNStart = frameN  # exact frame index
                insight_btn.tStart = t  # local t and not account for scr refresh
                insight_btn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(insight_btn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'insight_btn.started')
                # update status
                insight_btn.status = STARTED
                insight_btn.setAutoDraw(True)
            
            # if insight_btn is active this frame...
            if insight_btn.status == STARTED:
                # update params
                pass
            
            # *insight_btn_txt* updates
            
            # if insight_btn_txt is starting this frame...
            if insight_btn_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                insight_btn_txt.frameNStart = frameN  # exact frame index
                insight_btn_txt.tStart = t  # local t and not account for scr refresh
                insight_btn_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(insight_btn_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'insight_btn_txt.started')
                # update status
                insight_btn_txt.status = STARTED
                insight_btn_txt.setAutoDraw(True)
            
            # if insight_btn_txt is active this frame...
            if insight_btn_txt.status == STARTED:
                # update params
                pass
            # *insight_mse* updates
            
            # if insight_mse is starting this frame...
            if insight_mse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                insight_mse.frameNStart = frameN  # exact frame index
                insight_mse.tStart = t  # local t and not account for scr refresh
                insight_mse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(insight_mse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('insight_mse.started', t)
                # update status
                insight_mse.status = STARTED
                insight_mse.mouseClock.reset()
                prevButtonState = insight_mse.getPressed()  # if button is down already this ISN'T a new click
            if insight_mse.status == STARTED:  # only update if started and not finished!
                buttons = insight_mse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(insight_btn, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(insight_mse):
                                gotValidClick = True
                                insight_mse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            insight_mse.clicked_name.append(None)
                        x, y = insight_mse.getPos()
                        insight_mse.x.append(x)
                        insight_mse.y.append(y)
                        buttons = insight_mse.getPressed()
                        insight_mse.leftButton.append(buttons[0])
                        insight_mse.midButton.append(buttons[1])
                        insight_mse.rightButton.append(buttons[2])
                        insight_mse.time.append(insight_mse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
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
                    currentRoutine=insight2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                insight2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in insight2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "insight2" ---
        for thisComponent in insight2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for insight2
        insight2.tStop = globalClock.getTime(format='float')
        insight2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('insight2.stopped', insight2.tStop)
        trials.addData('moi.response', moi.getRating())
        trials.addData('moi.rt', moi.getRT())
        trials.addData('iihs.response', iihs.getRating())
        trials.addData('iihs.rt', iihs.getRT())
        # store data for trials (TrialHandler)
        trials.addData('insight_mse.x', insight_mse.x)
        trials.addData('insight_mse.y', insight_mse.y)
        trials.addData('insight_mse.leftButton', insight_mse.leftButton)
        trials.addData('insight_mse.midButton', insight_mse.midButton)
        trials.addData('insight_mse.rightButton', insight_mse.rightButton)
        trials.addData('insight_mse.time', insight_mse.time)
        trials.addData('insight_mse.clicked_name', insight_mse.clicked_name)
        # the Routine "insight2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
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

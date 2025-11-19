#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on November 19, 2025, at 09:50
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
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
_winSize = [1920, 1200]
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
        originPath='C:\\Users\\weberbe\\Nextcloud\\__PsychoPy\\MR_AUT\\MR_AUT_lastrun.py',
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
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
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
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
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
    if deviceManager.getDevice('AUTidea_key') is None:
        # initialise AUTidea_key
        AUTidea_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='AUTidea_key',
        )
    if deviceManager.getDevice('likertRating_end_key') is None:
        # initialise likertRating_end_key
        likertRating_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='likertRating_end_key',
        )
    if deviceManager.getDevice('insight_end_key') is None:
        # initialise insight_end_key
        insight_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='insight_end_key',
        )
    if deviceManager.getDevice('insi_intensity_end_key') is None:
        # initialise insi_intensity_end_key
        insi_intensity_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='insi_intensity_end_key',
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
    if expInfo['session']:
        bids_handler = BIDSHandler(dataset='MR_AUT_BIDS',
         subject=expInfo['participant'], task=expInfo['expName'],
         session=expInfo['session'], data_type='func', acq='',
         runs=True)
    else:
        bids_handler = BIDSHandler(dataset='MR_AUT_BIDS',
         subject=expInfo['participant'], task=expInfo['expName'],
         data_type='func', acq='', runs=True)
    bids_handler.createDataset()
    bids_handler.addLicense('CC-BY-ND-4.0', force=True)
    bids_handler.addTaskCode(force=True)
    bids_handler.addEnvironment()
    
    # --- Initialize components for Routine "settings" ---
    # Run 'Begin Experiment' code from set_things
    ##  some start time info
    print(data.getDateStr(format="%H:%M:%S.%m"))   #local: set/mark start of current run
    
    wavDirName = '%s_%s_%s_%s_%s' % (os.path.join(_thisDir, 'data', ''), expInfo['participant'], expName, data.getDateStr(format="%Y-%m-%d_%H%M"), '_wav')
    if not os.path.isdir(wavDirName):
        os.makedirs(wavDirName)  # to hold .wav files
    #print(f"{wavDirName=}")
    
    ##  define some handy keys
    goRight_key = '7'
    goOn_key    = '8'
    goLeft_key  = '9' 
    
    ##  ### Options for DEBUG/RESARCH-mode ###  ##
    DEBUG = 1 #0=research-mode; 1=debug-mode 
    if DEBUG: 
        fixDur = 2
        itemDur = 5
        carryOver_lock_time = 0.5       #for 'carryOver_lock_time' [s] it's not possible to stop the answer
        MR_AUT_taskMaxTime = .5 * 60     #debugging: task lasts for 30s/1min
    else: 
        fixDur = 6
        itemDur = 15
        carryOver_lock_time = 1         #for 'carryOver_lock_time' [s] it's not possible to stop the answer
        MR_AUT_taskMaxTime = 20 * 60    #orig: task lasts for 20min
        #MR_AUT_taskMaxTime = 5 * 60    #kind of debug: task lasts for 5min
    
    
    # --- Initialize components for Routine "wait4scanner" ---
    GOfromMRT_txt = visual.TextStim(win=win, name='GOfromMRT_txt',
        text='...awaiting clearance from the MRI-Scanner...',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    GOfromMRT_key = keyboard.Keyboard(deviceName='GOfromMRT_key')
    
    # --- Initialize components for Routine "AUT_item" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    AUTitem_txt = visual.TextStim(win=win, name='AUTitem_txt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    AUTidea_key = keyboard.Keyboard(deviceName='AUTidea_key')
    
    # --- Initialize components for Routine "AUT_response" ---
    # Run 'Begin Experiment' code from AUTidea_code
    import sounddevice as sd
    import soundfile as sf
    import time
    
    ## set audio recording parameters
    sample_rate = 44100  # Sample rate (Hz)
    channels = 1         # Number of audio channels (2 for stereo)
    
    ## get/access the keyboard
    t_kb = keyboard.Keyboard()
    
    ### safe stop stream  ##
    def safe_stop_stream(stream):
        ##stop and close an input stream safely##
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        return None
    
    
    ###   save_recording   ###
    def save_recording(audio_data, recording_count, recordings_dir, sample_rate):
        ###Helper function to save a recording###
        if not audio_data:
            print("No audio data to save.")
            return False
        
        # Convert list of numpy arrays into one large numpy array
        recording = np.concatenate(audio_data, axis=0)
        
        ## prepare the recording
        audio_file = os.path.join(wavDirName, f"{expInfo['participant']}_{MR_AUTitem}.wav")    #name the recording with 'participant'/'sub.-ID'
        
        try:
            sf.write(audio_file, recording, sample_rate)            #save the recording to a file
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False
            
        return True
    
    AUTidea = visual.TextStim(win=win, name='AUTidea',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='yellow', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "AUT_likertRating" ---
    likertRating = visual.Slider(win=win, name='likertRating',
        startValue=3, size=(0.8, 0.025), pos=(0, 0), units=win.units,
        labels=None, ticks=[1,2,3,4,5], granularity=1,
        style=['radio'], styleTweaks=(), opacity=1,
        labelColor='pink', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.035,
        flip=False, ori=0, depth=0, readOnly=False)
    likertRating_header = visual.TextStim(win=win, name='likertRating_header',
        text='Kreativ',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    likertRating_l_label = visual.TextStim(win=win, name='likertRating_l_label',
        text='gar nicht',
        font='Arial',
        pos=(-.4, -.05), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    likertRating_r_label = visual.TextStim(win=win, name='likertRating_r_label',
        text='sehr',
        font='Arial',
        pos=(.4, -.05), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    likertRating_end_key = keyboard.Keyboard(deviceName='likertRating_end_key')
    
    # --- Initialize components for Routine "AUT_insight" ---
    insi_possible = visual.Slider(win=win, name='insi_possible',
        startValue=1, size=(0.5, 0.025), pos=(0, 0), units=win.units,
        labels=["Ja","Nein","Weiß\nnicht"],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='white', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.035,
        flip=False, ori=0.0, depth=0, readOnly=False)
    insi_possible_header = visual.TextStim(win=win, name='insi_possible_header',
        text='Insight',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    insight_end_key = keyboard.Keyboard(deviceName='insight_end_key')
    
    # --- Initialize components for Routine "insi_intensity" ---
    insi_intensity_likert = visual.Slider(win=win, name='insi_intensity_likert',
        startValue=3, size=(0.8, 0.025), pos=(0, 0), units=win.units,
        labels=["schwach", "stark"], ticks=[1,2,3,4,5], granularity=1,
        style=['radio'], styleTweaks=(), opacity=1,
        labelColor='white', markerColor='white', lineColor='white', colorSpace='rgb',
        font='Open Sans', labelHeight=0.035,
        flip=False, ori=0, depth=0, readOnly=False)
    insi_intensity_header = visual.TextStim(win=win, name='insi_intensity_header',
        text='Wie stark',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-3.0);
    insi_intensity_end_key = keyboard.Keyboard(deviceName='insi_intensity_end_key')
    
    # --- Initialize components for Routine "chk4_n2item" ---
    
    # --- Initialize components for Routine "chk4_n2block" ---
    
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
        nReps=1.0, 
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
        # Run 'End Routine' code from start_MR_AUT_code
        ##  store start time of MR_AUT task  ##
        globalClock.reset()
        MR_AUT_startTime = globalClock.getTime()
        
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
        
    # completed 1.0 repeats of 'wait4scanner_byp'
    wait4scanner_byp.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    MR_AUT_blocks = data.TrialHandler2(
        name='MR_AUT_blocks',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('stim/MR_AUT_blocks.csv'), 
        seed=None, 
    )
    thisExp.addLoop(MR_AUT_blocks)  # add the loop to the experiment
    thisMR_AUT_block = MR_AUT_blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMR_AUT_block.rgb)
    if thisMR_AUT_block != None:
        for paramName in thisMR_AUT_block:
            globals()[paramName] = thisMR_AUT_block[paramName]
    
    for thisMR_AUT_block in MR_AUT_blocks:
        MR_AUT_blocks.status = STARTED
        if hasattr(thisMR_AUT_block, 'status'):
            thisMR_AUT_block.status = STARTED
        currentLoop = MR_AUT_blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisMR_AUT_block.rgb)
        if thisMR_AUT_block != None:
            for paramName in thisMR_AUT_block:
                globals()[paramName] = thisMR_AUT_block[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            MR_AUT_condition_file, 
            selection='17:19'
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
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            trials.status = STARTED
            if hasattr(thisTrial, 'status'):
                thisTrial.status = STARTED
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "AUT_item" ---
            # create an object to store info about Routine AUT_item
            AUT_item = data.Routine(
                name='AUT_item',
                components=[fixation, AUTitem_txt, AUTidea_key],
            )
            AUT_item.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            AUTitem_txt.setText(MR_AUTitem)
            # create starting attributes for AUTidea_key
            AUTidea_key.keys = []
            AUTidea_key.rt = []
            _AUTidea_key_allKeys = []
            # store start times for AUT_item
            AUT_item.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            AUT_item.tStart = globalClock.getTime(format='float')
            AUT_item.status = STARTED
            thisExp.addData('AUT_item.started', AUT_item.tStart)
            AUT_item.maxDuration = None
            # keep track of which components have finished
            AUT_itemComponents = AUT_item.components
            for thisComponent in AUT_item.components:
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
            
            # --- Run Routine "AUT_item" ---
            AUT_item.forceEnded = routineForceEnded = not continueRoutine
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
                
                # *fixation* updates
                
                # if fixation is starting this frame...
                if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation.frameNStart = frameN  # exact frame index
                    fixation.tStart = t  # local t and not account for scr refresh
                    fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.started')
                    # update status
                    fixation.status = STARTED
                    fixation.setAutoDraw(True)
                
                # if fixation is active this frame...
                if fixation.status == STARTED:
                    # update params
                    pass
                
                # if fixation is stopping this frame...
                if fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation.tStartRefresh + fixDur-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation.tStop = t  # not accounting for scr refresh
                        fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation.stopped')
                        # update status
                        fixation.status = FINISHED
                        fixation.setAutoDraw(False)
                
                # *AUTitem_txt* updates
                
                # if AUTitem_txt is starting this frame...
                if AUTitem_txt.status == NOT_STARTED and fixation.status==FINISHED:
                    # keep track of start time/frame for later
                    AUTitem_txt.frameNStart = frameN  # exact frame index
                    AUTitem_txt.tStart = t  # local t and not account for scr refresh
                    AUTitem_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(AUTitem_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'AUTitem_txt.started')
                    # update status
                    AUTitem_txt.status = STARTED
                    AUTitem_txt.setAutoDraw(True)
                
                # if AUTitem_txt is active this frame...
                if AUTitem_txt.status == STARTED:
                    # update params
                    pass
                
                # if AUTitem_txt is stopping this frame...
                if AUTitem_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > AUTitem_txt.tStartRefresh + itemDur-frameTolerance:
                        # keep track of stop time/frame for later
                        AUTitem_txt.tStop = t  # not accounting for scr refresh
                        AUTitem_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        AUTitem_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'AUTitem_txt.stopped')
                        # update status
                        AUTitem_txt.status = FINISHED
                        AUTitem_txt.setAutoDraw(False)
                
                # *AUTidea_key* updates
                waitOnFlip = False
                
                # if AUTidea_key is starting this frame...
                if AUTidea_key.status == NOT_STARTED and AUTitem_txt.status==STARTED:
                    # keep track of start time/frame for later
                    AUTidea_key.frameNStart = frameN  # exact frame index
                    AUTidea_key.tStart = t  # local t and not account for scr refresh
                    AUTidea_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(AUTidea_key, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'AUTidea_key.started')
                    # update status
                    AUTidea_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(AUTidea_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(AUTidea_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if AUTidea_key is stopping this frame...
                if AUTidea_key.status == STARTED:
                    if bool(AUTitem_txt.status==FINISHED):
                        # keep track of stop time/frame for later
                        AUTidea_key.tStop = t  # not accounting for scr refresh
                        AUTidea_key.tStopRefresh = tThisFlipGlobal  # on global time
                        AUTidea_key.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'AUTidea_key.stopped')
                        # update status
                        AUTidea_key.status = FINISHED
                        AUTidea_key.status = FINISHED
                if AUTidea_key.status == STARTED and not waitOnFlip:
                    theseKeys = AUTidea_key.getKeys(keyList=['8','3'], ignoreKeys=["escape"], waitRelease=False)
                    _AUTidea_key_allKeys.extend(theseKeys)
                    if len(_AUTidea_key_allKeys):
                        AUTidea_key.keys = _AUTidea_key_allKeys[-1].name  # just the last key pressed
                        AUTidea_key.rt = _AUTidea_key_allKeys[-1].rt
                        AUTidea_key.duration = _AUTidea_key_allKeys[-1].duration
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
                        currentRoutine=AUT_item,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    AUT_item.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in AUT_item.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "AUT_item" ---
            for thisComponent in AUT_item.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for AUT_item
            AUT_item.tStop = globalClock.getTime(format='float')
            AUT_item.tStopRefresh = tThisFlipGlobal
            thisExp.addData('AUT_item.stopped', AUT_item.tStop)
            # Run 'End Routine' code from fix_code
            AUTidea_key.clearEvents(eventType='keyboard')  # Lösche alle vorherigen Tasteneingaben
            
            try:
                if fixation.tStopRefresh is not None:
                    duration_val = fixation.tStopRefresh - fixation.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_item.stopped'] - fixation.tStartRefresh
                bids_event = BIDSTaskEvent(
                    onset=fixation.tStartRefresh,
                    duration=duration_val,
                    event_type='cross',
                    trial_type='fixation',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_fix.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            try:
                if AUTitem_txt.tStopRefresh is not None:
                    duration_val = AUTitem_txt.tStopRefresh - AUTitem_txt.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_item.stopped'] - AUTitem_txt.tStartRefresh
                bids_event = BIDSTaskEvent(
                    onset=AUTitem_txt.tStartRefresh,
                    duration=duration_val,
                    event_type=f'{MR_AUTitem}',
                    trial_type='AUTitem',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_AUTitem.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            # check responses
            if AUTidea_key.keys in ['', [], None]:  # No response was made
                AUTidea_key.keys = None
            trials.addData('AUTidea_key.keys',AUTidea_key.keys)
            if AUTidea_key.keys != None:  # we had a response
                trials.addData('AUTidea_key.rt', AUTidea_key.rt)
                trials.addData('AUTidea_key.duration', AUTidea_key.duration)
            try:
                if AUTidea_key.tStopRefresh is not None:
                    duration_val = AUTidea_key.tStopRefresh - AUTidea_key.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_item.stopped'] - AUTidea_key.tStartRefresh
                if hasattr(AUTidea_key, 'rt'):
                    rt_val = AUTidea_key.rt
                else:
                    rt_val = None
                    logging.warning('The linked component "AUTidea_key" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                bids_event = BIDSTaskEvent(
                    onset=AUTidea_key.tStartRefresh,
                    duration=duration_val,
                    response_time=rt_val,
                    event_type='button_press',
                    trial_type='AUTidea_key',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_AUTideakey.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            # the Routine "AUT_item" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "AUT_response" ---
            # create an object to store info about Routine AUT_response
            AUT_response = data.Routine(
                name='AUT_response',
                components=[AUTidea],
            )
            AUT_response.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from AUTidea_code
            event.clearEvents('keyboard')
            
            ##  recording flag  ##
            recording_started = False
            
            ##  Counter for recording enumeration
            recording_count = 1
            
            ##  get start-time for 't-...'-routine
            t_startTime = core.getTime()
            
            ## Vor dem ersten Aufruf von getKeys() und getState() in "each frame"
            t_kb.clearEvents(eventType='keyboard')  # Lösche alle vorherigen Tasteneingaben
            
            audio_data = []
            stream = None
            
            max_answer_time = 4
            
            AUTidea.setText('yummuy')
            # store start times for AUT_response
            AUT_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            AUT_response.tStart = globalClock.getTime(format='float')
            AUT_response.status = STARTED
            thisExp.addData('AUT_response.started', AUT_response.tStart)
            AUT_response.maxDuration = None
            # keep track of which components have finished
            AUT_responseComponents = AUT_response.components
            for thisComponent in AUT_response.components:
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
            
            # --- Run Routine "AUT_response" ---
            AUT_response.forceEnded = routineForceEnded = not continueRoutine
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
                # Run 'Each Frame' code from AUTidea_code
                ##  check 'max_answer_time'  ##
                if core.getTime() > t_startTime + max_answer_time:
                    #print("'max_answer_time' over...")
                    
                    if recording_started:
                        stream = safe_stop_stream(stream)
                        if audio_data:
                            save_recording(audio_data, recording_count, wavDirName, sample_rate)
                    
                    is_recording = False
                    continueRoutine = False # finish trial
                
                
                ##  Initialize recording on first call  ##
                if not recording_started:
                    recording_started = True
                    
                    # Start audio recording
                    audio_data = []
                    try:
                        stream = sd.InputStream(
                            samplerate=sample_rate,
                            channels=channels,
                            callback=lambda indata, frames, time, status: audio_data.append(indata.copy())
                        )
                        stream.start()
                
                    except Exception as e:
                        print(f"Recording failed to start: {e}")
                        continueRoutine = False
                        
                    # Display AUTitem, but different color
                    AUTidea.text = MR_AUTitem
                    #win.flip()
                
                
                ##  Check for exit keys
                keys = t_kb.getKeys()
                if keys and any(k.name == 'escape' for k in keys):
                    safe_stop_stream(stream)
                    continueRoutine = False
                
                ##  Check for spacebar/'goOn-Key' to stop  ##
                #if t_kb.getState('space'):
                if t_kb.getState(goOn_key):
                    # Wait for release
                    #while t_kb.getState('space'):
                    while t_kb.getState(goOn_key):
                        core.wait(0.01)
                    
                    if core.getTime() > t_startTime + carryOver_lock_time:
                        # Stop and save
                        safe_stop_stream(stream)
                    
                        if audio_data:
                            save_recording(audio_data, recording_count, wavDirName, sample_rate)
                        
                        # Clear screen and finish
                        AUTidea.text = ''
                        win.flip()
                        continueRoutine = False
                
                core.wait(0.01)
                
                
                # *AUTidea* updates
                
                # if AUTidea is starting this frame...
                if AUTidea.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    AUTidea.frameNStart = frameN  # exact frame index
                    AUTidea.tStart = t  # local t and not account for scr refresh
                    AUTidea.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(AUTidea, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'AUTidea.started')
                    # update status
                    AUTidea.status = STARTED
                    AUTidea.setAutoDraw(True)
                
                # if AUTidea is active this frame...
                if AUTidea.status == STARTED:
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
                        currentRoutine=AUT_response,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    AUT_response.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in AUT_response.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "AUT_response" ---
            for thisComponent in AUT_response.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for AUT_response
            AUT_response.tStop = globalClock.getTime(format='float')
            AUT_response.tStopRefresh = tThisFlipGlobal
            thisExp.addData('AUT_response.stopped', AUT_response.tStop)
            try:
                if AUTidea.tStopRefresh is not None:
                    duration_val = AUTidea.tStopRefresh - AUTidea.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_response.stopped'] - AUTidea.tStartRefresh
                bids_event = BIDSTaskEvent(
                    onset=AUTidea.tStartRefresh,
                    duration=duration_val,
                    event_type=f'idea_{MR_AUTitem}',
                    trial_type='AUTaudioResponse',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_AUTresponse.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            # the Routine "AUT_response" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "AUT_likertRating" ---
            # create an object to store info about Routine AUT_likertRating
            AUT_likertRating = data.Routine(
                name='AUT_likertRating',
                components=[likertRating, likertRating_header, likertRating_l_label, likertRating_r_label, likertRating_end_key],
            )
            AUT_likertRating.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            likertRating.reset()
            # Run 'Begin Routine' code from likertRating_code
            ##  set likert scale starting point  ##
            event.clearEvents('keyboard')
            likertRating.markerPos = 3
            
            # create starting attributes for likertRating_end_key
            likertRating_end_key.keys = []
            likertRating_end_key.rt = []
            _likertRating_end_key_allKeys = []
            # store start times for AUT_likertRating
            AUT_likertRating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            AUT_likertRating.tStart = globalClock.getTime(format='float')
            AUT_likertRating.status = STARTED
            thisExp.addData('AUT_likertRating.started', AUT_likertRating.tStart)
            AUT_likertRating.maxDuration = None
            # keep track of which components have finished
            AUT_likertRatingComponents = AUT_likertRating.components
            for thisComponent in AUT_likertRating.components:
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
            
            # --- Run Routine "AUT_likertRating" ---
            AUT_likertRating.forceEnded = routineForceEnded = not continueRoutine
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
                
                # *likertRating* updates
                
                # if likertRating is starting this frame...
                if likertRating.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    likertRating.frameNStart = frameN  # exact frame index
                    likertRating.tStart = t  # local t and not account for scr refresh
                    likertRating.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(likertRating, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'likertRating.started')
                    # update status
                    likertRating.status = STARTED
                    likertRating.setAutoDraw(True)
                
                # if likertRating is active this frame...
                if likertRating.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from likertRating_code
                keys = event.getKeys()
                
                if len(keys):
                    if goLeft_key in keys:
                        likertRating.markerPos = likertRating.markerPos - 1
                    elif goRight_key in keys:
                        likertRating.markerPos = likertRating.markerPos  + 1
                
                
                # *likertRating_header* updates
                
                # if likertRating_header is starting this frame...
                if likertRating_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    likertRating_header.frameNStart = frameN  # exact frame index
                    likertRating_header.tStart = t  # local t and not account for scr refresh
                    likertRating_header.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(likertRating_header, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'likertRating_header.started')
                    # update status
                    likertRating_header.status = STARTED
                    likertRating_header.setAutoDraw(True)
                
                # if likertRating_header is active this frame...
                if likertRating_header.status == STARTED:
                    # update params
                    pass
                
                # *likertRating_l_label* updates
                
                # if likertRating_l_label is starting this frame...
                if likertRating_l_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    likertRating_l_label.frameNStart = frameN  # exact frame index
                    likertRating_l_label.tStart = t  # local t and not account for scr refresh
                    likertRating_l_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(likertRating_l_label, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    likertRating_l_label.status = STARTED
                    likertRating_l_label.setAutoDraw(True)
                
                # if likertRating_l_label is active this frame...
                if likertRating_l_label.status == STARTED:
                    # update params
                    pass
                
                # *likertRating_r_label* updates
                
                # if likertRating_r_label is starting this frame...
                if likertRating_r_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    likertRating_r_label.frameNStart = frameN  # exact frame index
                    likertRating_r_label.tStart = t  # local t and not account for scr refresh
                    likertRating_r_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(likertRating_r_label, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    likertRating_r_label.status = STARTED
                    likertRating_r_label.setAutoDraw(True)
                
                # if likertRating_r_label is active this frame...
                if likertRating_r_label.status == STARTED:
                    # update params
                    pass
                
                # *likertRating_end_key* updates
                waitOnFlip = False
                
                # if likertRating_end_key is starting this frame...
                if likertRating_end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    likertRating_end_key.frameNStart = frameN  # exact frame index
                    likertRating_end_key.tStart = t  # local t and not account for scr refresh
                    likertRating_end_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(likertRating_end_key, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    likertRating_end_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(likertRating_end_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(likertRating_end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if likertRating_end_key.status == STARTED and not waitOnFlip:
                    theseKeys = likertRating_end_key.getKeys(keyList=[goOn_key,'return'], ignoreKeys=["escape"], waitRelease=False)
                    _likertRating_end_key_allKeys.extend(theseKeys)
                    if len(_likertRating_end_key_allKeys):
                        likertRating_end_key.keys = _likertRating_end_key_allKeys[-1].name  # just the last key pressed
                        likertRating_end_key.rt = _likertRating_end_key_allKeys[-1].rt
                        likertRating_end_key.duration = _likertRating_end_key_allKeys[-1].duration
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
                        currentRoutine=AUT_likertRating,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    AUT_likertRating.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in AUT_likertRating.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "AUT_likertRating" ---
            for thisComponent in AUT_likertRating.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for AUT_likertRating
            AUT_likertRating.tStop = globalClock.getTime(format='float')
            AUT_likertRating.tStopRefresh = tThisFlipGlobal
            thisExp.addData('AUT_likertRating.stopped', AUT_likertRating.tStop)
            trials.addData('likertRating.response', likertRating.getRating())
            trials.addData('likertRating.rt', likertRating.getRT())
            # Run 'End Routine' code from likertRating_code
            thisExp.addData("likertRating", likertRating.markerPos)
            
            try:
                if likertRating.tStopRefresh is not None:
                    duration_val = likertRating.tStopRefresh - likertRating.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_likertRating.stopped'] - likertRating.tStartRefresh
                if hasattr(likertRating, 'rt'):
                    rt_val = likertRating.rt
                else:
                    rt_val = None
                    logging.warning('The linked component "likertRating" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                bids_event = BIDSTaskEvent(
                    onset=likertRating.tStartRefresh,
                    duration=duration_val,
                    response_time=rt_val,
                    event_type=f'likertRating_{likertRating.markerPos}',
                    trial_type='AUTselfrating',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_AUTlikert.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            # check responses
            if likertRating_end_key.keys in ['', [], None]:  # No response was made
                likertRating_end_key.keys = None
            trials.addData('likertRating_end_key.keys',likertRating_end_key.keys)
            if likertRating_end_key.keys != None:  # we had a response
                trials.addData('likertRating_end_key.rt', likertRating_end_key.rt)
                trials.addData('likertRating_end_key.duration', likertRating_end_key.duration)
            try:
                if likertRating_end_key.tStopRefresh is not None:
                    duration_val = likertRating_end_key.tStopRefresh - likertRating_end_key.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_likertRating.stopped'] - likertRating_end_key.tStartRefresh
                if hasattr(likertRating_end_key, 'rt'):
                    rt_val = likertRating_end_key.rt
                else:
                    rt_val = None
                    logging.warning('The linked component "likertRating_end_key" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                bids_event = BIDSTaskEvent(
                    onset=likertRating_end_key.tStartRefresh,
                    duration=duration_val,
                    response_time=rt_val,
                    event_type='certifyLikertRating',
                    trial_type='eof_likertRating',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_likertRating_key.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            # the Routine "AUT_likertRating" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "AUT_insight" ---
            # create an object to store info about Routine AUT_insight
            AUT_insight = data.Routine(
                name='AUT_insight',
                components=[insi_possible, insi_possible_header, insight_end_key],
            )
            AUT_insight.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            insi_possible.reset()
            # Run 'Begin Routine' code from insi_code
            show_insi_intensity = 0
            
            event.clearEvents('keyboard')
            insi_possible.markerPos = 1
            
            pos_insi_rating = 99
            
            # create starting attributes for insight_end_key
            insight_end_key.keys = []
            insight_end_key.rt = []
            _insight_end_key_allKeys = []
            # store start times for AUT_insight
            AUT_insight.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            AUT_insight.tStart = globalClock.getTime(format='float')
            AUT_insight.status = STARTED
            thisExp.addData('AUT_insight.started', AUT_insight.tStart)
            AUT_insight.maxDuration = None
            # keep track of which components have finished
            AUT_insightComponents = AUT_insight.components
            for thisComponent in AUT_insight.components:
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
            
            # --- Run Routine "AUT_insight" ---
            AUT_insight.forceEnded = routineForceEnded = not continueRoutine
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
                
                # *insi_possible* updates
                
                # if insi_possible is starting this frame...
                if insi_possible.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    insi_possible.frameNStart = frameN  # exact frame index
                    insi_possible.tStart = t  # local t and not account for scr refresh
                    insi_possible.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(insi_possible, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'insi_possible.started')
                    # update status
                    insi_possible.status = STARTED
                    insi_possible.setAutoDraw(True)
                
                # if insi_possible is active this frame...
                if insi_possible.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from insi_code
                keys = event.getKeys()
                
                if len(keys):
                    if goLeft_key in keys:
                        insi_possible.markerPos = insi_possible.markerPos - 1
                    elif goRight_key in keys:
                        insi_possible.markerPos = insi_possible.markerPos  + 1 
                    pos_insi_rating = insi_possible.markerPos
                
                if insi_possible.markerPos == 0:    #'Ja'
                    show_insi_intensity = 1
                elif insi_possible.markerPos == 1 or insi_possible.markerPos == 2:  #'Nein'/'Weiß nicht'
                    show_insi_intensity = 0
                
                
                # *insi_possible_header* updates
                
                # if insi_possible_header is starting this frame...
                if insi_possible_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    insi_possible_header.frameNStart = frameN  # exact frame index
                    insi_possible_header.tStart = t  # local t and not account for scr refresh
                    insi_possible_header.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(insi_possible_header, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'insi_possible_header.started')
                    # update status
                    insi_possible_header.status = STARTED
                    insi_possible_header.setAutoDraw(True)
                
                # if insi_possible_header is active this frame...
                if insi_possible_header.status == STARTED:
                    # update params
                    pass
                
                # *insight_end_key* updates
                waitOnFlip = False
                
                # if insight_end_key is starting this frame...
                if insight_end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    insight_end_key.frameNStart = frameN  # exact frame index
                    insight_end_key.tStart = t  # local t and not account for scr refresh
                    insight_end_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(insight_end_key, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    insight_end_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(insight_end_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(insight_end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if insight_end_key.status == STARTED and not waitOnFlip:
                    theseKeys = insight_end_key.getKeys(keyList=[goOn_key,'return'], ignoreKeys=["escape"], waitRelease=False)
                    _insight_end_key_allKeys.extend(theseKeys)
                    if len(_insight_end_key_allKeys):
                        insight_end_key.keys = _insight_end_key_allKeys[-1].name  # just the last key pressed
                        insight_end_key.rt = _insight_end_key_allKeys[-1].rt
                        insight_end_key.duration = _insight_end_key_allKeys[-1].duration
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
                        currentRoutine=AUT_insight,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    AUT_insight.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in AUT_insight.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "AUT_insight" ---
            for thisComponent in AUT_insight.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for AUT_insight
            AUT_insight.tStop = globalClock.getTime(format='float')
            AUT_insight.tStopRefresh = tThisFlipGlobal
            thisExp.addData('AUT_insight.stopped', AUT_insight.tStop)
            trials.addData('insi_possible.response', insi_possible.getRating())
            trials.addData('insi_possible.rt', insi_possible.getRT())
            # Run 'End Routine' code from insi_code
            thisExp.addData("Insight", insi_possible.markerPos)
            
            try:
                if insi_possible.tStopRefresh is not None:
                    duration_val = insi_possible.tStopRefresh - insi_possible.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_insight.stopped'] - insi_possible.tStartRefresh
                if hasattr(insi_possible, 'rt'):
                    rt_val = insi_possible.rt
                else:
                    rt_val = None
                    logging.warning('The linked component "insi_possible" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                bids_event = BIDSTaskEvent(
                    onset=insi_possible.tStartRefresh,
                    duration=duration_val,
                    response_time=rt_val,
                    event_type=f'INSI_possible_{insi_possible.markerPos}',
                    trial_type='INSI_possible',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_possible_insi.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            # check responses
            if insight_end_key.keys in ['', [], None]:  # No response was made
                insight_end_key.keys = None
            trials.addData('insight_end_key.keys',insight_end_key.keys)
            if insight_end_key.keys != None:  # we had a response
                trials.addData('insight_end_key.rt', insight_end_key.rt)
                trials.addData('insight_end_key.duration', insight_end_key.duration)
            try:
                if insight_end_key.tStopRefresh is not None:
                    duration_val = insight_end_key.tStopRefresh - insight_end_key.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['AUT_insight.stopped'] - insight_end_key.tStartRefresh
                if hasattr(insight_end_key, 'rt'):
                    rt_val = insight_end_key.rt
                else:
                    rt_val = None
                    logging.warning('The linked component "insight_end_key" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                bids_event = BIDSTaskEvent(
                    onset=insight_end_key.tStartRefresh,
                    duration=duration_val,
                    response_time=rt_val,
                    event_type='certify_insight_possible',
                    trial_type='eof_INSI_possible',
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    trials.addData('bidsE_insi_key.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
            # the Routine "AUT_insight" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            insi_strength_on = data.TrialHandler2(
                name='insi_strength_on',
                nReps=show_insi_intensity, 
                method='fullRandom', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(insi_strength_on)  # add the loop to the experiment
            thisInsi_strength_on = insi_strength_on.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisInsi_strength_on.rgb)
            if thisInsi_strength_on != None:
                for paramName in thisInsi_strength_on:
                    globals()[paramName] = thisInsi_strength_on[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisInsi_strength_on in insi_strength_on:
                insi_strength_on.status = STARTED
                if hasattr(thisInsi_strength_on, 'status'):
                    thisInsi_strength_on.status = STARTED
                currentLoop = insi_strength_on
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisInsi_strength_on.rgb)
                if thisInsi_strength_on != None:
                    for paramName in thisInsi_strength_on:
                        globals()[paramName] = thisInsi_strength_on[paramName]
                
                # --- Prepare to start Routine "insi_intensity" ---
                # create an object to store info about Routine insi_intensity
                insi_intensity = data.Routine(
                    name='insi_intensity',
                    components=[insi_intensity_likert, insi_intensity_header, insi_intensity_end_key],
                )
                insi_intensity.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                insi_intensity_likert.reset()
                # Run 'Begin Routine' code from insi_intensity_code
                event.clearEvents('keyboard')
                insi_intensity_likert.markerPos = 3
                
                # create starting attributes for insi_intensity_end_key
                insi_intensity_end_key.keys = []
                insi_intensity_end_key.rt = []
                _insi_intensity_end_key_allKeys = []
                # store start times for insi_intensity
                insi_intensity.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                insi_intensity.tStart = globalClock.getTime(format='float')
                insi_intensity.status = STARTED
                thisExp.addData('insi_intensity.started', insi_intensity.tStart)
                insi_intensity.maxDuration = None
                # keep track of which components have finished
                insi_intensityComponents = insi_intensity.components
                for thisComponent in insi_intensity.components:
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
                
                # --- Run Routine "insi_intensity" ---
                insi_intensity.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisInsi_strength_on, 'status') and thisInsi_strength_on.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *insi_intensity_likert* updates
                    
                    # if insi_intensity_likert is starting this frame...
                    if insi_intensity_likert.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        insi_intensity_likert.frameNStart = frameN  # exact frame index
                        insi_intensity_likert.tStart = t  # local t and not account for scr refresh
                        insi_intensity_likert.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(insi_intensity_likert, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'insi_intensity_likert.started')
                        # update status
                        insi_intensity_likert.status = STARTED
                        insi_intensity_likert.setAutoDraw(True)
                    
                    # if insi_intensity_likert is active this frame...
                    if insi_intensity_likert.status == STARTED:
                        # update params
                        insi_intensity_likert.setColor('white', colorSpace='rgb', log=False)
                        insi_intensity_likert.setFillColor('red', log=False)
                        insi_intensity_likert.setBorderColor('white', log=False)
                    # Run 'Each Frame' code from insi_intensity_code
                    keys = event.getKeys()
                    
                    if len(keys):
                        if goLeft_key in keys:
                            insi_intensity_likert.markerPos = insi_intensity_likert.markerPos - 1
                        elif goRight_key in keys:
                            insi_intensity_likert.markerPos = insi_intensity_likert.markerPos  + 1 
                    
                    
                    # *insi_intensity_header* updates
                    
                    # if insi_intensity_header is starting this frame...
                    if insi_intensity_header.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        insi_intensity_header.frameNStart = frameN  # exact frame index
                        insi_intensity_header.tStart = t  # local t and not account for scr refresh
                        insi_intensity_header.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(insi_intensity_header, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'insi_intensity_header.started')
                        # update status
                        insi_intensity_header.status = STARTED
                        insi_intensity_header.setAutoDraw(True)
                    
                    # if insi_intensity_header is active this frame...
                    if insi_intensity_header.status == STARTED:
                        # update params
                        pass
                    
                    # *insi_intensity_end_key* updates
                    waitOnFlip = False
                    
                    # if insi_intensity_end_key is starting this frame...
                    if insi_intensity_end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        insi_intensity_end_key.frameNStart = frameN  # exact frame index
                        insi_intensity_end_key.tStart = t  # local t and not account for scr refresh
                        insi_intensity_end_key.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(insi_intensity_end_key, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        insi_intensity_end_key.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(insi_intensity_end_key.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(insi_intensity_end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if insi_intensity_end_key.status == STARTED and not waitOnFlip:
                        theseKeys = insi_intensity_end_key.getKeys(keyList=[goOn_key,'return'], ignoreKeys=["escape"], waitRelease=False)
                        _insi_intensity_end_key_allKeys.extend(theseKeys)
                        if len(_insi_intensity_end_key_allKeys):
                            insi_intensity_end_key.keys = _insi_intensity_end_key_allKeys[-1].name  # just the last key pressed
                            insi_intensity_end_key.rt = _insi_intensity_end_key_allKeys[-1].rt
                            insi_intensity_end_key.duration = _insi_intensity_end_key_allKeys[-1].duration
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
                            currentRoutine=insi_intensity,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        insi_intensity.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in insi_intensity.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "insi_intensity" ---
                for thisComponent in insi_intensity.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for insi_intensity
                insi_intensity.tStop = globalClock.getTime(format='float')
                insi_intensity.tStopRefresh = tThisFlipGlobal
                thisExp.addData('insi_intensity.stopped', insi_intensity.tStop)
                insi_strength_on.addData('insi_intensity_likert.response', insi_intensity_likert.getRating())
                insi_strength_on.addData('insi_intensity_likert.rt', insi_intensity_likert.getRT())
                # Run 'End Routine' code from insi_intensity_code
                thisExp.addData("insight_intensity", insi_intensity_likert.markerPos)
                
                try:
                    if insi_intensity_likert.tStopRefresh is not None:
                        duration_val = insi_intensity_likert.tStopRefresh - insi_intensity_likert.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['insi_intensity.stopped'] - insi_intensity_likert.tStartRefresh
                    if hasattr(insi_intensity_likert, 'rt'):
                        rt_val = insi_intensity_likert.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "insi_intensity_likert" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=insi_intensity_likert.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type=f'INSI_intensity_{insi_intensity_likert.markerPos}',
                        trial_type='INSI_intensity',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        insi_strength_on.addData('bidsE_insi_intensity.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
                # check responses
                if insi_intensity_end_key.keys in ['', [], None]:  # No response was made
                    insi_intensity_end_key.keys = None
                insi_strength_on.addData('insi_intensity_end_key.keys',insi_intensity_end_key.keys)
                if insi_intensity_end_key.keys != None:  # we had a response
                    insi_strength_on.addData('insi_intensity_end_key.rt', insi_intensity_end_key.rt)
                    insi_strength_on.addData('insi_intensity_end_key.duration', insi_intensity_end_key.duration)
                try:
                    if insi_intensity_end_key.tStopRefresh is not None:
                        duration_val = insi_intensity_end_key.tStopRefresh - insi_intensity_end_key.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['insi_intensity.stopped'] - insi_intensity_end_key.tStartRefresh
                    if hasattr(insi_intensity_end_key, 'rt'):
                        rt_val = insi_intensity_end_key.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "insi_intensity_end_key" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=insi_intensity_end_key.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type='certify_INSI_intensity',
                        trial_type='eof_INSI_intensity',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        insi_strength_on.addData('bidsE_insi_intensity_key.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                logging.log(level=24, msg={k: v for k, v in bids_event.items() if v is not None})
                # the Routine "insi_intensity" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisInsi_strength_on as finished
                if hasattr(thisInsi_strength_on, 'status'):
                    thisInsi_strength_on.status = FINISHED
                # if awaiting a pause, pause now
                if insi_strength_on.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    insi_strength_on.status = STARTED
                thisExp.nextEntry()
                
            # completed show_insi_intensity repeats of 'insi_strength_on'
            insi_strength_on.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "chk4_n2item" ---
            # create an object to store info about Routine chk4_n2item
            chk4_n2item = data.Routine(
                name='chk4_n2item',
                components=[],
            )
            chk4_n2item.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from n2_item_check
            ##  check (remaining) time for items of 'n2' MR_AUT-Item-pool  ##
            if MR_AUT_condition_file == 'stim/MR_AUT_items_n2.csv':
                if globalClock.getTime() > (MR_AUT_startTime + MR_AUT_taskMaxTime):
                    trials.finished = True
                    continueRoutine = False
            
            # store start times for chk4_n2item
            chk4_n2item.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            chk4_n2item.tStart = globalClock.getTime(format='float')
            chk4_n2item.status = STARTED
            thisExp.addData('chk4_n2item.started', chk4_n2item.tStart)
            chk4_n2item.maxDuration = None
            # keep track of which components have finished
            chk4_n2itemComponents = chk4_n2item.components
            for thisComponent in chk4_n2item.components:
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
            
            # --- Run Routine "chk4_n2item" ---
            chk4_n2item.forceEnded = routineForceEnded = not continueRoutine
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
                        currentRoutine=chk4_n2item,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    chk4_n2item.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in chk4_n2item.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "chk4_n2item" ---
            for thisComponent in chk4_n2item.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for chk4_n2item
            chk4_n2item.tStop = globalClock.getTime(format='float')
            chk4_n2item.tStopRefresh = tThisFlipGlobal
            thisExp.addData('chk4_n2item.stopped', chk4_n2item.tStop)
            # the Routine "chk4_n2item" was not non-slip safe, so reset the non-slip timer
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
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "chk4_n2block" ---
        # create an object to store info about Routine chk4_n2block
        chk4_n2block = data.Routine(
            name='chk4_n2block',
            components=[],
        )
        chk4_n2block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from n2_block_check
        ##  check (remaining) time for items of 'n2' MR_AUT-Item-pool  ##
        if globalClock.getTime() > (MR_AUT_startTime + MR_AUT_taskMaxTime):
            trials.finished = True
            MR_AUT_blocks.finished = True
            continueRoutine = False
        
        # store start times for chk4_n2block
        chk4_n2block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        chk4_n2block.tStart = globalClock.getTime(format='float')
        chk4_n2block.status = STARTED
        thisExp.addData('chk4_n2block.started', chk4_n2block.tStart)
        chk4_n2block.maxDuration = None
        # keep track of which components have finished
        chk4_n2blockComponents = chk4_n2block.components
        for thisComponent in chk4_n2block.components:
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
        
        # --- Run Routine "chk4_n2block" ---
        chk4_n2block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisMR_AUT_block, 'status') and thisMR_AUT_block.status == STOPPING:
                continueRoutine = False
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
                    currentRoutine=chk4_n2block,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                chk4_n2block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in chk4_n2block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "chk4_n2block" ---
        for thisComponent in chk4_n2block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for chk4_n2block
        chk4_n2block.tStop = globalClock.getTime(format='float')
        chk4_n2block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('chk4_n2block.stopped', chk4_n2block.tStop)
        # the Routine "chk4_n2block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisMR_AUT_block as finished
        if hasattr(thisMR_AUT_block, 'status'):
            thisMR_AUT_block.status = FINISHED
        # if awaiting a pause, pause now
        if MR_AUT_blocks.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            MR_AUT_blocks.status = STARTED
    # completed 1.0 repeats of 'MR_AUT_blocks'
    MR_AUT_blocks.status = FINISHED
    
    
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
    try:
        if thx_key.tStopRefresh is not None:
            duration_val = thx_key.tStopRefresh - thx_key.tStartRefresh
        else:
            duration_val = thisExp.thisEntry['thx.stopped'] - thx_key.tStartRefresh
        if hasattr(thx_key, 'rt'):
            rt_val = thx_key.rt
        else:
            rt_val = None
            logging.warning('The linked component "thx_key" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
        bids_event = BIDSTaskEvent(
            onset=thx_key.tStartRefresh,
            duration=duration_val,
            response_time=rt_val,
            event_type='finalTaskKey',
            trial_type='endOfTask',
        )
        if bids_handler:
            bids_handler.addEvent(bids_event)
        else:
            thisExp.addData('bidsE_thx_key.event', bids_event)
    except BIDSError as e:
        print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thx.maxDurationReached:
        routineTimer.addTime(-thx.maxDuration)
    elif thx.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    thisExp.nextEntry()
    # the Routine "MR_AUT_BIDS" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    ignore_list = [
        'participant',
        'session',
        'date',
        'expName',
        'psychopyVersion',
        'OS',
        'frameRate'
    ]
    participant_info = {
        key: thisExp.extraInfo[key]
        for key in thisExp.extraInfo
        if key not in ignore_list
    }
    # write tsv file and update
    try:
        if bids_handler.events:
            bids_handler.writeEvents(participant_info, add_stimuli=True, execute_sidecar=True, generate_hed_metadata=True)
    except Exception as e:
        print(f"[psychopy-bids(settings)] An error occurred when writing BIDS events: {e}")
    
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

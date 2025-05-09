from psychopy import visual, core, monitors, event
import numpy as np
# Definition screen
mon = monitors.Monitor('myMonitor')
resolution = [1024 ,768]
mon.setDistance(90) #cm
mon.setWidth(33)#cm
mon.setSizePix(resolution)
mon.saveMon()
# Get t h e s c r e e n r e s o l u t i o n
screen_width, screen_height=resolution
# C a l c u l a t e t h e c e n t e r o f t h ec e n t e r x = s c r e e n w i d t h / 2
center_x = screen_width/2
center_y = screen_height/2
position=[center_x, center_y]
# C r e a t e a s m a l l e r window t o d i s p l a y t h e
winsize = [150,100] # A d j u s t t h e s i z e
win = visual.Window(size = winsize,
        monitor =mon ,
        color = 'gray',
        units = 'deg')


def create_and_save_gabor_patch(contrast_value, filename, position):
    # Create the f i x a t i o n cross
    fixation = visual.GratingStim(win=win, mask= 'cross' ,
        size = 0.4 , pos = [ 0 , 0 ] ,sf =0)
        
    gabor_left = visual.GratingStim(win, tex ='sin' , mask= 'gauss' ,
            texRes = 256 , pos = [-0.6,0],
            size = 2.5 , sf = [ 1.2 , 0 ] , ori =0, name='gabor_left')
    gabor_left.contrast = contrast_value -0.025
    gabor_left.draw()

    gabor_right = visual.GratingStim(win, tex ='sin' , mask= 'gauss' ,
            texRes = 256 , pos = [0.6,0],
            size = 2.5 , sf = [ 1.2 , 0 ] , ori =0, name='gabor_left')
    gabor_right.contrast = contrast_value +0.025
    gabor_right.draw()

    fixation.draw()
    win.flip()
    win.getMovieFrame()
    win.saveMovieFrames(filename)
    
    
ranges = np.array([3.5, 93])/100
for idx, contrast_value in enumerate(ranges):
    filename = f"gabor_contrast_{contrast_value*100:.1f}.png"
    create_and_save_gabor_patch(contrast_value, filename, position=position)
'''  
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import numpy as np
import cv2
import copy
import  time
def main():
    cap = cv2.VideoCapture(0)
    # pip install opencv-contrib-python
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)


    # number of frames is a variable for development purposes, you can change the for loop to a while(cap.isOpened()) instead to go through the whole video
   # num_frames = 350
    
    last_recorded_time = time.time()
    last_recorded_time_1 = time.time()
    
    first_iteration_indicator = 1
    #for i in range(0, num_frames):
    while(cap.isOpened()):
        '''
        There are some important reasons this if statement exists:
            -in the first run there is no previous frame, so this accounts for that
            -the first frame is saved to be used for the overlay after the accumulation has occurred
            -the height and width of the video are used to create an empty image for accumulation (accum_image)
        '''
        curr_time = time.time()
        ret, frame = cap.read()
        if first_iteration_indicator == 1:

            
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            print(height,width)
            h=400
            w=600
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0

        
        
        #if (first_iteration_indicator == 1):
        if curr_time - last_recorded_time >=1: 
            first_frame = copy.deepcopy(frame)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #first_iteration_indicator = 0
            #accum_image = np.zeros((height, width), np.uint8)
            
            last_recorded_time = curr_time
        if curr_time - last_recorded_time_1 >=7.0: 
            #first_frame = copy.deepcopy(frame)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #first_iteration_indicator = 0
            accum_image = np.zeros((height, width), np.uint8)
            
            last_recorded_time_1 = curr_time
        
        
            
            
        else:
            #ret, frame = cap.read()  # read a frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            fgmask = fgbg.apply(gray)  # remove the background

            # for testing purposes, show the result of the background subtraction
            #cv2.imshow('diff-bkgnd-frame', fgmask)

            # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
            # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
            thresh = 12
            maxValue = 40
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_TRUNC)
            # for testing purposes, show the threshold image
            #cv2.imshow('op',th1)
            # cv2.imwrite('diff-th1.jpg', th1)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
            # for testing purposes, show the accumulated image
            cv2.imshow('diff-accum.jpg', accum_image)

            # for testing purposes, control frame by frame
            # raw_input("press any key to continue")

        # for testing purposes, show the current frame
        # cv2.imshow('frame', gray)
        #cv2.imshow('frame', accum_image)
        color_image = im_color = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
        #cv2.imshow('heat', color_image)
        result_overlay = cv2.addWeighted(first_frame, 1, color_image, 0.7, 0)
        cv2.imshow('result over', result_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # apply a color map
    # COLORMAP_PINK also works well, COLORMAP_BONE is acceptable if the background is dark
    color_image = im_color = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    # for testing purposes, show the colorMap image
    cv2.imshow('heat', color_image)

    # overlay the color mapped image to the first frame
    result_overlay = cv2.addWeighted(accum_image, 0.7, color_image, 0.7, 0)

    # save the final overlay image
    cv2.imwrite('diff-overlay.jpg', result_overlay)

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
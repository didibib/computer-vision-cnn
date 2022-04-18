import tv_hi_data as thd
import cv2 as cv
import numpy as np

# https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
def get_optical_flow(set):
    for name in set:
        video = cv.VideoCapture('tv-hi_data/videos/' + name + '.avi')
    
        _, first_frame = video.read()
        prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(first_frame)
        
        # Sets image saturation to maximum
        mask[..., 1] = 255
        
        stack_size = 16
        frames_frac = video.get(cv.CAP_PROP_FRAME_COUNT) / (stack_size + 2)
        curr_frame = 0
        i = 1

        while(video.isOpened()):     
            # ret = a boolean return value from getting
            # the frame, frame = the current frame being
            # projected in the video
            #video.set(cv.CAP_PROP_POS_FRAMES, int(i * frames_frac) - 1)
            success, frame = video.read()
            if frame is None:
                break
            # Opens a new window and displays the input
            # frame
            #cv.imshow("input", frame)
            
            # Converts each frame to grayscale - we previously 
            # only converted the first frame to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Calculates dense optical flow by Farneback method
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                            None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Sets image hue according to the optical flow 
            # direction
            mask[..., 0] = angle * 180 / np.pi / 2
            
            # Sets image value according to the optical flow
            # magnitude (normalized)
            mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
            
            # Converts HSV to RGB (BGR) color representation
            rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
            
            # Opens a new window and displays the output frame
            cv.imshow("dense optical flow", rgb)
            
            # Updates previous frame
            prev_gray = gray
            
            curr_frame += 1
            if(curr_frame == int(i * frames_frac)):
                i += 1
                # save the image
                cv.imwrite('tv-hi_data/optical_flow/' + name + '_' + str(i-1) + '.jpg', mask)

            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
        # The following frees up resources and
        # closes all windows
        video.release()
        cv.destroyAllWindows()

get_optical_flow(thd.set_1)
get_optical_flow(thd.set_2)

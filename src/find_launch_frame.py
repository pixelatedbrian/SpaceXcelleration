import cv2
import numpy as np
import sys

import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

from multiprocessing import Pool
import glob

from PIL import Image

import time


def find_time_loc(img, debug=False):
    '''
    This function takes an image and a mask and tries to locate where the "T" for
    time in a SpaceX launch video is.  (This is what transitions from T-, before launch,
    to T+ after launch.)
    
    img:  numpy array representing an image from a SpaceX launch, should have 3 color channels
            
    debug:  Show the results of the alg to ensure that it's working as expected.
    
    Returns:  Tuple of the row, column which represents the upper left corner of the matching "T"
              the upper left has 1 pixel of padding in either direction for the top-leftmost
              pixel of the T
    
    '''

    # t_mask is the filter that will be convolved around the masked image looking for a
    # match.  It is simply the letter 'T' in the font used by spacex when the image is
    # sized to 640x360.
    
    # honestly just easier to hard code the array instead of storing it as a file or
    # getting passed in as a parameter.
    t_mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int)
    
    
    # working image size is 640x360 so check that the image is of this size
    # otherwise rescale the image so that it is the correct size
    if img.shape[0] != 360 or img.shape[1] != 640:
        temp_img = np.array(Image.fromarray(img).resize((640, 360), Image.BICUBIC))
    
    # put img in range of 0.0 - 1.0
    img = img / 255.0

    # convert to sort of grey scale
    mask = img[..., 0] + img[..., 1] + img[..., 2]
    mask = mask / 3.0

    # use pixel values to create a mask of the bright pixels, of which text is bright
    # if a pixel is 75% bright or higher it's true, otherwise false
    mask = mask > 0.75
    
    # convert mask to int's so it's more discrete
    mask = np.array(mask, dtype=int)

    if debug is True:
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.imshow(mask)

    # create ranges to convolve on
    row_range = mask.shape[0] - t_mask.shape[0] # 360 - 13
    col_range = mask.shape[1] - t_mask.shape[1] # 640 - 10

    if debug is True:
        print("row range: {:} col range: {:}".format(row_range, col_range))
    
        # for debugging only
        result_mask = np.zeros(shape=(row_range, col_range), dtype=int)

    # as we convolve on the big mask, we will keep track of what the best agreement so far is
    # and if we get a new maximum then row_max, col_max will be set to that location
    current_max_agreement = 0
    row_max = -1
    col_max = -1

    # walk through rows
    for row in range(row_range):

        # and columns
        for col in range(col_range):
            
            # extract pixels from the mask that are the same size/orientation as the verification mask
            sub_mask = mask[row:row + 13, col:col + 10]

            # check for agreement between the masks, in other words that a pixel that is True
            # on the big image mask is also true on the t_mask.
            agreement = sub_mask == t_mask

            # now quantify how much "agreement" there is between the masks
            # in python False is implicitly 0 and True is implicitly 1 so:
            agreement = np.sum(agreement)

            if debug is True:
                result_mask[row, col] = agreement

            if agreement > current_max_agreement:
                current_max_agreement = agreement

                row_max = row
                col_max = col


    if debug is True:
        show_mask = mask[row_max:row_max + 13, col_max:col_max + 10]
        fig, ax = plt.subplots()

        ax.imshow(show_mask)
        ax.set_title("mask from find_time_loc")

        print("best match: row: {:}  col: {:}".format(row, col))
        
        fig, ax = plt.subplots(figsize=(19.2, 10.8))

        ax.set_title("Map of Mask Activations from Convolutions", size=16)
        ax.imshow(result_mask)
        ax.scatter([col_max], [row_max], s=100, lw=2, edgecolor="red", alpha=1.0, color="black", label="Best Filter Match")

        ax.legend()

        fig, ax = plt.subplots(figsize=(19.2, 10.8))

        ax.imshow(mask)

        # Create a Rectangle patch
        rect = patches.Rectangle((col_max, row_max),10,13,linewidth=1,edgecolor='r',facecolor="r", alpha=0.5)

        # Add the patch to the Axes
        ax.add_patch(rect)

        ax.set_title("Overlay of the Best Match", size=16)
        
    return row_max, col_max



def crop_time_key(img, mode="img"):
    '''
    inputs:
    
    img: numpy array that we're working on
    mode: return an image or a mask: 'img', 'mask'
    
    returns:
    image (numpy array) or mask if mode == 'mask'
    '''
    
    # find the location of "T" in the image
    _row, _col = find_time_loc(img)
    
    # the existing masks have padding of ~4px so adjust find_time_loc from padding of 1 to padding of 4
    _row -= 2
    _col -= 3
    
    # the overall size of the image mask that we want to focus on for this task:    
    ROW_DIM = 17
    COL_DIM = 97
    
    # dims check out now 4-10-2020
#     print("before: 326:343, 270:367")
#     print("after: {:}:{:}, {:}:{:}".format(_row, _row + ROW_DIM, _col, _col + COL_DIM))
    
    test = np.array(img)
#     test = np.array(test[326:343, 270:367, :], dtype=np.float32)
    test = np.array(test[_row:_row + ROW_DIM, _col:_col + COL_DIM, :], dtype=np.float32)
    test = test / 255.0

    # ensure that mode is one of the correct options
    assert mode == "img" or mode == "mask", "Mode provided '{:}' does not match options: 'img', 'mask'".format(mode)
    
    if mode == "img":
        return test
    
    else: # mode == "mask"
    
        mask = test[..., 0] + test[..., 1] + test[..., 2]
        mask = mask / 3.0
                
        # arabsat fix, because the glare of the launch was so high the background blended with
        # the digits and the mask threshold failed, which broke the ability of the model to 
        # find the correct transition point
        if np.min(mask) > 0.20:
            new_mask = mask > 0.85
        else:
            new_mask = mask > 0.75

#         new_mask = mask > 0.75
                
        del test, mask
        
        return new_mask

def extract_tstate(mask):
    '''
    Inputs:
    mask: mask of larger timestamp, ex: T- 00:00:02
    
    RETURNS:
    cropped mask (numpy array) of "T" section
    from above example "T-" as a mask
    '''
    
    return mask[:, :23]
    

def get_launch_state(img, debug=False):
    '''
    bundle together the image processing steps so that a t-state is returned
    
    Input:
    img: numpy array of input image
    debug: print out the resulting mask if True
    
    returns:
    mask with t-state
    '''
    crop = crop_time_key(img, mode="mask")
    
    mask = extract_tstate(crop)
    
    if debug is True:
        fig, ax = plt.subplots()
        ax.imshow(mask)
        
    return mask 

def process_frame(frame, debug=False):
    '''
    Input:
    frame: a frame of the video from the opencv video reader
    
    returns:
    {"frame_num": index, "mask": mask} : a tuple that has the index of the image in the video, and the resulting time mode mask
    '''
    
    

    # have to resize frame to 640x360 which is what the first video was sized at
    
    # convert to an Image object so we have access to Image.resize method
    temp_img = Image.fromarray(np.array(frame))
    
    # resize to the standard size for video processing, in this library:
    temp_img = temp_img.resize((640, 360))
    
    # convert back to a numpy array so that it can be worked on
    temp_img = np.array(temp_img)
    
    crop = get_launch_state(temp_img, debug=debug)
    
    time_mode = predict_time_mode(crop)
    
    if debug is True:
        print(time_mode)
    
    return time_mode
    


def predict_time_mode(cropped_mask):
    # T- example:
    t_min = np.load("../models/t_minus_golden_mask.npy")
#     print("t_min.shape", t_min.shape)

    # T+ example:
    t_max = np.load("../models/t_plus_golden_mask.npy")
#     print("t_max.shape", t_max.shape)
    
    # the exemplar masks have been flattened into 1D arrays:
    cropped_mask = cropped_mask.flatten()
    
    # figure out euclidean distance comparing the cropped_mask with the two states
    # that we're interested in.  If the cropped_mask matches a state, for example
    # "t-minus" then the distance would be relatively small and therefore a match
    t_min_dist = np.sum((cropped_mask - t_min)**2)**0.5
    
    t_max_dist = np.sum((cropped_mask - t_max)**2)**0.5
    
    if t_min_dist < 2.0:
        return "prelaunch"
    
    elif t_max_dist < 2.0:
        return "launched"
    
    else:
        return "no_time_state"
    
    
    
    
    
def find_starting_frame(filepath, debug=False):
    '''
    Find which frame the launch starts on.  This will let us later focus on the frames after
    launch for detecting velocity, etc.
    
    The way that this function works is by breaking the video into chunks via time/frames.
    Then inspecting each chunk to see if the video has transitioned to launch
    If so then take the 'zone' (of step_size length), make step_size smaller, and check the new (smaller) zones
    Keep doing this until the step size is 1 and we should be finding the exact frame of transition
    
    Input: 
    filepath:  path to the video to be analyzed for velocity
    debug:     False, if it is True then some debug print statements and plots will be shown
    
    Returns:
    frame number when the launch time transitions from T- (before launch) to T+ (launching/launched)
    '''
    start_time = time.time()

    cap = cv2.VideoCapture(filepath)

    fps = cap.get(cv2.CAP_PROP_FPS)

    print("FPS detected in video: {:}".format(fps))

    counter = 0
    img_counter = 0
    
    # TODO: start this at 1000
    zone_start = 1000 # start at this frame for parity with old version
    step_size = int(fps) * 60 * 6 # it's one minute of frames at 30 fps * 6 for 6 minutes of frames per step
    zone_end = zone_start + step_size

#     previous_prediction = "prelaunch"

    _go = True

    while cap.isOpened() and _go is True:
        # zone_start and zone_end are updated at the end of each loop
        
        if counter == 0:  # first check
            cap.set(1, zone_start)
            
            ret, frame = cap.read()
            
            if ret != False:
                frame = np.array(frame)

                frame = frame[..., ::-1] # reverse color channel to fix openCV to RGB

                if debug is True:
                    print("\n\nchecking frame: {:}".format(zone_end))

                mode_prediction = process_frame(frame, debug=False)
                
                if mode_prediction == "launched": # the first frame is already launched so we should return that
                    
                    print("\n\nThe first checked frame is already launched, returning this frame and quitting early")
                    _go = False
                    
                    return zone_start
            else:
                print("\n\nERROR: Seemed to have moved past the last frame of the video, or otherwise gotten an empty frame from OpenCV.VideoCapture object.")

        counter +=1

        if debug is True:
            # keep track which frame is being worked on
            sys.stdout.write("\rWorking on Frame: {:}".format(zone_end))
            sys.stdout.flush()
            
        if counter > 100:
            # seems like this is just repeating and not actually finding the transition frame
            # this would happen if the video didn't have the time stamp (ie wrong kind of video)
            print("\n\n\nERROR: Could not find transition frame in 100 attempts.  Is this the correct kind of video?")
            
            # stop the loop
            _go = False

        img_counter += 1

        # get the frame from the video, convert to a numpy array, and rearrange color channels because of OpenCV
        cap.set(1, zone_end);

        ret, frame = cap.read()
        frame = np.array(frame)

        frame = frame[..., ::-1] # reverse color channel to fix openCV to RGB

        # get the actual prediction
        mode_prediction = process_frame(frame, debug=False)

        # if this is True then the transition happened within this zone
        if mode_prediction == "launched":

            if debug is True:
                print("\n\nseems like there was a transition between: {:} and {:}, new step size: {:}".format(zone_start, zone_end, step_size))

                
            # keep original working_frame for correct debug annotation on images:
            launched_frame = zone_end
            
            # zone_start is already good, but we need to update zone_end with a new step_size
              
            if step_size != 1:
                step_size = int(step_size / 4) 
                
                # if step size goes below 1.0, like 0.7, it will get rounded to zero by python int
                # but we need a step size at least 1
                if step_size == 0:
                    step_size = 1
                
            else:
                _go = False
                
                if debug is True:
                    print("Final Transition detected between frames at frame: {:} and {:}".format(launched_frame - 1, launched_frame))

            # update the new zone_end
            zone_end = zone_start + step_size
                    


            # we already know the current working_frame is "prelaunch" so step forward one new smaller step:
    #         working_frame += step_size

            if debug is True:
                fig, ax = plt.subplots(figsize=(12, 8))

                ax.imshow(frame)
                ax.annotate("frame:{:}".format(launched_frame), xy=(100, 100), size=30, color="#00FF00")
                ax.annotate("mode prediction: {:}".format(mode_prediction), xy=(100, 200), size=30, color="#00FF00")

        else: # mode_prediction is not "launched"
            
            # move to a new zone
            zone_start = zone_end
            zone_end += step_size
            
            
    #     img = Image.fromarray(frame)
    #     img.save("../data/mission_comparision_samples/starlink_6_{:06d}.png".format(counter))

        # somehow if we move past the end of the video then we need to stop and print an error
        if ret is False:
            _go = False
            
            print("\n\nERROR: Seemed to have moved past the last frame of the video, or otherwise gotten an empty frame from OpenCV.VideoCapture object.")

    print("\n\nTotal checks made to find launch frame:", counter)
            
    if debug is True:
        
        # show the frame before the transition frame so we can visually check that the image goes
        # from T- -> T+
        cap.set(1, launched_frame - 1);

        ret, frame = cap.read()

        fig, ax = plt.subplots(figsize=(12, 8))

        # fix color channels
        frame = np.array(frame)
        frame = frame[..., ::-1] # reverse color channel to fix openCV to RGB

        # get the actual prediction
        mode_prediction = process_frame(frame, debug=False)

        ax.imshow(frame)
        ax.annotate("frame:{:}".format(launched_frame  - 1), xy=(100, 100), size=30, color="#00FF00")
        ax.annotate("mode prediction: {:}".format(mode_prediction), xy=(100, 200), size=30, color="#00FF00")
            
    # free the video capture in memory now that we're done with it, for now
    cap.release()
    cv2.destroyAllWindows()

    print("\n\n\nFound Launch frame at {:}, completed search in {:4d}s".format(launched_frame -1, int(time.time() - start_time)))
    
    # the working frame should be the one where time has transitioned to + so use the prior
    # frame in the return
    return launched_frame - 1


# visual verified good transition frame: 71759
# starting_frame = find_starting_frame('../data/downloads/Arabsat-6A Mission (Hosted Webcast)-TXMGu2d8c8g.webm', debug=True)

# visual verified good transition frame: 26809
# starting_frame = find_starting_frame('../data/downloads/Starlink Mission-I4sMhHbHYXM.mkv', debug=True)

# visual verified good transition frame: 32344
# starting_frame = find_starting_frame('../data/downloads/Crew Dragon Launch Escape Demonstration-mhrkdHshb3E.webm', debug=True)

# visual verified good transition frame: 27006
# starting_frame = find_starting_frame('../data/downloads/CRS_18_Mission_YouTube.mp4', debug=True)

# visual verified good transition frame: 27010
# starting_frame = find_starting_frame('../data/downloads/CRS-19 Mission--aoAGdYXp_4.mkv', debug=True)

# visual verified good transition frame: 26809
# starting_frame = find_starting_frame('../data/downloads/Starlink Mission-I4sMhHbHYXM.mkv', debug=False)
# print(starting_frame)
    
    
    
    
    
# test code is working with a simple integration test
if __name__ == "__main__":
    
    # visual verified good transition frame: 26809
    starting_frame = find_starting_frame('../data/downloads/Starlink Mission-I4sMhHbHYXM.mkv', debug=False)
    print("test run on starlink debug file.")
    print("test run detected starting_frame: {:} vs known actual starting_frame: {:}".format(starting_frame, 26809))
    
    if starting_frame == 26809:
        print("TEST: PASSED")
        
    else:
        print("TEST: FAILED")
    
    
    
    
# ######################
# ### TEST CODE HERE ###
# ######################
# tminus_img = Image.open("../data/downloads/vid_imgs/sx_img_21606.png")

# # debug arabsat time transition failure
# tminus_img = Image.open("../data/mission_comparision_samples/arabsat_debug_01.png")

# temp = np.array(tminus_img)

# print(temp.shape)

# if temp.shape[0] != 360:
#     temp = Image.fromarray(temp)
#     temp = temp.resize((640, 360))
#     temp = np.array(temp)
    
#     print(temp.shape)

# fig, ax = plt.subplots(figsize=(19.2, 10.8))

# ax.set_title("T-minus Image")
# ax.imshow(tminus_img)

# result = process_frame(tminus_img, debug=True)

# print(result)

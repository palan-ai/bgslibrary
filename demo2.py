##############################################
# Demo file
# python demo2.py --- will use video
# python demo2.py image --- will use images
#############################################

import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pybgs as bgs

print("OpenCV Version: {}".format(cv2.__version__))


def is_cv2():
    return check_opencv_version("2.")


def is_cv3():
    return check_opencv_version("3.")


def is_lower_or_equals_cv347():
    [major, minor, revision] = str(cv2.__version__).split('.')
    return int(major) == 3 and int(minor) <= 4 and int(revision) <= 7


def is_cv4():
    return check_opencv_version("4.")


def check_opencv_version(major):
    return cv2.__version__.startswith(major)


image = False
if (len(sys.argv) == 2):
    if (sys.argv[1] == "image"):
        image = True
        img_folder = "dataset/frames"
        img_array = sorted(glob.iglob(img_folder + '/*.png'))


data_dir = "/home/vajra/Ongoing/flickering/"
evaluation_dataset_path = os.path.join(data_dir, "evaluation_data.csv")
eval_data = pd.read_csv(evaluation_dataset_path, index_col="video-name")
# eval_data = eval_data.sample(5)  # @TODO remove
master_result = []
for video_name, row in eval_data.iterrows():

    result = {"fg": [], "video_name": "", "fg_per_frame": 0}
    fg_count = []
    fg_per_frame = 0
    video_tag = video_name[video_name.find(
        "2023"):video_name.find("2023") + 17]
    video_names = glob.glob(data_dir + '*/*/*.mkv')
    video_file = ""
    for file in video_names:
        if "mkv" in file and video_tag in file:
            video_file = file
            break
    if os.path.exists(video_file):
        print(f"Processing {video_file}")
        pass
    else:
        print(f"Can not find {video_file}")
        continue

    # bgslibrary algorithms
    algorithms = []
    # algorithms.append(bgs.WeightedMovingMean())
    # algorithms.append(bgs.WeightedMovingVariance())
    # algorithms.append(bgs.AdaptiveBackgroundLearning())
    algorithms.append(bgs.AdaptiveSelectiveBackgroundLearning())
    # algorithms.append(bgs.MixtureOfGaussianV2())
    # algorithms.append(bgs.PixelBasedAdaptiveSegmenter())
    # algorithms.append(bgs.SigmaDelta())
    # algorithms.append(bgs.SuBSENSE())
    # algorithms.append(bgs.LOBSTER())
    # algorithms.append(bgs.PAWCS())
    # algorithms.append(bgs.TwoPoints())
    # algorithms.append(bgs.ViBe())
    # algorithms.append(bgs.CodeBook())

    if is_cv2():
        algorithms.append(bgs.MixtureOfGaussianV1())  # if opencv 2.x
        algorithms.append(bgs.GMG())  # if opencv 2.x

    if is_cv2() or is_cv3():
        algorithms.append(bgs.DPAdaptiveMedian())
        algorithms.append(bgs.DPGrimsonGMM())
        algorithms.append(bgs.DPZivkovicAGMM())
        algorithms.append(bgs.DPMean())
        algorithms.append(bgs.DPWrenGA())
        algorithms.append(bgs.DPPratiMediod())
        algorithms.append(bgs.DPEigenbackground())
        algorithms.append(bgs.DPTexture())
        algorithms.append(bgs.T2FGMM_UM())
        algorithms.append(bgs.T2FGMM_UV())
        algorithms.append(bgs.T2FMRF_UM())
        algorithms.append(bgs.T2FMRF_UV())
        algorithms.append(bgs.MultiCue())

    if is_cv2() or is_lower_or_equals_cv347():
        algorithms.append(bgs.LBP_MRF())
        algorithms.append(bgs.MultiLayer())

    print("Number of available algorithms: ", len(algorithms))
    # check if we want to use the images

    for algorithm in algorithms:
        print("Running ", algorithm.__class__)

        if (image):
            # loop x times as files in our folder
            for x in range(0, len(img_array)):

                # we can loop now through our array of images
                img_path = img_array[x]

                # read file into open cv and apply to algorithm to generate background model
                img = cv2.imread(img_path)
                img_output = algorithm.apply(img)
                img_bgmodel = algorithm.getBackgroundModel()

                # show images in python imshow window
                cv2.imshow('image', img)
                cv2.imshow('img_output', img_output)
                # cv2.imshow('img_bgmodel', img_bgmodel)

                # we need waitKey otherwise it wont display the image
                if 0xFF & cv2.waitKey(10) == 27:
                    break

                # Comment out to save images to bg and fg folder
                # img_bg = img_path.replace(img_folder, "output/bg")
                # img_fg = img_path.replace(img_folder, "output/fg")
                # cv2.imwrite(img_bg, img_bgmodel)
                # cv2.imwrite(img_fg, img_output)

                print("Frames left: " + str(len(img_array)-x))

        else:

            capture = cv2.VideoCapture(video_file)
            while not capture.isOpened():
                capture = cv2.VideoCapture(video_file)
                cv2.waitKey(1000)
                print("Wait for the header")

            # pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            # pos_frame = capture.get(cv2.CV_CAP_PROP_POS_FRAMES)
            pos_frame = capture.get(1)
            count = 0
            while True:
                flag, frame = capture.read()

                if flag:
                    cv2.imshow('video', frame)
                    # pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                    # pos_frame = capture.get(cv2.CV_CAP_PROP_POS_FRAMES)
                    pos_frame = capture.get(1)
                    # print str(pos_frame)+" frames"

                    img_output = algorithm.apply(frame)
                    img_bgmodel = algorithm.getBackgroundModel()
                    fg_count.append(len(np.where(img_output == 255)[0]))
                    cv2.imshow('img_output', img_output)
                    # cv2.imshow('img_bgmodel', img_bgmodel)
                    count += 1
                    capture.set(cv2.CAP_PROP_POS_FRAMES, count*10)

                else:
                    # capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
                    # capture.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
                    # capture.set(1, pos_frame-1)
                    # print "Frame is not ready"
                    cv2.waitKey(100)
                    break

                if 0xFF & cv2.waitKey(10) == 27:
                    break

                # if capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
                # if capture.get(cv2.CV_CAP_PROP_POS_FRAMES) == capture.get(cv2.CV_CAP_PROP_FRAME_COUNT):
                # if capture.get(1) == capture.get(cv2.CV_CAP_PROP_FRAME_COUNT):
                    # break
            fg_per_frame = sum(fg_count)/count
            result["video_name"] = video_name
            result["fg_per_frame"] = fg_per_frame
            result["fg"] = fg_count
    master_result.append(result)
    timestamp = time.strftime("%d-%m%-HH-%MM", time.localtime(time.time()))
    master_result_df = pd.DataFrame(master_result)
    master_result_df.to_csv(
        f"/home/vajra/Ongoing/flickering/master_result_{timestamp}.csv")

print("Finished")
cv2.destroyAllWindows()

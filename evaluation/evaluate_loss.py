# W4995-dl Colorization with attention
# Evaluate the loss as 1) RMSE, 2) PSNR peak signal to noise ratio, 
# 3) deviation in avg color saturation
# Others in consideration: percent deviation in class rebalancing, boundary limitation, boundary smoothness

from skimage import io, color, measure
import numpy as np
import sys
import glob


def RMSE_lab(pred, gtruth):
    #@pred, gtruth: nested arrays
    #OR RMSD? class rebalancing?
    loss = []
    m = pred[0].shape[0] * 3
    for i in range(len(pred)):
        pred_lab = color.rgb2lab(pred[i])
        gtruth_lab = color.rgb2lab(gtruth[i])
        #loss.append((pred[i][1]-gtruth[i][1])**2 + (pred[i][2]-gtruth[i][2])**2)
        loss.append(measure.compare_mse(gtruth_lab,pred_lab))
    return "RMSE (LAB): %f" % (np.sqrt(sum(loss))/(np.sqrt(m)*len(loss)))


def L2_AuC(pred,gtruth):
	for i in range(0, len(pred)):
		pred_lab = color.rgb2lab(pred[i]).reshape(len(pred[i])**2,3)
		gtruth_lab = color.rgb2lab(gtruth[i]).reshape(len(pred[i])**2,3)
		loss =  (pred_lab[:,1]-gtruth_lab[:,1])**2 + (pred_lab[:,2]-gtruth_lab[:,2])**2
		bins = np.arange(151)
		#his, edges = np.histogram(loss,bins)
		his = his/len(loss)

def PSNR_rgb(pred, gtruth):
	loss = []
	for i in range(0, len(pred)):
		loss.append(measure.compare_psnr(gtruth[i],pred[i]))
	return "PSNR (RGB): %f" % (sum(loss)/len(loss))

def saturation_hsv(pred, gtruth):
	loss = []
	for i in range(0, len(pred)):
		pred_hsv = color.rgb2hsv(pred[i])
		gtruth_hsv = color.rgb2hsv(gtruth[i])
		sat_pred = np.sum(pred_hsv[:,1])
		sat_truth = np.sum(gtruth_hsv[:,1])
		loss.append(abs(sat_pred-sat_truth)/sat_truth)
	return "Percent Saturation Deviation (HSV): %f" % (np.sum(loss)/len(loss))

#def color_bleed():

def get_files(folder):
    """
    Given path to folder, returns list of files in it
    """
    filenames = [file for file in glob.glob(folder+'*/*')]
    filenames.sort()
    return filenames

def get_img_array(path):
    """
    Given path of image, returns it's numpy array
    """
    #return scipy.misc.imread(path)
    return io.imread(path)

def get_images(folder):
    """
    returns numpy array of all samples in folder
    each column is a sample resized to 30x30 and flattened
    """
    files = get_files(folder)
    images = []
    count = 0
    
    for f in files:
        count += 1
        if count % 10000 == 0:
            print("Loaded {}/{}".format(count,len(files)))
        img_arr = get_img_array(f)
        #img_arr = img_arr.flatten() / 255.0
        images.append(img_arr)
    #X = np.column_stack(images)

    return images


#pred_rgb = get_images(sys.argv[1])
#gtruth_rgb = get_images(sys.argv[2])
#print(RMSE_lab(pred_rgb, gtruth_rgb))
#print(PSNR_rgb(pred_rgb,gtruth_rgb))
#print(saturation_hsv(pred_rgb,gtruth_rgb))



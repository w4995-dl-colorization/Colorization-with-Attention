# W4995-dl Colorization with attention
# Evaluate the loss as 1) sum of l2 norms, aka MSE, 2) PSNR peak signal to noise ratio, 
# 3) deviation in avg color saturation
# Others in progress: percent deviation in class rebalancing, boundary limitation, boundary smoothness

from skimage import io, color, measure
import numpy as np


def L2_lab(pred, gtruth):
	#@pred, gtruth: nested arrays
	#OR RMSD? class rebalancing?
	loss = []
	for i in range(0, len(pred)):
		pred_lab = color.rgb2lab(pred[i])
		gtruth_lab = color.rgb2lab(gtruth[i])
		#loss.append((pred[i][1]-gtruth[i][1])**2 + (pred[i][2]-gtruth[i][2])**2)
		loss.append(measure.compare_mse(gtruth_lab[i],pred_lab[i]))
	return sum(loss)/len(loss)

def L2_AuC(pred,gtruth):
	for i in range(0, len(pred)):
		pred_lab = color.rgb2lab(pred[i]).reshape(len(pred[i])**2,3)
		gtruth_lab = color.rgb2lab(gtruth[i]).reshape(len(pred[i])**2,3)
		loss =  (pred_lab[:,1]-gtruth_lab[:,1])**2 + (pred_lab[:,2]-gtruth_lab[:,2])**2
		bins = np.arange(151)
		his, edges = np.histogram(loss,bins)
		his = his/len(loss)


def PSNR_rgb(pred, gtruth):
	loss = []
	for i in range(0, len(pred)):
		loss.append(measure.compare_psnr(gtruth[i],pred[i]))
	return sum(loss)/len(loss)

def saturation_hsv(pred, gtruth):
	loss = []
	for i in range(0, len(pred)):
		pred_hsv = color.rgb2hsv(pred[i])
		gtruth_hsv = color.rgb2hsv(gtruth[i])
		sat_pred = sum(pred_hsv[i][:,1])
		sat_truth = sum(gtruth_hsv[i][:,1])
		loss.append(abs(sat_pred-sat_truth)/sat_truth)
	return sum(loss)/len(loss)

#def color_bleed():


pred_rgb = io.imread(filename_pred)
gtruth_rgb = io.imread(filename_gtruth)



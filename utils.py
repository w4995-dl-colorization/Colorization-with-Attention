from skimage import color
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import os
import sklearn.neighbors as nn
import warnings
import configparser
# *****************************
# ***** Utility functions *****
# *****************************


def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size == 1):
        if(inds == val):
            return True
    return False


def na():  # shorthand for new axis
    return np.newaxis


def flatten_nd_array(pts_nd, axis=1):
    """ Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd      N0xN1x...xNd array
        axis        integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array
    """
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)

    # Find all non-axis indices
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))

    # Calculate the dimension besides the axis dimension
    # by multiplying all non-axis dimensions
    NPTS = np.prod(SHP[nax])

    # Transpose the axes so the axis to be the last dimension
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    pts_flt = pts_nd.transpose((axorder))

    # Flatten the transposed ndarray
    pts_flt = pts_flt.reshape(NPTS, SHP[axis])
    return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    """ Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array except that
                    the axis dimension has dimension M
    """
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)

    # Find all non-axis indices
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[axis]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[axis]

        # Reshape to NaxNbx...xM
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)

        # Transpose to N0xN1x...xNd array except that
        # the axis dimension has dimension M
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out


class NNEncode():
    """ Encode points using NN search and Gaussian kernel
    """

    def __init__(self, k, sigma, km_filepath):
        """
        k: the number of nearest neighbors used
        sigma: sigma of the Gaussian kernel
        km_filepath: the path to the file that has a list of [a, b]
                     which are [left, bottom] starting values for
                     a 10x10 bin on the gamut
        """

        self.cc = np.load(km_filepath)
        self.num_of_bins = self.cc.shape[0]
        self.k = k
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.cc)
        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        """
        Encode each pixel of a batch of images into a distribution over 10
        neighbor bins among the 313 bins on the ab gamut
        INPUTS
          pts_nd: [N, 2, H, W]
          axis:   integer
        OUTPUTS
          pts_enc_nd: [N, 2, H, W]
        """
        # pts_flt [N*H*W, 2]
        pts_flt = flatten_nd_array(pts_nd, axis=axis)

        # P=N*H*W
        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            # pts_enc_flt [P, 313]
            self.pts_enc_flt = np.zeros((P, self.num_of_bins))
            # p_inds [P, 1]
            self.p_inds = np.arange(0, P, dtype='int')[:, na()]

        # Find distances and indices of k-NN for each pixel in pts_flt
        # dists [P, self.k]
        # inds  [P, self.k]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        # Calculate the probability/weight of each neighbor
        # for each pixel in pts_flt
        # wts  [P, self.k]
        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts, axis=1)[:, na()]
        
        # print(self.pts_enc_flt.shape, self.p_inds.shape, inds.shape, wts.shape)
        self.pts_enc_flt[self.p_inds, inds] = wts

        # pts_enc_nd [N, 313, H, W]
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)

        return pts_enc_nd

    # def decode_points_mtx_nd(self, pts_enc_nd, axis=1):
    #     pts_enc_flt = flatten_nd_array(pts_enc_nd, axis=axis)
    #     pts_dec_flt = np.dot(pts_enc_flt, self.cc)
    #     pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis=axis)
    #     return pts_dec_nd

    # def decode_1hot_mtx_nd(self, pts_enc_nd, axis=1, returnEncode=False):
    #     pts_1hot_nd = nd_argmax_1hot(pts_enc_nd, axis=axis)
    #     pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd, axis=axis)
    #     if(returnEncode):
    #         return (pts_dec_nd, pts_1hot_nd)
    #     else:
    #         return pts_dec_nd


def _nnencode(data_ab_ss):
    '''Encode groundtruth ab into 313bin ab gamut with gaussian distribution(sigma=5)
    Args:
      data_ab_ss: [N, H, W, 2]
    Returns:
      gt_ab_313 : [N, H, W, 313]
    '''
    NN = 10
    sigma = 5.0
    enc_dir = './resources/'

    # transpose data_ab_ss to [N, 2, H, W]
    data_ab_ss = np.transpose(data_ab_ss, (0, 3, 1, 2))
    nnenc = NNEncode(NN, sigma, km_filepath=os.path.join(enc_dir, 'pts_in_hull.npy'))

    gt_ab_313 = nnenc.encode_points_mtx_nd(data_ab_ss, axis=1)

    gt_ab_313 = np.transpose(gt_ab_313, (0, 2, 3, 1))

    return gt_ab_313


# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class ClassRebalance():
    ''' Class rebalancing for giving more weights to rare colors '''
    def __init__(self, alpha=1, gamma=0.5, verbose=False, priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        # self.prior_probs (313,)
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        # self.uni_probs (313,)
        self.uni_probs = np.ones_like(self.prior_probs)
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        # self.prior_mix (313,)
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set weights for each color bin
        # self.weights (313,)
        self.weights = self.prior_mix**-self.alpha
        # re-normalize
        self.weights = self.weights/np.sum(self.prior_probs*self.weights)

        # # implied empirical prior
        # self.implied_prior = self.prior_probs*self.weights
        # # re-normalize
        # self.implied_prior = self.implied_prior/np.sum(self.implied_prior)

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)' % (self.alpha, self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)' % (np.min(self.weights), np.max(self.weights), np.mean(self.weights), np.median(self.weights), np.sum(self.weights*self.prior_probs)))

    def forward(self, data_ab_quant, axis=1):
        """
        data_ab_quant (N, 313, H, W)
        """
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)

        # data_ab_maxind (N, 1, H, W)

        corr_factor = self.weights[data_ab_maxind]
        if(axis == 0):
            return corr_factor[na(), :]
        elif(axis == 1):
            return corr_factor[:, na(), :]
        elif(axis == 2):
            return corr_factor[:, :, na(), :]
        elif(axis == 3):
            return corr_factor[:, :, :, na()]


def _prior_boost(gt_ab_313):
    """
    Get the bin that is closest to each pixel
    Args:
      gt_ab_313: (N, H, W, 313)
    Returns:
      prior_boost: (N, H, W, 1)
    """
    enc_dir = './resources'
    gamma = 0.5
    alpha = 1.0

    pc = ClassRebalance(alpha, gamma, priorFile=os.path.join(enc_dir, 'prior_probs.npy'))

    # gt_ab_313 (N, 313, H, W)
    gt_ab_313 = np.transpose(gt_ab_313, (0, 3, 1, 2))
    # prior_boost (N, 1, H, W)
    prior_boost = pc.forward(gt_ab_313, axis=1)
    # prior_boost (N, H, W, 1)
    prior_boost = np.transpose(prior_boost, (0, 2, 3, 1))
    return prior_boost


def preprocess(data):
    '''Preprocess
    Args:
      data: RGB batch (N * H * W * 3)
    Return:
      data_l: L channel batch (N * H * W * 1)
      gt_ab_313: ab discrete channel batch (N * H/4 * W/4 * 313)
      prior_color_weight_nongray: the weight after rebalancing of each non-gray pixel
      at each image in the batch (N * H/4 * W/4 * 1)
    '''

    warnings.filterwarnings("ignore")

    # rgb2lab
    img_lab = color.rgb2lab(data)

    # slice
    # l: [0, 100]
    img_l = img_lab[:, :, :, :1]
    # ab: [-110, 110]
    data_ab = img_lab[:, :, :, 1:]

    # scale img_l to [-50, 50]
    data_l = img_l - 50

    # subsample 1/4  (N * H/4 * W/4 * 2)
    data_ab_ss = data_ab[:, ::4, ::4, :]

    # NonGrayMask {N, 1, 1, 1}
    thresh = 5
    nongray_mask = (np.sum(np.sum(np.sum(np.abs(data_ab_ss) > thresh, axis=1),
                    axis=1), axis=1) > 0)[:, na(), na(), na()]

    # NNEncoder
    # Find the weights given to 10 nearest bins on the ab gamut
    # of each pixel on images in a batch
    # by applying soft-encoding with a gaussian kernel
    # gt_ab_313: [N, H/4, W/4, 313]
    gt_ab_313 = _nnencode(data_ab_ss)

    # Prior_Boost
    # Get the weight of each pixel
    # prior_boost: [N, 1, H/4, W/4]
    prior_boost = _prior_boost(gt_ab_313)

    # Get the weight of each non-gray pixel
    # prior_color_weight_nongray: [N, H/4, W/4, 1]
    prior_color_weight_nongray = prior_boost * nongray_mask


    return data_l, gt_ab_313, prior_color_weight_nongray


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)  # only difference


# Combine gray-scale image and colorization into a rgb image
def decode(data_l, conv8_313, temperature=1):
    """
    Args:
      data_l   : [1, height, width, 1], real gray image (the l channel)
      conv8_313: [1, height/4, width/4, 313], predicted distribution
                 at each color bin
      temperature: a value between 0 and 1 that balance mode and mean in
                   the color bin distribution as discussed in sec2.3.
    Returns:
      img_rgb  : [height, width, 3], predicted colorized image
    """
    data_l = data_l + 50
    _, height, width, _ = data_l.shape

    # data_l (height/4, width/4, 1)
    data_l = data_l[0, :, :, :]

    # conv8_313 (height/4, width/4, 313)
    conv8_313 = conv8_313[0, :, :, :]
    enc_dir = './resources'
    
    conv8_313_rh = conv8_313/temperature


    class8_313_rh = softmax(conv8_313_rh)

    # Load color bin and combine them according to the predicted
    # distribution on color bins for each pixel
    cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))

    data_ab = np.dot(class8_313_rh, cc)

    # height/4 x width/4 -> height x width
    data_ab = resize(data_ab, (height, width))

    # Integrate color into the image
    img_lab = np.concatenate((data_l, data_ab), axis=-1)
    img_rgb = color.lab2rgb(img_lab)

    return img_rgb


def get_data_l(image_path):
    """
    Args:
      image_path
    Returns:
      data_l
    """
    data = imread(image_path)
    data = data[None, :, :, :]
    img_lab = color.rgb2lab(data)
    img_l = img_lab[:, :, :, 0:1]
    data_l = img_l - 50
    data_l = data_l.astype(dtype=np.float32)
    return data, data_l


def process_config(conf_file):
    """process configure file to generate CommonParams, DataSetParams, NetParams
    Args:
      conf_file: configure file path
    Returns:
      CommonParams, DataSetParams, NetParams, SolverParams
    """
    common_params = {}
    dataset_params = {}
    net_params = {}
    solver_params = {}

    # configure_parser
    config = configparser.ConfigParser()
    config.read(conf_file)

    # sections and options
    for section in config.sections():
        # construct common_params
        if section == 'Common':
            for option in config.options(section):
                common_params[option] = config.get(section, option)
        # construct dataset_params
        if section == 'DataSet':
            for option in config.options(section):
                dataset_params[option] = config.get(section, option)
        # construct net_params
        if section == 'Net':
            for option in config.options(section):
                net_params[option] = config.get(section, option)
        # construct solver_params
        if section == 'Solver':
            for option in config.options(section):
                solver_params[option] = config.get(section, option)

    return common_params, dataset_params, net_params, solver_params

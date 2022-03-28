import cv2
import logging
import numpy as np
import os
import os.path as osp
import png as pypng
import torch
import tqdm

from math import ceil
from imageio import imread
from pycocotools.mask import encode, decode, toBbox, area


def bbox_iou(boxA, boxB):
    """
    bbox in the form of [x1,y1,x2,y2]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# ==============================================================
# G-IoU, adapted from:
# https://github.com/generalized-iou/Detectron.pytorch/blob/master/lib/utils/net.py
# =============================================================
def compute_giou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(box1)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)

    return miouk


def open_flow_png_file(file_path_list):
    # Decode the information stored in the filename
    flow_png_info = {}
    for file_path in file_path_list:
        file_token_list = os.path.splitext(file_path)[0].split("_")
        minimal_value = int(file_token_list[-1].replace("minimal", ""))
        flow_axis = file_token_list[-2]
        flow_png_info[flow_axis] = {'path': file_path,
                                    'minimal_value': minimal_value}

    # Open both files and add back the minimal value
    for axis, flow_info in flow_png_info.items():
        png_reader = pypng.Reader(filename=flow_info['path'])
        flow_2d = np.vstack(list(map(np.float32, png_reader.asDirect()[2])))

        # Add the minimal value back
        flow_2d = flow_2d.astype(np.float32) + flow_info['minimal_value']

        flow_png_info[axis]['flow'] = flow_2d

    # Combine the flows
    flow_x = flow_png_info['x']['flow']
    flow_y = flow_png_info['y']['flow']
    flow = np.stack([flow_x, flow_y], 2)

    return flow


def warp_flow(img, flow, binarize=True):
    """
    Use the given optical-flow vector to warp the input image/mask in frame t-1,
    to estimate its shape in frame t.
    :param img: (H, W, C) numpy array, if C=1, then it's omissible. The image/mask in previous frame.
    :param flow: (H, W, 2) numpy array. The optical-flow vector.
    :param binarize:
    :return: (H, W, C) numpy array. The warped image/mask.
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    if binarize:
        res = np.equal(res, 1).astype(np.uint8)
    return res


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def calculate_flow(net, im1_fn, im2_fn):
    im_all = [imread(img) for img in [im1_fn, im2_fn]]
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H / divisor) * divisor)
    W_ = int(ceil(W / divisor) * divisor)
    for i in range(len(im_all)):
        im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
        im_all[_i] = im_all[_i][:, :, ::-1]
        im_all[_i] = 1.0 * im_all[_i] / 255.0

        im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
        im_all[_i] = torch.from_numpy(im_all[_i])
        im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
        im_all[_i] = im_all[_i].float()

    with torch.no_grad():
        # im_all = torch.autograd.Variable(torch.cat(im_all, 1).cuda(), volatile=True)
        im_all = torch.cat(im_all, 1).cuda()

        flo = net(im_all)
        flo = flo[0] * 20.0
        flo = flo.cpu().data.numpy()

        # scale the flow back to the input size
        flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)  #
        u_ = cv2.resize(flo[:, :, 0], (W, H))
        v_ = cv2.resize(flo[:, :, 1], (W, H))
        u_ *= W / float(W_)
        v_ *= H / float(H_)
        flo = np.dstack((u_, v_))

    return flo


def create_log(level=20):
    logger = logging.getLogger()
    logger.setLevel(level)
    console_handler = get_console_handler()
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger


# get logging console handler
def get_console_handler(format=None):
    console_handler = logging.StreamHandler()
    if format is None:
        formatter = logging.Formatter('%(asctime)-15s - %(levelname)s - %(message)s')
    else:
        formatter = format
    console_handler.setFormatter(formatter)
    return console_handler


# get logging file handler
def get_file_handler(file, format=None):
    file_handler = logging.FileHandler(file)
    if format is None:
        formatter = logging.Formatter('%(asctime)-15s - %(levelname)s - %(message)s')
    else:
        formatter = format
    file_handler.setFormatter(formatter)
    return file_handler


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def removefile(f):
    if os.path.exists(f):
        os.remove(f)


def frames2video(pathIn, pathOut, fps):
    """Convert continous single frames to a video.
    Args:
        pathIn (str): Example: "/home/yang/frames/"
        pathOut (str): Example: "/home/yang/frames/output_video.mp4"
        fps (int): frame rate per second of the output video
    """
    outdir = "/".join(pathOut.split("/")[:-1])
    if not os.path.exists(outdir):
        print(outdir, "does not exist, creating it...")
        os.makedirs(outdir)

    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f)) and f != '._.DS_Store']

    # for sorting the file names properly
    # files.sort(key=lambda x: int(x[5:-4]))
    files.sort()

    for i in tqdm.tqdm(range(len(files))):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # print(filename)

        # inserting the frames into an image array
        # frame_array.append(img)
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def remove_overlap_SegmentedObject(proposals, bbox_format):
    """
    Args:
        proposals: List[SegmentedObject], proposals should be sorted according specific scoring criterion.
                               higher score -> lower score.
        bbox_format: (str) Either "xywh" or "x1y1x2y1"
    Returns:
        selected_props, List[Dict]
    """
    proposals.sort(key=lambda obj: obj.score, reverse=True)
    masks = [decode(prop.mask) for prop in proposals]
    # idx = [i for i in range(len(proposals))]
    labels = np.arange(1, len(proposals) + 1)
    png = np.zeros_like(masks[0])

    # Put the mask there in reversed order, so that the latter one would just cover the previous one,
    # and the latter one has higher score. (Because proposals are sorted)
    for i in reversed(range(len(proposals))):
        png[masks[i].astype("bool")] = labels[i]

    refined_masks = [(png == id_).astype(np.uint8) for id_ in labels]
    refined_segmentations = [encode(np.asfortranarray(refined_mask)) for refined_mask in refined_masks]
    selected_props = []
    for prop, refined_segmentation, mask in zip(proposals, refined_segmentations, refined_masks):
        refined_segmentation['counts'] = refined_segmentation['counts'].decode("utf-8")
        if area(refined_segmentation) == 0:
            continue
        prop.mask = refined_segmentation
        bbox = toBbox(refined_segmentation).tolist()  # in the form of [xc, yc, w, h]
        if bbox_format == "x1y1x2y2":
            # convert [xc, yc, w, h] to [x1, y1, x2, y2]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        prop.bbox = bbox

        selected_props.append(prop)

    return selected_props


def print_config(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_config(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


# from https://github.com/TAO-Dataset/tao/blob/2930ebe6aca440a867698c0f4f34b8ae4a42b921/tao/utils/colormap.py#L26
def colormap(rgb=False, as_int=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    if as_int:
        color_list = color_list.astype(np.uint8)
    return color_list

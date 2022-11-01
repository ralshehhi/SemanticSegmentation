import argparse
import ctypes
import glob
import imp
import os
import re
import time
from multiprocessing import Array
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
from scipy import misc
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import model_from_json

from functions import *

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_db', type=str, default='/scratch/ra130/Projects/DeepGlobe/Data/Data/test')
    parser.add_argument('--model_dir', type=str, default='/scratch/ra130/Projects/DeepGlobe/Results/simple_vgg16_2_64_adam_categorical_crossentropy_0_2019-03-07_16-17-33')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--source_size', type=int, default=256)
    parser.add_argument('--label_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=2)

    return parser.parse_args()


def create_minibatch(args, img, queue):
    for d in range(0, args.label_size // 2, (args.label_size // 2) // args.offset):
        minibatch = []
        for y in range(d, args.h_limit, args.label_size):
            for x in range(d, args.w_limit, args.label_size):
                if (((y + args.source_size) > args.h_limit) or
                        ((x + args.source_size) > args.w_limit)):
                    break
                # img patch
                o_patch = img[
                    y:y + args.source_size, x:x + args.source_size, :].astype(
                    np.float32, copy=False)
                o_patch -= o_patch.reshape(-1, 3).mean(axis=0)
                o_patch /= o_patch.reshape(-1, 3).std(axis=0) + 1e-5 # (256,256,3)
                #o_patch = o_patch.transpose((2, 0, 1))
                #print('o_patch:', o_patch.shape)
                minibatch.append(o_patch)
                if len(minibatch) == args.batchsize:
                    queue.put(np.asarray(minibatch, dtype=np.float32))
                    minibatch = []
        queue.put(np.asarray(minibatch, dtype=np.float32))
    queue.put(None)


def tile_patches(args, canvas, queue):
    for d in range(0, args.label_size // 2, (args.label_size // 2) // args.offset):
        st = time.time()
        for y in range(d, args.h_limit, args.label_size):
            for x in range(d, args.w_limit, args.label_size):
                if (((y + args.source_size) > args.h_limit) or
                        ((x + args.source_size) > args.w_limit)):
                    break
                pred = queue.get()
                print('pred:', pred.shape)
                if pred is None:
                    break
                if pred.ndim == 3:
                    pred = pred.transpose((1, 2, 0)) #(256,256,7)
                    canvas[y:y + args.label_size, x:x + args.label_size, :] += pred
                else:
                	canvas[y:y + args.label_size, x:x + args.label_size, 0] += pred
                    
        print('offset:{} ({} sec)'.format(d, time.time() - st))


def get_predict(args, img, model):
   
    args.h_limit, args.w_limit = img.shape[0], img.shape[1]
    h_num = int(np.floor(args.h_limit / args.label_size)) # 2448/256 = 9
    w_num = int(np.floor(args.w_limit / args.label_size))
    args.canvas_h = h_num * args.label_size - \
        (args.source_size - args.label_size) + args.offset - 1
    args.canvas_w = w_num * args.label_size - \
        (args.source_size - args.label_size) + args.offset - 1
        
    canvas_ = Array(
        ctypes.c_float, args.canvas_h * args.canvas_w * args.num_classes)
    canvas = np.ctypeslib.as_array(canvas_.get_obj())
    canvas = canvas.reshape((args.canvas_h, args.canvas_w, args.num_classes))

    patch_queue = Queue()
    preds_queue = Queue()
    patch_worker = Process(
        target=create_minibatch, args=(args, img, patch_queue))
    canvas_worker = Process(
        target=tile_patches, args=(args, canvas, preds_queue))
    patch_worker.start()
    canvas_worker.start()

    while True:
        minibatch = patch_queue.get()
        if minibatch is None:
            break
        preds = model.predict(minibatch, batch_size=64, verbose=0)
        [preds_queue.put(pred) for pred in preds]

    preds_queue.put(None)
    patch_worker.join()
    canvas_worker.join()

    canvas = canvas[args.offset - 1:args.canvas_h - (args.offset - 1),
                    args.offset - 1:args.canvas_w - (args.offset - 1)]
    canvas /= args.offset
    return canvas


if __name__ == '__main__':
    args = get_args()
    model_dir = args.model_dir
    model_json = model_dir + '/epoch-'+str(args.epoch)+'.model.json'
    model_h5 = 'epoch-'+str(args.epoch)+'.state_weights.h5'
    model = model_from_json(open(model_json).read())
    model.load_weights(os.path.join(os.path.dirname(model_json), model_h5))
    
    data_db = args.data_db
    
    for fn in glob.glob('{}/*.jpg*'.format(data_db)):
		img = misc.imread(fn)
		pred = get_predict(args, img, model)
		pred_index, pred_rgb = convert_multilabel2rgb(pred)
		
		epoch = re.search('epoch-([0-9]+)', model_json).groups()[0]
		if args.offset > 1:
		    out_dir = '{}/ma_prediction_epoch{}'.format(model_dir, epoch)
		else:
		    out_dir = '{}/prediction_epoch{}'.format(model_dir, epoch)
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)
		
		out_fn = '{}/{}mask_pred.png'.format(out_dir,os.path.splitext(os.path.basename(fn[:-7]))[0])
		misc.imsave(out_fn, pred_rgb)
		
		out_fn = '{}/{}mask_pred.npy'.format(out_dir,os.path.splitext(os.path.basename(fn[:-7]))[0])
		np.save(out_fn, pred_index)
		

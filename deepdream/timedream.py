"""
Module containing deep dream functions. Most have been adapted from
https://github.com/tensorflow/tensorflow/blob/afee9b880a386dfa78756e861241483a15bf22ce/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

"""
from __future__ import print_function
import os
import numpy as np
from functools import partial
import PIL.Image
import PIL
import cPickle as pkl
import tensorflow as tf


def load_inception_model(model_fn='tensorflow_inception_graph.pb'):
    """Load a tensorflow model into the default graph."""
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})
    
    graph = tf.get_default_graph()

    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))
    
    return graph, t_input, [l.split('/')[1] for l in layers]


def tffunc(*argtypes):
    """Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    """
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


def resize(img, size):
    """Helper function that uses TF to resize an image."""
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, t_input, tile_size=512):
    """Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations."""
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = t_grad.eval({t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_deepdream(t_obj, t_input, img0,
        iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    """Deep dream an image with the given objective."""
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad, t_input)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = ' ')

    return img / 255.0


def render_deepdream_series(img, t_input, n_frames, loss,
        iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    """Deep dream a series of frames independently."""
    frames = []
    for i in range(n_frames):
        print('Frame %d' % i)

        frame_loss = loss(i) if callable(loss) else loss
        kwargs = dict(
            iter_n = iter_n(i) if callable(iter_n) else iter_n,
            step = step(i) if callable(step) else step,
            octave_n = octave_n,
            octave_scale = octave_scale
        )

        d_img = render_deepdream(frame_loss, t_input, img, **kwargs)
        d_img = np.uint8(np.clip(d_img, 0, 1) * 255)
        frames.append(d_img)

    return np.stack(frames)


if __name__ == '__main__':
    import dreams
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Data handling parameters
    parser.add_argument('--img', dest='img', type=str, default='museum.jpg', help='image filename')
    parser.add_argument('--frames', '-n', dest='n_frames', type=int, default=1, help='number of frames to dream')
    parser.add_argument('--dream', '-d', dest='dream', type=str, required=True, help='the name of the dream to run')
    parser.add_argument('--print_layers', '-p', dest='print_layers', action='store_true',
                        help='print model layers and exit')
    parser.add_argument('--frames_dir', '-f', dest='frames_dir', default=None, help='directory name to save frames')
    parser.add_argument('--frames_pkl', '-o', dest='frames_pkl', default=None, help='pickle file name to save frames')
    parser.add_argument('--frames_gif', '-g', dest='frames_gif', default=None, help='gif file name to save frames')
    args = parser.parse_args()
    
    # Create a default session
    sess = tf.InteractiveSession()
    
    # Build model
    graph, t_input, layers = load_inception_model()
    
    if args.print_layers:
        for l in layers:
            print(l)
        exit(0)

    # Open dream image
    img = np.float32(PIL.Image.open(args.img))
    
    # Get dream function with specified name and render frames
    t_obj, kwargs = dreams.DREAMS[args.dream](args.n_frames)
    frames = render_deepdream_series(img, t_input, args.n_frames, t_obj, **kwargs)

    # Save frame images
    if args.frames_dir:
        if not os.path.isdir(args.frames_dir):
            os.makedirs(args.frames_dir)
        for i, frame in enumerate(frames):
            PIL.Image.fromarray(frame).save(os.path.join(args.frames_dir, str(i) + '.jpg'))

    # Save tensor pickle
    if args.frames_pkl:
        pkl.dump(frames, open(args.frames_pkl, 'w'))

    # Save gif
    if args.frames_gif:
        import pkl2gif
        pkl2gif.save_tensor_as_gif(frames, args.frames_gif)


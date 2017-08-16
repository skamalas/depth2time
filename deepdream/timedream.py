"""
Many of the functions below have been adapted from
https://github.com/tensorflow/tensorflow/blob/afee9b880a386dfa78756e861241483a15bf22ce/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
"""
from __future__ import print_function
import os
import numpy as np
import PIL.Image
import PIL
import cPickle as pkl
import tensorflow as tf
import re


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


def calc_grad_tiled(img, t_grad, t_input, tile_size=512, t_neighbor=None, neighbor_img=None):
    """Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations."""
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)

    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    if neighbor_img is not None:
        neighbor_img_shift = np.roll(np.roll(neighbor_img, sx, 1), sy, 0)

    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            
            if neighbor_img is None:
                g = t_grad.eval({t_input: sub})
            else:
                n_sub = neighbor_img_shift[y:y+sz,x:x+sz]
                g = t_grad.eval({t_input: sub, t_neighbor: n_sub})

            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_deepdream(t_obj, t_input, img0,
        iter_n=10, step=1.5, octave_n=4, octave_start=1, octave_scale=1.4):
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
        if octave < octave_start:
            continue
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad, t_input)
            img += g*(step / (np.abs(g).mean()+1e-7))

    return img / 255.0


def render_deepdream_series(img, t_input, n_frames, loss,
        iter_n=10, step=1.5, octave_n=4, octave_start=1, octave_scale=1.4, use_prev=False):
    """Deep dream a series of frames independently."""
    frames = []
    for i in range(n_frames):
        print('Frame %d' % i)

        frame_loss = loss(i) if callable(loss) else loss
        kwargs = dict(
            iter_n = iter_n(i) if callable(iter_n) else iter_n,
            step = step(i) if callable(step) else step,
            octave_n = octave_n(i) if callable(octave_n) else octave_n,
            octave_start = octave_start(i) if callable(octave_start) else octave_start,
            octave_scale = octave_scale,
            
        )

        d_img = render_deepdream(frame_loss, t_input, img, **kwargs)
        if use_prev:
            img = d_img * 255.
        d_img = np.uint8(np.clip(d_img, 0, 1) * 255)
        frames.append(d_img)

    return np.stack(frames)


def get_next_naming_number(directory, name, ext):
    """Get the next available number to append to a filename to avoid name collision."""
    files = os.listdir(directory)
    pat = '^%s([0-9]+).%s$' % (name, ext)
    matches = map(lambda f: re.match(pat, f), files)
    matches = filter(lambda x: x, matches)
    nums = map(lambda m: int(m.group(1)), matches)
    if nums:
        num = max(nums) + 1
    else:
        num = 0
        
    return os.path.join(directory, '%s%d.%s' % (name, num, ext))


def get_layer_by_name(layer):
    """Helper for getting layer output tensor."""
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name("import/%s:0" % layer)


def first_zero_lambda(fn, v=0):
    """Create a function that returns a fixed value when called with 0.
    
    Otherwise defer to the passed in function. This is helpful
    to maintain a fixed value for the first frame of a series.
    """
    def wrap(i):
        if i == 0:
            return v
        else:
            return fn(i)
    return wrap


def piecewise_lambda(l, n_frames):
    """Create a lambda function that indexs into a list."""
    def wrap(i):
        ind = len(l) * i // n_frames
        return l[ind]
    return wrap


######################################
#   Deep dream parameter functions   #
######################################
"""
Functions below can be referenced by name via the --dream param to provide
a loss tensor and a dictionary of deep dream parameters. Each function can
take a desired number of frames and a string that can be parsed to provide
additional arguments/parameters from the commandline. This allows the logic
in the function to determine the dream parameters but also enabled convenient
control from the commandline for experimentation and customization.
"""

def layers_fast_dream(n_frames, dream_params):
    """Continue on each frame, iterate deeply."""
    loss_layers = dream_params.split(',')
    print('Loss layers:', loss_layers)

    losses = [tf.reduce_sum(tf.square(get_layer_by_name(x))) for x in loss_layers]
    frame_loss = piecewise_lambda(losses, n_frames)

    return frame_loss, dict(
        iter_n=first_zero_lambda(lambda i: 100),
        step=1.5,
        octave_n=4,
        use_prev=True
    )


def layers_fast_dream_independent(n_frames, dream_params):
    """Start over on each frame, iterate deeply."""
    loss_layers = dream_params.split(',')
    print('Loss layers:', loss_layers)

    losses = [tf.reduce_sum(tf.square(get_layer_by_name(x))) for x in loss_layers]
    frame_loss = piecewise_lambda(losses, n_frames)

    return frame_loss, dict(
        iter_n=first_zero_lambda(lambda i: 100),
        step=1.5,
        octave_n=4,
        use_prev=False
    )

def iteration_layers_dream(n_frames, dream_params):
    """Continue on each frame, shallow iteration"""
    loss_layers = dream_params.split(',')
    print('Loss layers:', loss_layers)

    losses = [tf.reduce_sum(tf.square(get_layer_by_name(x))) for x in loss_layers]
    frame_loss = piecewise_lambda(losses, n_frames)

    return frame_loss, dict(
        iter_n=first_zero_lambda(lambda i: 3),
        step=1.5,
        octave_n=4,
        use_prev=True
    )


# Named parameter template functions
DREAMS = {
    'layers_fast_dream': layers_fast_dream,
    'layers_fast_dream_independent': layers_fast_dream_independent,
    'iteration_layers_dream': iteration_layers_dream,
}


if __name__ == '__main__':
    import dreams
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--img', dest='img', type=str, help='image filename')
    parser.add_argument('--frames', '-n', dest='n_frames', type=int, default=1, help='number of frames to dream')
    parser.add_argument('--dream', '-d', dest='dream', type=str, help='the name of the dream to run')
    parser.add_argument('--print_layers', '-p', dest='print_layers', action='store_true',
                        help='print model layers and exit')
    parser.add_argument('--frames_dir', '-f', dest='frames_dir', default=None, help='directory name to save frames')
    parser.add_argument('--frames_pkl', '-o', dest='frames_pkl', default=None, help='pickle file name to save frames')
    parser.add_argument('--frames_gif', '-g', dest='frames_gif', default=None, help='gif file name to save frames')
    parser.add_argument('--dream_params', '-e', dest='dream_params', default=None, help='')
    args = parser.parse_args()

    # Create a default session
    sess = tf.InteractiveSession()
    
    # Build model
    graph, t_input, layers = load_inception_model()
    t_neighbor = tf.placeholder(tf.float32, name='neighbor_input')
    
    # Print a list of layer names available for dreaming
    if args.print_layers:
        for l in layers:
            print(l)
        exit(0)

    # Open dream image
    img = np.float32(PIL.Image.open(args.img))
    
    # Get dream function with specified name and render frames
    t_obj, kwargs = DREAMS[args.dream](args.n_frames, args.dream_params)
    frames = render_deepdream_series(img, t_input, args.n_frames, t_obj, **kwargs)

    # Save frame images
    if args.frames_dir:
        if not os.path.isdir(args.frames_dir):
            os.makedirs(args.frames_dir)
        for i, frame in enumerate(frames):
            PIL.Image.fromarray(frame).save(os.path.join(args.frames_dir, str(i) + '.jpg'))

    # Automatically create a filename based on the source if only an output
    # directory was provided
    source_name = os.path.splitext(os.path.basename(args.img))[0]

    # Save tensor pickle
    if args.frames_pkl:
        if os.path.isdir(args.frames_pkl):
            args.frames_pkl = get_next_naming_number(args.frames_pkl, source_name, 'pkl')
        pkl.dump(frames, open(args.frames_pkl, 'w'), -1)
        print('Saved ' + args.frames_pkl)

    # Save gif
    if args.frames_gif:
        import pkl2gif
        if os.path.isdir(args.frames_gif):
            args.frames_gif = get_next_naming_number(args.frames_gif, source_name, 'gif')
        pkl2gif.save_tensor_as_gif(frames, args.frames_gif)
        print('Saved ' + args.frames_gif)


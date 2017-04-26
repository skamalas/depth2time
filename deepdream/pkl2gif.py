"""
Script to convert a pickled numpy of frame rgb data into
a gif. E.g.

    python pkl2gif tensor.pkl
    
will create a tensor.gif file in the current directory.
Frame indexes can be passed in to save out jpg frames
instead, e.g.

    python pkl2gif tensor.pkl 0 9 19
    
will create 3 files: 
    tensor.frame_0.jpg
    tensor.frame_9.jpg
    tensor.frame_19.jpg
"""
import os
import imageio
import cPickle as pkl
import PIL.Image


def save_tensor_as_gif(t, fn):
    imageio.mimsave(fn, t)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pkl', type=str, nargs=1)
    parser.add_argument('frame', type=int, nargs='?')
    args = parser.parse_args()

    print 'Loading tensor'
    fn = args.pkl[0]
    d = pkl.load(open(fn))
    
    out_fn_prefix = os.path.splitext(fn)[0]
    if args.frame is None:
        out_fn = out_fn_prefix + '.gif'
        save_tensor_as_gif(d, out_fn)
        print 'Wrote ' + out_fn
    else:
        frame = args.frame[0]
        out_fn = out_fn_prefix + '.frame_%d.jpg' % frame
        PIL.Image.fromarray(d[frame]).save(out_fn)
        print 'Wrote ' + out_fn

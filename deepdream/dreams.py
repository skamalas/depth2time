import inspect
import sys
import numpy as np
import tensorflow as tf
import timedream


def T(layer):
    """Helper for getting layer output tensor."""
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name("import/%s:0" % layer)


def index_lamba(d):
    """Helper to build a lambda that indexes into a list."""
    return lambda i: d[i]


def make_peak(r, offset, length):
    d = np.concatenate([np.linspace(0, 1, r), np.linspace(1, 0, r)])
    x = offset - r
    if x > 0:
        d = np.concatenate([np.zeros(x), d])
    else:
        d = d[-x:]
    
    if len(d) < length:
        d = np.concatenate([d, np.zeros(length - len(d))])
    else:
        d = d[:length]
        
    return d


def loss_list_lambda(losses, steps, r=None):
    if r is None:
        r = steps // len(losses)
    
    w = steps // len(losses)
    lines = [make_peak(r, w*(i+1), steps) for i in range(len(losses))]

    def fn(i):
        x = np.array([l[i] for l in lines])
        xs = x.sum()
        if xs != 0:
            x /= xs
        return tf.reduce_sum(tf.stack([x[j] * losses[j] for j in range(len(losses))]))

    return fn


def first_zero_lambda(fn):
    def wrap(i):
        if i == 0:
            return 0
        else:
            return fn(i)
    return wrap


######################################

def iteration_dream(n_frames):
    return tf.square(T('mixed4c')), dict(
        iter_n=index_lamba(np.linspace(10, 30, n_frames, dtype=np.int32))
    )


def layers_dream(n_frames):
    loss_layers = ['mixed3a_3x3_pre_relu', 'mixed3b_3x3_pre_relu', 'mixed4a_3x3_pre_relu',
                   'mixed4b_3x3_pre_relu', 'mixed4c_3x3_pre_relu', 'mixed5b_3x3_pre_relu']
    losses = [tf.reduce_sum(tf.square(T(x))) for x in loss_layers]
    return loss_list_lambda(losses, n_frames), dict(
        iter_n=first_zero_lambda(lambda x: 50)
    )


######################################

# Collect functions that contain "dream" in their name
DREAMS = dict()
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isfunction(obj) and 'dream' in name:
        DREAMS[name] = obj


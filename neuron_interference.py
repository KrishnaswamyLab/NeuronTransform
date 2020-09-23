import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sklearn.metrics
import os
import sklearn.neighbors


###################################################################
###################################################################
###################################################################
# load data
def get_data_artificial():
    f_matrix1 = [-.75, 0]
    f_matrix2 = [0, .75]

    def f(iinput):
        iinput = np.where(iinput[:, 0][:, np.newaxis] < .75, iinput + f_matrix1, iinput + f_matrix2)
        return iinput

    x = np.concatenate([makegmm(-1.25, 0., 200), makegmm(1.25, 2, 200), makegmm(2.25, 2, 100)], axis=0)
    source = np.concatenate([makegmm(-1.25, 0, 250), makegmm(1.25, 2, 250)], axis=0)

    y = f(x)
    target = f(source)

    return x, source, target, y

def makegmm(x, y, n, sd=[.1, .1]):
    x = np.random.normal(x, sd[0], (n, 1))
    y = np.random.normal(y, sd[1], (n, 1))

    return np.concatenate([x, y], axis=1)

def calculate_mmd(dist1, dist2):

    def calculate_mmd_(k1, k2, k12):

        return k1.sum()/(k1.shape[0]*k1.shape[1]) + k2.sum()/(k2.shape[0]*k2.shape[1]) - 2*k12.sum()/(k12.shape[0]*k12.shape[1])

    k1 = sklearn.metrics.pairwise.pairwise_distances(dist1, dist1)
    k2 = sklearn.metrics.pairwise.pairwise_distances(dist2, dist2)
    k12 = sklearn.metrics.pairwise.pairwise_distances(dist1, dist2)

    mmd = 0
    for sigma in [.01, .1, 1., 10.]:
        k1_ = np.exp(-k1 / sigma**2)
        k2_ = np.exp(-k2 / sigma**2)
        k12_ = np.exp(-k12 / sigma**2)

        mmd += calculate_mmd_(k1_, k2_, k12_)
    return mmd

def build_config(limit_gpu_fraction=0.2, limit_cpu_fraction=10):
    if limit_gpu_fraction > 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config = tf.ConfigProto(device_count={'GPU': 0})
    if limit_cpu_fraction is not None:
        if limit_cpu_fraction <= 0:
            # -2 gives all CPUs except 2
            cpu_count = min(
                1, int(os.cpu_count() + limit_cpu_fraction))
        elif limit_cpu_fraction < 1:
            # 0.5 gives 50% of available CPUs
            cpu_count = min(
                1, int(os.cpu_count() * limit_cpu_fraction))
        else:
            # 2 gives 2 CPUs
            cpu_count = int(limit_cpu_fraction)
        config.inter_op_parallelism_threads = cpu_count
        config.intra_op_parallelism_threads = cpu_count
        os.environ['OMP_NUM_THREADS'] = str(1)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    return config

def lrelu(x, leak=0.2, name="lrelu"):

    return tf.maximum(x, leak * x)

def tbn(name):

    return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):

    return tf.get_default_graph().get_operation_by_name(name)

def nameop(op, name):

    return tf.identity(op, name=name)


MODEL = 'NeuronEdit'
DATA = 'artificial'
print('Training {} {}'.format(MODEL, DATA))

x, source, target, y = get_data_artificial()

allinput = np.concatenate([x, source, target, y], axis=0)
labels = np.concatenate([0 * np.ones(x.shape[0]), 1 * np.ones(source.shape[0]), 2 * np.ones(target.shape[0]), 3 * np.ones(y.shape[0])], axis=0)
r = list(range(labels.shape[0]))
np.random.shuffle(r)

class NeuronEdit(object):
    def __init__(self, input_dim, hdim, name=''):
        self.name = name
        self.input_dim = input_dim
        self.hdim = hdim

        self.build()

    def mlp(self, x, hdim=200, neurons=None):
        h1 = tf.layers.dense(x, hdim * 4, activation=lrelu, name='h1')
        h2 = tf.layers.dense(h1, hdim * 2, activation=lrelu, name='h2')
        h3 = tf.layers.dense(h2, hdim, activation=lrelu, name='h3')
        edit_h = h3
        if neurons is not None:
            h3 = neurons
        h4 = tf.layers.dense(h3, hdim * 2, activation=lrelu, name='h4')
        h5 = tf.layers.dense(h4, hdim * 4, activation=lrelu, name='h5')
        recon = tf.layers.dense(h5, input_dim, activation=None, name='recon')

        return edit_h, recon

    def build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            tfsource = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='tfsource')
            tftarget = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='tftarget')
            tfx = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='tfx')

            source_neurons = tf.placeholder(tf.float32, shape=[None, self.hdim], name='source_neurons')
            target_neurons = tf.placeholder(tf.float32, shape=[None, self.hdim], name='target_neurons')
            x_neurons = tf.placeholder(tf.float32, shape=[None, self.hdim], name='x_neurons')

            # mask = tf.random.categorical(tf.math.log([(np.ones(hdim * 2) / np.ones(hdim * 2).sum()).tolist()]), tf.shape(tfx)[0])
            # mask = 1 - tf.one_hot(mask, hdim * 2)
            # mask = tf.cast(mask[0], tf.float32)
            mask = 1

            with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
                tfneurons_source, recon_source = self.mlp(tfsource, hdim=self.hdim, neurons=None)
                tfneurons_target, recon_target = self.mlp(tftarget, hdim=self.hdim, neurons=None)
                tfneurons_x, recon_x = self.mlp(tfx, hdim=self.hdim, neurons=None)

                _, recon_source_edited = self.mlp(tfsource, hdim=self.hdim, neurons=source_neurons)
                _, recon_target_edited = self.mlp(tftarget, hdim=self.hdim, neurons=target_neurons)
                _, recon_x_edited = self.mlp(tfx, hdim=self.hdim, neurons=x_neurons)

                _, recon_source_edited2 = self.mlp(tfsource, hdim=self.hdim, neurons=tfneurons_source * mask)
                _, recon_x_edited2 = self.mlp(tfx, hdim=self.hdim, neurons=tfneurons_x * mask)


            nameop(tfneurons_source, 'tfneurons_source')
            nameop(tfneurons_target, 'tfneurons_target')
            nameop(tfneurons_x, 'tfneurons_x')
            nameop(recon_source, 'recon_source')
            nameop(recon_target, 'recon_target')
            nameop(recon_x, 'recon_x')
            nameop(recon_source_edited, 'recon_source_edited')
            nameop(recon_target_edited, 'recon_target_edited')
            nameop(recon_x_edited, 'recon_x_edited')

            # loss
            self.loss = 0.
            self.loss += tf.reduce_mean((tfsource - recon_source)**2)
            self.loss += tf.reduce_mean((tftarget - recon_target)**2)
            self.loss += tf.reduce_mean((tfx - recon_x)**2)
            nameop(self.loss, 'loss')

            # optimization
            opt = tf.train.AdamOptimizer(.001)
            self.train_op = opt.minimize(self.loss, name='train_op')

            # session
            self.sess = tf.Session(config=build_config(limit_gpu_fraction=.1))
            self.sess.run(tf.global_variables_initializer())

    def train(self, source, target, x):
        feed = {tbn('tfsource:0'): source, tbn('tftarget:0'): target, tbn('tfx:0'): x}
        self.sess.run(obn('train_op'), feed_dict=feed)

    def get_loss(self, source, target, x):
        feed = {tbn('tfsource:0'): source, tbn('tftarget:0'): target, tbn('tfx:0'): x}
        loss = self.sess.run(tbn('loss:0'), feed_dict=feed)

        return loss

    def get_recon(self, source, target, x):
        feed = {tbn('tfsource:0'): source, tbn('tftarget:0'): target, tbn('tfx:0'): x}
        s, t, x = self.sess.run([tbn('recon_source:0'), tbn('recon_target:0'), tbn('recon_x:0')], feed_dict=feed)

        return s, t, x

    def get_neurons(self, source, target, x):
        feed = {tbn('tfsource:0'): source, tbn('tftarget:0'): target, tbn('tfx:0'): x}
        s, t, x = self.sess.run([tbn('tfneurons_source:0'), tbn('tfneurons_target:0'), tbn('tfneurons_x:0')], feed_dict=feed)

        return s, t, x

    def get_recon_edited(self, neurons):
        feed = {tbn('x_neurons:0'): neurons}
        x = self.sess.run(tbn('recon_x_edited:0'), feed_dict=feed)

        return x

    def calc_mi_emd(self, neuronx, neurony, nbins=10):
        both_batches = np.concatenate([neuronx, neurony], axis=0)
        emd = np.zeros((both_batches.shape[1]))

        for i in range(both_batches.shape[1]):
            bins = np.linspace(both_batches[:,i].min(), both_batches[:,i].max(), nbins)

            countsx, _ = np.histogram(neuronx[:,i], bins=bins)
            countsx = countsx / countsx.sum()
            countsx = countsx.cumsum()

            countsy, _ = np.histogram(neurony[:,i], bins=bins)
            countsy = countsy / countsy.sum()
            countsy = countsy.cumsum()

            emd[i] = np.abs(countsx - countsy).sum()

        return emd

    def earth_mover_transform(self, knbors, out, b1, b2, bins=100):
        out = out.copy()
        # percentiles_out = [np.percentile(out, 100 * i / bins) for i in range(0, bins + 1)]
        percentiles_b1 = [np.percentile(b1, 100 * i / bins) for i in range(0, bins + 1)]
        percentiles_b2 = [np.percentile(b2, 100 * i / bins) for i in range(0, bins + 1)]

        b1 = b1.reshape((-1, 1))
        out = out.reshape((-1, 1))

        if knbors is None:
            knn = sklearn.neighbors.NearestNeighbors(1)
            knn.fit(b1)
            knbors = knn.kneighbors(out, return_distance=False)

        pctiles = np.zeros((b1.shape[0], 1))
        for i in range(len(percentiles_b1) - 1):
            pctile = np.logical_and(percentiles_b1[i] < b1, b1 < percentiles_b1[i + 1])
            pctiles = np.where(pctile, i, pctiles)


        # for i, k in enumerate(knbors):
        #     k = k[ii]
        #     ii = int(pctiles[k])
        #     old = .5 * (percentiles_b1[ii] + percentiles_b1[ii + 1])
        #     shift = .5 * (percentiles_b2[ii] + percentiles_b2[ii + 1]) - old
        #     out[i, 0] = out[i, 0] + shift

        for i, kk in enumerate(knbors):
            shifts = []
            for iii in range(knbors.shape[1]):
                k = kk[iii]
                ii = int(pctiles[k])
                old = .5 * (percentiles_b1[ii] + percentiles_b1[ii + 1])
                shift = .5 * (percentiles_b2[ii] + percentiles_b2[ii + 1]) - old
                shifts.append(shift)
            shift = np.stack(shifts, axis=0)
            out[i, 0] = out[i, 0] + np.mean(shift, axis=0)

        return out.reshape(-1)

###################################################################
tf.reset_default_graph()
batch_size = 50
hdim = 50
input_dim = x.shape[1]

model = NeuronEdit(input_dim=input_dim, hdim=hdim)

###################################################################
# training
for e in range(50):
    r1 = list(range(source.shape[0])); np.random.shuffle(r1);
    r2 = list(range(target.shape[0])); np.random.shuffle(r2);
    r3 = list(range(x.shape[0])); np.random.shuffle(r3);
    per_epoch = min(len(r1), len(r2), len(r3)) // batch_size

    for i in range(per_epoch):
        source_ = source[r1[i:i + batch_size]]
        target_ = target[r2[i:i + batch_size]]
        x_ = x[r3[i:i + batch_size]]

        model.train(source_, target_, x_)

    print(e, model.get_loss(source_, target_, x_))

    outsource, outtarget, outx = model.get_recon(source, target, x)

    allout = np.concatenate([x, outsource, outtarget, y], axis=0)
    outlabels = np.concatenate([0 * np.ones(x.shape[0]), 1 * np.ones(source.shape[0]), 2 * np.ones(target.shape[0]), 3 * np.ones(y.shape[0])], axis=0)
    r = list(range(outlabels.shape[0])); np.random.shuffle(r)

    fig.clf()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(allout[r, 0], allout[r, 1], c=outlabels[r], s=5, cmap=mpl.cm.coolwarm)
    fig.canvas.draw()


###################################################################
# evaluation
neurons_source, neurons_target, neurons_x = model.get_neurons(source, target, x)

emd_spk = model.calc_mi_emd(neurons_source, neurons_target, nbins=100)
neurons_to_interfere = np.arange(emd_spk.shape[0])[np.argsort(emd_spk, axis=0)].reshape(-1)[::-1]

K = 1
knn = sklearn.neighbors.NearestNeighbors(K)
knn.fit(neurons_source)
knbors1 = knn.kneighbors(neurons_x, return_distance=False)

knn = sklearn.neighbors.NearestNeighbors(K)
knn.fit(neurons_source)
knbors2 = knn.kneighbors(neurons_source, return_distance=False)

neurons_x_transformed = neurons_x.copy()
neurons_source_transformed = neurons_source.copy()
print("Interfering with {} neuron(s)".format(len(neurons_to_interfere)))
for i, neuron in enumerate(neurons_to_interfere):
    if i % 10 == 0: print("{} / {}".format(i, len(neurons_to_interfere)))

    a = neurons_x_transformed[:, neuron]
    b = neurons_source_transformed[:, neuron]
    c = neurons_target[:, neuron]
    neurons_x_transformed[:, neuron] = model.earth_mover_transform(knbors1, a, b, c, bins=100)
    neurons_source_transformed[:, neuron] = model.earth_mover_transform(knbors2, b, b, c, bins=100)

source_transformed = model.get_recon_edited(neurons=neurons_source_transformed)
x_transformed = model.get_recon_edited(neurons=neurons_x_transformed)




allout = np.concatenate([x, source, target, y], axis=0)
outlabels = np.concatenate([0 * np.ones(x.shape[0]), 1 * np.ones(source.shape[0]), 2 * np.ones(target.shape[0]), 3 * np.ones(y.shape[0])], axis=0)
r = list(range(outlabels.shape[0])); np.random.shuffle(r)
alledited = np.concatenate([x_transformed, source_transformed, target, y], axis=0)
e1 = allout
e2 = alledited

fig.clf()
ax1, ax2 = fig.subplots(1, 2, sharex=True, sharey=True)
ax1.scatter(e1[r, 0], e1[r, 1], c=outlabels[r], s=5, cmap=mpl.cm.coolwarm)

ax2.scatter(e2[r, 0], e2[r, 1], c=outlabels[r], s=5, cmap=mpl.cm.coolwarm)

fig.canvas.draw()



















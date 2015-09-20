

"""
=========================================
Segmenting the picture of Lena in regions
=========================================
This example uses :ref:`spectral_clustering` on a graph created from
voxel-to-voxel difference on an image to break this image into multiple
partly-homogeneous regions.
This procedure (spectral clustering on an image) is an efficient
approximate solution for finding normalized graph cuts.
There are two options to assign labels:
* with 'kmeans' spectral clustering will cluster samples in the embedding space
  using a kmeans algorithm
* whereas 'discrete' will iteratively search for the closest partition
  space to the embedding space.
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause


import matplotlib.image as mpimg


import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


def read_image():
    import scipy
    res = scipy.misc.imread('small_sample_gray.png', flatten = True)
    # res = scipy.misc.bytescale(res)
    return res


def read_image2():
    import scipy
    res = scipy.misc.imread('tsu-left.png', flatten = True)
    # res = scipy.misc.bytescale(res)
    return res

# lena = sp.misc.lena()
lena = read_image()

org = read_image2()

# lena = org

print lena

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 5

beta = 5
eps = 1e-6

p = 0.1

# Downsample the image by a factor of 4
# lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
# lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
print lena.shape
lena = lena.astype(float)

# lena = lena[:5, :5]
print lena.shape

# Convert the image into a graph with the value of the gradient on the
# edges.
import random

if True:
    graph = image.img_to_graph(lena, return_as=np.ndarray)
    print 'ttt'
    graph1 = image.img_to_graph(lena)
    graph2 = graph
    print type(graph1.data)
    print type(graph2.data)
    # graph2.data = np.exp(-beta * graph2.data / lena.std()) + eps

    print graph
    print graph.shape

    row = []
    col = []
    data = []

    for i in xrange(graph.shape[0]):
        for j in xrange(graph.shape[1]):
            if graph[i][j] != 0:
                row.append(i)
                col.append(j)
                data.append(graph[i][j])

    graph = sp.sparse.coo_matrix((data, (row,col)), shape=graph.shape, dtype = np.float)# .todense()# sp.sparse.coo_matrix(graph)


    # print graph.getrow(0)
    # print graph1.getrow(0)
    # graph = graph1
    # print graph.data
    # print graph1.data
    # assert graph.data == graph1.data


    setted = {}
    num_set = 0
    for i in xrange(len(graph1.data)):
        r = graph1.row[i]
        c = graph1.col[i]
        if r == c:
            continue

        assert (r, c) not in setted.keys()

        setted[(r, c)] = graph1.data[i]

        if r != c and setted.has_key((c, r)):
            graph1.data[i] = setted[(c, r)]
        elif r <= c and random.random() >= p:
            graph1.data[i] = 0
            setted[(r, c)] = 0
            num_set += 1
    graph = graph1
    print 'has set: ', num_set, ' in: ', len(graph1.data)

    # assert 0

    check = False
    if check:
        assert graph.shape == graph1.shape
        for i in xrange(graph.shape[0]):
            r = graph.getrow(i)
            r1 = graph1.getrow(i)
            a = r.toarray()
            a1 = r1.toarray()
            assert a.shape == a1.shape

            for j in xrange(a.shape[0]):
                for k in xrange(a.shape[1]):
                    assert a[j][k] == a1[j][k]
            # print type(r)
        # assert graph1 == graph
else:
    graph = image.img_to_graph(lena)

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
graph.data = np.exp(-beta * graph.data / lena.std()) + eps

# print graph
# print graph.shape

# assert 0

###############################################################################
# Visualize the resulting regions

a = ('kmeans', 'discretize')
b = ['discretize']


import time
start_time = time.time()

def getLabels(org, small_labels):
    scale = org.shape[0] / small_labels.shape[0]
    print scale
    labels = np.ones(shape = org.shape)
    print labels.shape
    for i in xrange(org.shape[0] / scale):
        for i0 in xrange(scale):
            index_i = i * scale + i0
            if index_i >= org.shape[0]:
                continue

            for j in xrange(org.shape[1] / scale):
                for j0 in xrange(scale):
                    index_j = j * scale + j0
                    if index_j >= org.shape[1]:
                        continue
                    labels[index_i][index_j] = small_labels[i][j]

    assert labels.shape == org.shape
    return labels

for assign_labels in (b):
    t0 = time.time()
    small_labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels,
                                 random_state=1)
    t1 = time.time()
    small_labels = small_labels.reshape(lena.shape)

    labels = getLabels(org, small_labels)

    plt.figure(figsize=(5, 5))
    plt.imshow(org,   cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS)), ])
    plt.xticks(())
    plt.yticks(())
    plt.title('Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0)))

endtime = time.time()
print 'cost: ', endtime - start_time

plt.show()

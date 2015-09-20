

import matplotlib.image as mpimg

from sklearn.feature_extraction import image
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import lil_matrix

import hierarchy

scale_size = 4
near_thrd = 5

def get_label_res2(similar_matrix, n_subs):

    cluster = AffinityPropagation(damping = 0.75 , affinity = 'precomputed') # preference = -1000)# n_clusters = n_subs, affinity = 'precomputed')

    res = cluster.fit(similar_matrix)

    size_labels = len(set(res.labels_))
    assert size_labels < 10, size_labels
    assert size_labels > 1, size_labels

    print res.labels_
    return res.labels_

def get_label_res(similar_matrix, n_subs):

    # cluster = AffinityPropagation(damping = 0.75)# , affinity = 'precomputed') # preference = -1000)# n_clusters = n_subs, affinity = 'precomputed')

    if True:
        labels = spectral_clustering(lil_matrix(similar_matrix), n_clusters = n_subs, eigen_solver='arpack') # affinity = 'precomputed',
        return labels
    elif False:
        cluster = SpectralClustering(n_clusters = n_subs, affinity = 'precomputed', eigen_solver='arpack')
    else:
        cluster = SpectralClustering(n_clusters = n_subs, affinity = 'nearest_neighbors', eigen_solver='arpack')

    res = cluster.fit(similar_matrix)

    size_labels = len(set(res.labels_))
    assert size_labels < 10, size_labels
    assert size_labels > 1, size_labels

    print res.labels_
    return res.labels_

def save_shrinked(shrinked_gray, scale_size):
    length = len(shrinked_gray)
    width = len(shrinked_gray[0])

    gray = np.zeros(shape=(length * scale_size, width * scale_size))

    for i in xrange(length):
        for j in xrange(width):
            for k in xrange(scale_size):
                for l in xrange(scale_size):
                    gray[i*scale_size + k][j*scale_size + l] = shrinked_gray[i][j]

    mpimg.imsave('sample_gray.png', gray, vmax= 1, vmin= 0, cmap = plt.cm.gray)
    mpimg.imsave('small_sample_gray_%d.png' % (scale_size), shrinked_gray, vmax= 1, vmin= 0, cmap = plt.cm.gray)

def near(ix, iy, jx, jy):
    manhattan_dis = abs(jy - iy) + abs(jx - ix)
    return manhattan_dis <= near_thrd

def build_simi_matrix(gray):
    length = len(gray)
    width = len(gray[0])

    shrinkted_length = length / scale_size
    shrinkted_width = width / scale_size

    shrinked_gray = np.zeros(shape = (shrinkted_length, shrinkted_width))

    for i in xrange(shrinkted_length):
        for j in xrange(shrinkted_width):
            average_gray = 0
            for k in xrange(scale_size):
                for l in xrange(scale_size):
                    average_gray += gray[i * scale_size + k][j * scale_size + l]
            average_gray = average_gray / (scale_size * scale_size)
            shrinked_gray[i][j] = average_gray

    save_shrinked(shrinked_gray, scale_size)


    matr_size = shrinkted_length * shrinkted_width

    similar_matrix = [[0] * matr_size for i in xrange(matr_size)]

    for ix in xrange(shrinkted_length):
        for iy in xrange(shrinkted_width):
            p_i = shrinked_gray[ix][iy]
            index_i = ix + iy * scale_size

            assert index_i < matr_size, '%d %d %d'%(ix, iy, index_i)

            for jx in xrange(shrinkted_length):
                for jy in xrange(shrinkted_width):
                    if not near(ix, iy, jx, jy):
                        continue

                    p_j = shrinked_gray[jx][jy]
                    index_j = jx + jy * scale_size

                    assert index_j < matr_size, '%d %d %d'%(jx, jy, index_j)

                    v = abs(p_i - p_j)
                    # v = math.exp(-(v ** 2)/0.02)
                    # print v

                    similar_matrix[index_i][index_j] = v
                    similar_matrix[index_j][index_i] = v

    return similar_matrix

def get_sub_matr(gray):
    length = len(gray)
    width = len(gray[0])

    shrinkted_length = length / scale_size
    shrinkted_width = width / scale_size

    shrinked_gray = np.zeros(shape = (shrinkted_length, shrinkted_width))

    for i in xrange(shrinkted_length):
        for j in xrange(shrinkted_width):
            average_gray = 0
            for k in xrange(scale_size):
                for l in xrange(scale_size):
                    average_gray += gray[i * scale_size + k][j * scale_size + l]
            average_gray = average_gray / (scale_size * scale_size)
            shrinked_gray[i][j] = average_gray

    save_shrinked(shrinked_gray, scale_size)
    return shrinked_gray

def get_subs(gray, labels, n_subs):

    n_subs = len(set(labels))

    sub_grays = [np.ones(shape = gray.shape) for i in xrange(n_subs)]

    length = len(gray)
    width = len(gray[0])

    shrinkted_length = length / scale_size
    shrinkted_width = width / scale_size

    matr_size = shrinkted_length * shrinkted_width

    for i in xrange(matr_size):
        if i % 100 == 0:
            print 'iter: ', i

        l_i = i % shrinkted_length * scale_size
        w_i = i / shrinkted_length * scale_size
        p_i = gray[l_i][w_i]

        label = labels[i]

        for j in xrange(scale_size):
            if l_i+j >= length:
                continue
            for k in xrange(scale_size):
                if w_i + k >= width:
                    continue
                sub_grays[label][l_i+j][w_i+k] = gray[l_i+j][w_i+k]

    return sub_grays

def build_image(gray):

    graph = image.img_to_graph(gray)
    print type(graph)
    graph.data = np.exp(-graph.data / graph.data.std())
    return graph

def cluster(gray):
    if True:
        similar_matrix = build_simi_matrix(gray)
    else:
        similar_matrix = build_image(gray)

    if False:
        sub_gray = get_sub_matr(gray)
        flatten_gray = []
        for i in xrange(len(sub_gray)):
            for j in xrange(len(sub_gray[0])):
                flatten_gray.append(sub_gray[i][j])

    n_subs = 5
    # labels = hierarchy.cluster(flatten_gray, len(sub_gray), len(sub_gray[0]), n_subs) # get_label_res(similar_matrix, n_subs)
    labels = get_label_res(similar_matrix, n_subs)

    assert len(set(labels)) <= n_subs, len(set(labels))
    sub_grays = get_subs(gray, labels, n_subs)
    return sub_grays

def read_image():
    import scipy
    res = mpimg.imread('tsu-left.png')
    # res = scipy.misc.bytescale(res)
    return res

if __name__ == "__main__":
    gray = read_image()
    print gray
    # assert 0

    sub_grays = cluster(gray)

    for i in xrange(len(sub_grays)):
        pic_name  = 'res_%d.png' % (i)
        mpimg.imsave(pic_name, sub_grays[i], vmax= 1, vmin= 0, cmap = plt.cm.gray)

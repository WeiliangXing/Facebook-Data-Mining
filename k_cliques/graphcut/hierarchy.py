

import heapq

glb_heap = []
length = 0
width = 0

connect_threshold = 10

def w_push(q, v):
    if v[0] != 1:
        heapq.heappush(q, v)

def getpos(n):
    return n % length, n / length

def node_connect(n1, n2):
    x1, y1 = getpos(n1)
    x2, y2 = getpos(n2)

    if abs(x1 - x2) + abs(y1 - y2) <= connect_threshold:
        # return abs(x1 - x2) != 2 and abs(y1 - y2) != 2
        return True
    return False

def connect(nodes1, nodes2):
    for n1 in nodes1:
        for n2 in nodes2:
            if node_connect(n1, n2):
                return True
    return False

class Group:
    def __init__(self, nodes, grays):
        self.nodes = nodes
        self.grays = grays
        self.sames = {}

    def isleaf(self):
        return len(self.nodes) == 1

    def calculate_same(self, agroup):
        if not connect(self.nodes, agroup.nodes):
            return 1

        grays_sum = 0
        for i in xrange(len(self.grays)):
            for j in xrange(len(agroup.grays)):
                grays_sum += abs(self.grays[i] - agroup.grays[j])
        res = grays_sum * 1.0 / (len(self.grays) * len(agroup.grays))
        # self.add_same(agroup, res)
        return res

    def add_same(self, agroup, same):
        # assert not self.sames.has_key(agroup)
        self.sames[agroup] = same

    def __str__(self):
        return '%s' % (self.nodes)

    def __repr__(self):
        return '%s' % (self.nodes)

def closest(all_nodes):
    item = heapq.heappop(glb_heap)
    v, x, y = item
    while x not in all_nodes or y not in all_nodes:
        item = heapq.heappop(glb_heap)
        v, x, y = item
    return all_nodes.index(x), all_nodes.index(y)
    return x, y

    res_max = 10000000.0
    res_i = 0
    res_j = 0
    for i in xrange(len(all_nodes)):
        for j in xrange(i+1, len(all_nodes)):
            same = all_nodes[i].calculate_same(all_nodes[j])
            assert isinstance(same, float), same
            assert isinstance(res_max, float), res_max
            if same < res_max:
                res_max = same
                res_i = i
                res_j = j
    return res_i, res_j

def create_group(n1, n2):
    assert isinstance(n1, Group)
    assert isinstance(n2, Group)
    new_group = Group(n1.nodes + n2.nodes, n1.grays + n2.grays)
    # print "new: ", new_group
    return new_group

def merge(all_nodes, x, y):
    # replace
    new_node = create_group(all_nodes[x], all_nodes[y])
    all_nodes[x] = new_node

    if y != len(all_nodes) - 1:
        all_nodes[y] = all_nodes[len(all_nodes) - 1]

    all_nodes = all_nodes[:-1]

    for i in all_nodes:
        if i != new_node:
            w_push(glb_heap, (new_node.calculate_same(i), new_node, i))

    return all_nodes


def cluster(grays, lent, wid, nums = 2):
    global length
    global width

    length = lent
    width = wid

    l = len(grays)

    all_nodes = []
    for i in xrange(l):
        all_nodes.append(Group([i], [grays[i]]))

    for i in xrange(l):
        for j in xrange(i+1, l):
            w_push(glb_heap, (all_nodes[i].calculate_same(all_nodes[j]), all_nodes[i], all_nodes[j]))

    while len(all_nodes) > nums:
        # print "cluster:", len(all_nodes)
        if len(all_nodes) % 100 == 0:
            print "cluster:", len(all_nodes)
        x, y = closest(all_nodes)
        all_nodes = merge(all_nodes, x, y)

        # for i in xrange(len(all_nodes)):
        #     print 'group:', i
        #     print all_nodes[i].nodes, all_nodes[i].grays
        #     print all_nodes[i].sames

    labels = [0] * l
    for i in xrange(len(all_nodes)):
        for j in xrange(len(all_nodes[i].nodes)):
            labels[all_nodes[i].nodes[j]] = i

    assert len(all_nodes) <= nums
    assert len(set(labels)) <= nums, len(set(labels))

    return labels

if __name__ == "__main__":
    array = [1, 2, 5, 7, 3, 8]
    labels = cluster(array)
    print labels
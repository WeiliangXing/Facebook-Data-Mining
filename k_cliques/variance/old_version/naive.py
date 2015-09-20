

def is_clique(mat, k):
    for i in xrange(len(k)):
        for j in xrange(i+1, len(k)):
            if mat[k[i]][k[j]] == 0:
                return False
    return True

if __name__ == "__main__":
    mat = [[0] * 145 for i in xrange(145)]
    with open('data/raw_data.txt', 'r') as f:
        for l in f:
            l = l.split(', ')
            # print l
            assert len(l) == 2
            a = int(l[0])
            b = int(l[1])
            mat[a][b] = 1
            mat[b][a] = 1

    res = 0
    for i0 in xrange(145):
        for i1 in xrange(i0, 145):
            if mat[i0][i1] == 0:
                continue
            for i2 in xrange(i1+1, 145):
                if mat[i1][i2] == 0 or mat[i0][i2] == 0:
                    continue
                for i3 in xrange(i2+1, 145):
                    if mat[i0][i3] == 0 or mat[i1][i3] == 0 or mat[i2][i3] == 0:
                        continue
                    res += 1
    result = {}
    for i in xrange(2, 11):
        result[i] = 0
        k = [0]

        while len(k) != 0:
            if k[-1] == 145:
                k = k[:-1]
                if len(k) != 0:
                    k[-1] += 1
                continue

            if not is_clique(mat, k):
                k[-1] += 1
                continue

            if len(k) < i:
                k.append(k[-1]+1)
            else:
                result[len(k)] += 1
                k[-1] += 1
    print res

    print result
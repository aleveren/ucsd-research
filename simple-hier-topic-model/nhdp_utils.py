def K_means_L1(X, K, maxite):
    D = X.shape[1]
    b = np.random.permutation(np.arange(D))
    if D >= K:
        B = X[:, b[:K]]
    else:
        B = np.random.uniform(0, 1, (X.shape[0], K))
        B /= B.sum(axis=0, keepdims=True)
    c = np.zeros(D)

    for ite in range(maxite):
        for d in range(D):
            c[d] = np.argmin(np.sum(np.abs(B - X[:, [d]]), axis=0))
        for k in range(K):
            B[:, k] = np.mean(X[:, c == k], axis=1)

    cnt = np.histogram(c, np.arange(K+1))
    t2 = np.argsort(-cnt)
    B = B[:, t2]
    c2 = np.zeros(D)
    for i in range(len(t2)):
        idx = np.nonzero(c == t2[i])[0]
        c2[idx] = i
    c = c2

    return B, c

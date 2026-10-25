"""Line-by-line Python port of the authors' APEM code (HW 5.3b).

Source: OSF archive gvcpz, Code/runapem.m with SliderGame/updatestatecapem.m,
getagreelaopt.m, crop.m, fraught.m, winningorthant.m and PWLF.m.
The Matlab names are kept in the comments so the two can be read side by side.
"""
import numpy as np


def runapem(A, atheta=0.5, stheta=0.5, meanstep=0.1, timelimit=300, rng=None):
    # one game of runapem.m; A has no self-loops (the Matlab matrix has 1s on the diagonal)
    if rng is None:
        rng = np.random.default_rng()
    n = len(A)
    Aself = A + np.eye(n)                         # adjacency used by getagreelaopt (self included)

    dirvec = rng.choice([-1.0, 1.0], size=n)      # random direction for every player
    posvec = meanstep * dirvec                    # first move: one meanstep, not random
    currtime = meanstep
    thist = [0.0, 0.0]                            # thist(2) is never set in runapem.m, so it stays 0
    shist = [np.zeros(n), posvec.copy()]
    success = False

    while not success and currtime < timelimit:   # the last step may go past the time limit
        curragree = np.abs(Aself @ posvec) / Aself.sum(axis=1)     # getagreelaopt
        timecands = rng.exponential(meanstep, size=n)              # one latency per player, mean meanstep
        nextind = np.argmin(timecands)                             # first to check in
        nextlat = timecands[nextind]
        if curragree[nextind] <= atheta and abs(posvec[nextind]) >= stheta:
            dirvec[nextind] = -dirvec[nextind]                     # switch BEFORE moving
        posvec = np.clip(posvec + nextlat * dirvec, -1, 1)         # crop
        currtime = currtime + nextlat
        thist.append(currtime)
        shist.append(posvec.copy())
        success = abs(posvec.sum()) == n                           # all at +1 or all at -1

    return {"A": A, "t": np.array(thist), "X": np.array(shist), "weights": np.ones(n), "success": success}


def fraught(currpos, A):
    # fraught.m: 1 if currpos is fraught on the graph A (no self-loops)
    if len(currpos) == abs(np.sign(currpos).sum()):    # winningorthant
        return 0
    if np.prod(currpos) == 0:
        return 0
    for p in range(len(currpos)):
        gfinds = np.sign(currpos) == np.sign(currpos[p])   # players on p's side
        fsum = abs(A[p] @ (currpos * gfinds))              # pull from neighbours on p's side
        esum = abs(A[p] @ (currpos * ~gfinds))             # pull from neighbours on the other side
        if esum > fsum:
            return 0
    return 1


def warp(run, nwarp=1000):
    # time warp of runapem.m: PWLF through the record, evaluated at nwarp + 1 evenly spaced times
    t, X = run["t"], run["X"]
    grid = np.linspace(t[0], t[-1], nwarp + 1)
    # PWLF never uses the zero-length first segment (t = 0 to t = 0), so drop row 1 and interpolate
    return np.column_stack([np.interp(grid, t[1:], X[1:, j]) for j in range(X.shape[1])])


def pfraught(run, nwarp=1000):
    # share of the warped states that are fraught (runapem.m: sum(fhist) / nrstates)
    newshist = warp(run, nwarp)
    return np.mean([fraught(s, run["A"]) for s in newshist])

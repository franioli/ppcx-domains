"""Filters for displacement/vector fields taken from LAMMA v.2024.10.03"""

import warnings

import joblib as jb
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree


def vector_field_filter(values, nodes, method="Delaunay", k=None):
    """
    cleandispmap adjusts 2D-3D displacement components according to the neighbouring values

    Parameters
    ----------
    values : list of ndarrays
        1D or 2D array containing the x, y, [z] vector components
    nodes : Nx2 array with the x,y coordinates of the displacement vectors
        or NxMx2 matrix with the coordinates of the displacement matrixes
    method : str, optional. Default is Delaunay
        can be "Delaunay", "Neighbours" or "Radius"

    Returns
    -------
    out : list of array
        it contains the filtered displacement values
        out[0] -> DX
        out[1] -> DY
        out[2] -> DZ [if values is Nx2, out[2]=None]
        out[3] -> nodes

    Other parameters
    ----------------
    k : integer, optional. Default is None
        number of neighbourood that is considered to apply the median filter

    Description
    -----------
    If the displacement data are provided as vectors:
        In the "Delaunay" case, the considered neighbourhoods of a given node
        are the nodes linked by an edge of a Delaunay triangulation. This is
        the default method
        in the "Neighbours" case the considered neighbourhoods are the closest
        k nodes. default k is 4
        in the "Radius" case, the considered neighbourhoods are those within a
        circumference of radius=k. Default radius is 1
    If the displacement are given in matrix form
        the considered neighbourhoods are the 4 closest nodes
    """

    # -some check before to begin
    for i in values:
        if type(i) is not np.ndarray:
            raise NameError("Values must be a list of numpy arrays")
    if type(nodes) is not np.ndarray:
        raise NameError("Nodes must a numpy array")

    # -check inputs
    if method not in ["Delaunay", "Neighbours", "Radius"]:
        warnings.warn(
            "WARNING: unknown/unspecified filter method. 'Delaunay' will be adopted"
        )
        method = "Delaunay"
    elif (method == "Neighbours") and (k is None):
        k = 4
    elif (method == "Radius") and (k is None):
        k = 1
    if len(values) == 2:
        X, Y = values[0], values[1]
        if X.shape != Y.shape:
            raise NameError("The sizes of the values arrays must be the same")
    elif len(values) == 3:
        X, Y, Z = values[0], values[1], values[2]
        if (X.shape != Y.shape) or (X.shape != Z.shape):
            raise NameError("The sizes of the values arrays must be the same")
    if (len(X.shape) == 1) or (
        X.shape[1] == 1
    ):  # reshape the input data as column vectors and stack them horizontally
        X = np.reshape(X, [-1, 1])
        Y = np.reshape(Y, [-1, 1])
        if len(values) == 2:
            values = np.hstack((X, Y))
        else:
            Z = np.reshape(Z, [-1, 1])
            values = np.hstack((X, Y, Z))
        if nodes.shape[1] != 2:
            raise NameError("Nodes must be a Nx2 numpy array")
        out = loopScattered(values, nodes, method, k)  # processing scattered points
    else:  # -regular grid - approx. 2.5x faster
        if len(values) == 2:
            values = np.stack((X, Y), axis=2)
        else:
            values = np.stack((X, Y, Z), axis=2)
        if len(nodes.shape) != 3:
            raise NameError("Nodes must be a NxMx2 numpy array")
        out = loopMatrix(values)  # processing matrixes
        out = out[0], out[1], out[2], nodes

    return out


# -##########################################


def loopScattered(values, nodes, method, k):
    """
    this function searches the neigbours of every node in a set of scattered nodes
    the neighbours can be searched considering different options:
        within a given radius
        the closest k-neighbours
        the connected vertex of a Delaunay triangulation
    subsequently, it applies a local median-based filter to correct the outliers

    Parameters
    ----------
    values : array-like
        it can have size Nx2 or Nx3. This is the input data to be filtered
    nodes : array like
        coordinates of the nodes
    method : str
        method to search neighbouroods : "Delaunay", "Radius", "Neighbour"
    k : integer
        number of neighbour nodes to be used to filter the outliers

    Returns
    -------
    U, V, W : array-like
        these the filtered values. If values is Nx2 => W=[]
    nannodes : this is equal to nodes
    """

    # -check whether there are NaNs in nodes
    temp = np.sum(values, axis=1)
    nanpun = np.argwhere(np.isnan(temp))
    realpun = np.argwhere(~np.isnan(temp))
    sz = nodes.shape[0]
    if len(nanpun) > 0:
        # delete nodes with NaN values
        nodes = np.delete(nodes, nanpun, 0)
        values = np.delete(values, nanpun, 0)
    if method == "Delaunay":  # create a Delaunay Triangulation
        DT = Delaunay(nodes)
        indptr, indices = DT.vertex_neighbor_vertices
        # -search the closest neighbours of every node
        O = jb.Parallel(n_jobs=CPUs // 2)(
            jb.delayed(searchScattered)(
                values[i, :], values, nodes[i, :], nodes, indptr, indices, i
            )
            for i in range(nodes.shape[0])
        )
        # -store the results in two variables
        Zi = []  # this is values
        z = []  # these are the values of the neighbourood nodes
        for i in range(len(O)):
            Zi.append(O[i][0])
            z.append(O[i][1])
    else:
        DT = KDTree(nodes)
        if method == "Radius":
            tree = KDTree(nodes)
            ind = tree.query_radius(nodes, k)
        elif method == "Neighbours":
            tree = KDTree(nodes)
            ind = tree.query(nodes, k=k, return_distance=False)
        z = []  # these are the values of the neighbourood nodes
        for i in ind:
            z.append(values[i, :])
        Zi = values

    # -apply filter
    O = jb.Parallel(n_jobs=CPUs // 2)(
        jb.delayed(loc_filter)(Zi[i], z[i]) for i in range(len(Zi))
    )
    O = np.array(O)
    O = O.squeeze()  # O is Nx2 or Nx3 array with the filtered values

    if len(nanpun) > 0:  # if there were NaNs they are reinserted
        nannodes = np.zeros(sz, 2) * np.nan
        nanValues = np.zeros((sz, values.shape[1])) * np.nan
        # reinsert NaNs
        nannodes[realpun, :] = nodes
        nanValues[realpun, :] = O
    else:
        nannodes = nodes
        nanValues = O

    if values.shape[1] == 2:
        U, V, W = nanValues[:, 0].squeeze(), nanValues[:, 1].squeeze(), []
    else:
        U, V, W = (
            nanValues[:, 0].squeeze(),
            nanValues[:, 1].squeeze(),
            nanValues[:, 2].squeeze(),
        )

    return U, V, W, nannodes


# -##########################################
def searchScattered(Zi, Z, Ni, nodes, indptr, indices, i):
    # Zi is the i-th displacement vector
    # Z is the whole set of displacement vectors
    # Ni is the i-th node
    # nodes are the coordinates of the displacement vectors
    # DT is the Delaunay triangulation
    # i is the iterate number
    # -the output is a 3-element tuple where 1) Zi: Zi, 2) z: is the set oh neighbouring
    # -vectors, and 3) are the indexes of z

    # -get the neighbours
    c = indices[indptr[i] : indptr[i + 1]]
    # -add the ith nodes to the closest ones
    c = np.append(c, i)
    # -take Z values of the val+1 nodes
    z = Z[c, :]
    # -reshape vectors
    Zi = np.reshape(Zi, [1, -1])
    z = np.reshape(z, [-1, Z.shape[1]])
    c = np.reshape(c, [-1, 1])
    # -create the tuple of the output
    O = (Zi, z, c)

    return O


# -##########################################
def loopMatrix(values):
    """
    this function searches the 4 closest neigbours of every node in a regular grid
    subsequently, it applies a local median-based filter to correct the outliers
    """

    sz1 = values.shape[0]
    sz2 = values.shape[1]
    values[np.isnan(values)] = -999
    if len(values.shape) == 3:
        dimension = 2
        X, Y = values[:, :, 0], values[:, :, 1]
        NeighX = np.ones((sz1, sz2, 9)) * (-999)
        NeighY = np.ones((sz1, sz2, 9)) * (-999)
    else:
        dimension = 3
        X, Y, Z = values[:, :, 0], values[:, :, 1], values[:, :, 2]
        NeighX = np.ones((sz1, sz2, 9)) * (-999)
        NeighY = np.ones((sz1, sz2, 9)) * (-999)
        NeighZ = np.ones((sz1, sz2, 9)) * (-999)
    # this loop could be speeded with joblib
    for i in range(1, sz1 - 1):
        for j in range(1, sz2 - 1):
            NeighX[i, j, :] = np.hstack(
                (X[i - 1, j - 1 : j + 2], X[i, j - 1 : j + 2], X[i + 1, j - 1 : j + 2])
            )
            NeighY[i, j, :] = np.hstack(
                (Y[i - 1, j - 1 : j + 2], Y[i, j - 1 : j + 2], Y[i + 1, j - 1 : j + 2])
            )
            if dimension == 3:
                NeighZ[i, j, :] = np.hstack(
                    (
                        Z[i - 1, j - 1 : j + 2],
                        Z[i, j - 1 : j + 2],
                        Z[i + 1, j - 1 : j + 2],
                    )
                )

    Zi = np.reshape(values, [-1, dimension])
    NeighX = np.reshape(NeighX, [-1, 9])
    NeighY = np.reshape(NeighY, [-1, 9])
    if dimension == 3:
        NeighZ = np.reshape(NeighZ, [-1, 9])

    z = []
    for i in range(Zi.shape[0]):
        if dimension == 2:
            temp = np.zeros((9, 2))
        else:
            temp = np.zeros((9, 3))
        for j in range(9):
            if dimension == 2:
                temp[j, :] = NeighX[i, j], NeighY[i, j]
            else:
                temp[j, :] = NeighX[i, j], NeighY[i, j], NeighZ[i, j]
        z.append(temp)

    # -apply filter
    O = jb.Parallel(n_jobs=CPUs // 2)(
        jb.delayed(loc_filter)(Zi[i], z[i]) for i in range(len(Zi))
    )
    # -refine the results
    O = np.array(O)
    O = O.squeeze()

    if dimension == 2:
        U, V = O[:, 0], O[:, 1]
        W = []
    else:
        U, V, W = O[:, 0], O[:, 1], O[:, 2]
        W = np.reshape(W, [values.shape[0], values.shape[1]])

    U = np.reshape(U, [values.shape[0], values.shape[1]])
    V = np.reshape(V, [values.shape[0], values.shape[1]])

    return U, V, W


# -##########################################


def loc_filter(Zi, z):
    """
    this filter applies to the multidimensional vector 'Zi'
    it calculates the reciprocal distance 'di' of a set of neighouring vectors 'z'
    and selects the 50% of vectors with the lowest 'di'
    if 'Zi' does not belong to the selected vectors it is replaced with their median
    """

    # -WARNING: np.linalg.norm requires a lot of calculi when z caontins many elements (e.g. >10)
    # -if loc_filter() has to be computed over a very large set of nodes,
    # -it can takes a while to complete

    # -determine the reciprocal distance
    di = np.zeros((z.shape[0], z.shape[0]))
    for ii in range(z.shape[0]):
        for jj in range(z.shape[0]):
            di[ii, jj] = np.linalg.norm(z[ii, :] - z[jj, :])
    # -take the 50% of the val+1 vector with the lowest reciprocal distance
    num = int(np.round(z.shape[0] / 2))
    ptr = np.argpartition(np.mean(di, axis=1), num)
    ptr = ptr[:num]
    # -if the ith nodes is included in the selected vectors, keep it unchanged
    # -if not, replace it with thir median value
    if Zi in z[ptr]:
        out = Zi
    else:
        out = np.median(z[ptr, :], axis=0)

    return out

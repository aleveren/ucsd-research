import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

def extract(m, diagnostics = None, depth_penalty = None, apply_cond = True):
    remaining_indices = set(range(m.shape[0]))
    used_indices = set()
    m = m.copy()
    if apply_cond:
        m /= m.sum(axis=0, keepdims=True)
    root = np.argmax(np.mean(m, axis=1))
    depths = {root: 0}
    remaining_indices.remove(root)
    used_indices.add(root)
    if diagnostics is not None:
        diagnostics.append({"matrix": m, "raw_matrix": m.copy(), "root": root})
    g = nx.Graph()
    g.add_node(root)
    while len(remaining_indices) > 0:
        rem_ind_arr = np.array(sorted(list(remaining_indices)))
        used_ind_arr = np.array(sorted(list(used_indices)))
        depths_arr = np.array([depths[i] for i in used_ind_arr])
        submatrix = m[rem_ind_arr.reshape(-1,1), used_ind_arr.reshape(1,-1)]
        if depth_penalty is not None:
            submatrix = submatrix.copy() * (depth_penalty ** depths_arr[np.newaxis, :])
        argmax = np.argmax(submatrix)
        ii, jj = np.unravel_index(argmax, submatrix.shape)
        i = rem_ind_arr[ii]
        j = used_ind_arr[jj]
        depths[i] = depths[j] + 1
        g.add_edge(i, j)
        if diagnostics is not None:
            diagnostics.append({"matrix": submatrix, "selection": (ii, jj), "edge": (i, j),
                "rem_ind_arr": rem_ind_arr, "used_ind_arr": used_ind_arr, "depths_arr": depths_arr})
        remaining_indices.remove(i)
        used_indices.add(i)
    g.graph["root"] = root
    return g

def display_diagnostics(diagnostics):
    mmm = diagnostics[0]["matrix"]
    for d in diagnostics:
        fig, ax = plt.subplots()
        ax.imshow(mmm, vmin=0, vmax=1, cmap='gray_r')
        title = None
        if "selection" in d:
            for i in d["rem_ind_arr"]:
                for j in d["used_ind_arr"]:
                    ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=None, edgecolor='blue'))
            i, j = d["edge"]
            ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=None, edgecolor='red'))
            title = "Add Edge: {} --> {}".format(j, i)
            ax.set_yticks(np.arange(mmm.shape[0]))
            ax.set_yticklabels(np.arange(mmm.shape[0]))
            ax.set_xticks(np.arange(mmm.shape[1]))
            ax.set_xticklabels(np.arange(mmm.shape[1]))
        elif "root" in d:
            ax.add_patch(mpl.patches.Rectangle((-0.5, d["root"] - 0.5), d["matrix"].shape[1], 1, fill = None, edgecolor='red'))
            title = "Select Root: {}".format(d["root"])
        if title is not None:
            ax.set_title(title)
            print(title)

def display_diagnostics_submatrices(diagnostics):
    for d in diagnostics:
        fig, ax = plt.subplots()
        ax.imshow(d["matrix"], vmin=0, vmax=1, cmap='gray_r')
        title = None
        if "selection" in d:
            i, j = d["selection"]
            #print("col (x) = {}, row (y) = {}".format(i, j))
            ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=None, edgecolor='red'))
            title = "Add Edge: {} --> {}".format(d["edge"][1], d["edge"][0])
            ax.set_yticks(np.arange(len(d["rem_ind_arr"])))
            ax.set_yticklabels(d["rem_ind_arr"])
            ax.set_xticks(np.arange(len(d["used_ind_arr"])))
            ax.set_xticklabels(d["used_ind_arr"])
        elif "root" in d:
            ax.add_patch(mpl.patches.Rectangle((-0.5, d["root"] - 0.5), d["matrix"].shape[1], 1, fill = None, edgecolor='red'))
            title = "Select Root: {}".format(d["root"])
        if title is not None:
            ax.set_title(title)
            print(title)


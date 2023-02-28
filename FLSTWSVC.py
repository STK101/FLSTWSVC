import libtsvm
from libtsvm.preprocess import DataReader
from libtsvm.estimators import TSVM
from libtsvm.model_selection import Validator
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm
import math

def dist(pt: np.array, z: np.array):
    return np.abs(((z.T @ np.concatenate([pt, [1]])))/np.linalg.norm(z[:-1]))
def dist_RBF(ker_pt: np.array, z: np.array):
    # ker_pt is kernelised point
    return np.abs(z.T @ np.concatenate([ker_pt, [1]])/np.linalg.norm(z[:-1]))
    
def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    d=0
    for i in range(len(p1)):
        d=d+(p1[i]-p2[i])**2
    return d

def find(parent, i):
    """Find the parent of a node in a disjoint-set data structure."""
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, i, j):
    """Union two nodes in a disjoint-set data structure."""
    i_root = find(parent, i)
    j_root = find(parent, j)
    if i_root == j_root:
        return
    if rank[i_root] < rank[j_root]:
        parent[i_root] = j_root
    elif rank[i_root] > rank[j_root]:
        parent[j_root] = i_root
    else:
        parent[j_root] = i_root
        rank[i_root] += 1

def kruskals_algorithm(points, k):
    """Use Kruskal's algorithm to create k clusters from a set of n 2D points."""
    n = len(points)

    # Calculate the distance between all pairs of points.
    distances=[]
    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(points[i], points[j])
            distances.append((distance, i, j))

    # Sort the distances in ascending order.
    distances.sort()
    #print(distances)

    # Initialize the disjoint-set data structure.
    parent = [i for i in range(n)]
    rank = [0 for i in range(n)]

    # Merge edges until k clusters are formed.
    num_clusters = n
    for distance, i, j in distances:
        if num_clusters == k:
            break
        if find(parent, i) != find(parent, j):
            #print("clustering ",i," , ",j)
            union(parent, rank, i, j)
            num_clusters -= 1
    #make a parent to cluster map
    parent_to_cluster={}
    curclust=0
    for i in range(n):
        parent[i]=find(parent,i)
    for i in range(n):
        if(parent[i] not in parent_to_cluster):
            parent_to_cluster[parent[i]]=curclust
            curclust+=1
    # Assign each point to its cluster.
    #print(parent_to_cluster)
    #print(parent)
    clusters = [[] for i in range(k)]
    for i in range(n):
        clusters[parent_to_cluster[parent[i]]].append(points[i])

    return clusters
def kernelize(X, kernel = 'linear'):
    m, n = X.shape
    X_new = []
    for i in range(m):
        if(kernel == 'linear'):
            new_x = linear_kernel(X[i], X)
        if(kernel == 'RBF'):
            new_x = rbf_kernel(X[i], X)
        X_new.append(new_x)
    # X_new shape : (m, m)
    return np.array(X_new)
def linear_kernel(x, X):
    # X shape = (m, n), such that m is the number of points and n is the dimensionality
    # x shape = (n, )
    return X @ x

def rbf_kernel(x, X, gamma = 1):
    # X shape = (m, n), such that m is the number of points and n is the dimensionality
    # x shape = (n, )
    return np.exp(-gamma * (np.linalg.norm(X - x.reshape(1, -1), axis = 1)**2))

def w_b_S_init(Points, k, kernel = 'linear'):
    clusters=kruskals_algorithm(Points,k)
    X=np.zeros((len(Points),len(Points[0])))
    y=np.zeros(len(Points))
    curclust=0;
    curpoint=0;
    for cluster in clusters:
        for point in cluster:
            X[curpoint,:]=np.array(point)
            y[curpoint]=curclust
            curpoint+=1
            curclust+=1
    
    ycpy=np.ones(y.shape[0])
    weights=[]
    new_X = kernelize(X, kernel)
    #print(weights.shape)
    b=np.zeros(len(clusters))
    for i in range(len(clusters)):
        ycpy[np.where(y==i)]=-1
        tsvm_clf = TSVM(kernel='linear')
        if(kernel == 'RBF'):
            tsvm_clf.fit(new_X,ycpy)
        else:
            tsvm_clf.fit(X,ycpy)
        
        weights.append(tsvm_clf.w2[:,0])
        b[i]=tsvm_clf.b2
        print(np.linalg.norm(tsvm_clf.w2[:,0]), b[i])
        ycpy=np.ones(y.shape[0])
    weights = np.array(weights).T
    #initialise S
    S=np.zeros((X.shape[0],k))
    for i in range(X.shape[0]):
        for j in range(k):
        #print(X[i,:].shape)
        #print(weights[:,j].shape)
        #print(np.linalg.norm(weights[:,j],2))
            if(kernel == 'linear'):
                S[i][j]= 1/(1 + dist(X[i], np.concatenate([weights[:, j], [b[j]]])))
            else:
                S[i][j] = 1/(1 + dist(new_X[i], np.concatenate([weights[:, j], [b[j]]])))
    return S, X, y,weights,b

def get_points(z: np.array, x1 = 1., x2 = 2., ym1 = 1.,ym2 = 2.):

    y1 = (-z[2] - x1*z[0])/(z[1])
    x1n = x1
    if (y1 < ym1):
        x1n = x1*(((ym1 * z[1]) + z[2])/((y1 * z[1]) + z[2]))
        y1 = ym1
    elif (y1 > ym2):
        x1n = x1*(((ym2 * z[1]) + z[2])/((y1 * z[1]) + z[2]))
        y1 = ym2
    y2 = (-z[2] - x2*z[0])/(z[1])
    x2n = x2
    if (y2 > ym2):
        x2n = x2*(((ym2 * z[1]) + z[2])/((y2 * z[1]) + z[2]))
        y2 = ym2
    elif (y2 < ym1):
        x2n = x2*(((ym1 * z[1]) + z[2])/((y2 * z[1]) + z[2]))
        y2 = ym1
    return [x1n, x2n], [y1, y2]

def plot_y(X, y, num_cluster, Z = None):
    colors=["red","green","blue","yellow","black",'teal','indigo']
    plt.scatter(X[:,0],X[:,1],c=[colors[int(x)] for x in y])
    if(Z is not None):
        for c in range(num_cluster):
            xs, ys = get_points(Z[c], min_x, max_x)
            plt.plot(xs, ys, colors[c])
    plt.show()
def get_X_S(X, y, S, num_cluster):
    Xs = []
    X_bars = []
    Ss = []
    S_bars = []
    for c in range(num_cluster):
        Xs.append(X[np.where(y == c)])
        X_bars.append(X[np.where(y != c)])
        Ss.append(np.diag(S[np.where(y == c), c][0]))
        S_bars.append(np.diag(S[np.where(y != c), c][0]))
    return Xs, X_bars, Ss, S_bars

def run_algorithm(X, num_cluster, C = 0.001, v = 0.001, num_iters = 1000, init_lines_plot = False, kernel = 'linear'):
    num_points = X.shape[0]
    colors=["red","green","blue","black",'teal','indigo',"yellow"]
    Ys = []
    Zs = []

    # Initialization
    S, ordered_points, y, weights,b = w_b_S_init(X, num_cluster, kernel = kernel)
    ws = np.array(weights).T
    bs = np.array(b).reshape(-1, 1)
    Z_old = np.concatenate([ws, bs], 1)
    Zs.append(Z_old)
    min_x = np.min(ordered_points[:,0])
    max_x = np.max(ordered_points[:,0])
    min_y = np.min(ordered_points[:,1])
    max_y = np.max(ordered_points[:,1])
    # Plotting initial cluster
    if(init_lines_plot):
        for c in range(num_cluster):
                xs, ys = get_points(Z_old[c], min_x, max_x, min_y,max_y)
                plt.plot(xs, ys, colors[c])
        plt.scatter(ordered_points[:,0],ordered_points[:,1],c=[colors[int(x)] for x in y])
        plt.savefig('pics/iter0000.png')
        plt.show()

    X = np.array(ordered_points)
    if(kernel == 'RBF'):
        new_X = kernelize(X, kernel)
    y = np.array(y)
    Ys.append(y)
    #assert(np.allclose(y, np.argmax(S, axis = 1)))
    S_old = np.array(S)

    if(kernel == 'linear'):
        new_X = X
        # Update Algorithm
        iter = 0
        for iter in tqdm(range(num_iters)):
            Z_new = []
            Xs, X_bars, Ss, S_bars = get_X_S(X, y, S_old, num_cluster)

            # Compute weights and biases for each cluster
            for c in range(num_cluster):
                H1 = np.concatenate([Ss[c] @ Xs[c], Ss[c] @ np.ones((Ss[c].shape[0], 1))], axis = 1)
                H2 = np.concatenate([S_bars[c] @ X_bars[c], S_bars[c] @ np.ones((S_bars[c].shape[0], 1))], axis = 1)
                G_sub = (H2 @ Z_old[c]).reshape(-1)
                G = np.diag(G_sub/np.linalg.norm(G_sub))
                m, n = H1.shape
                Z_new.append(C * np.linalg.inv((H1.T @ H1) + np.eye(n)*v + C*(H2.T @ H2)) @ H2.T @ G.T @ np.ones((G.shape[0])))
            
            
            # Calculate new fuzzy matrix and cluster assignments
            S_new = np.array([[1/(1 + dist(X[i], Z_new[j])) for j in range(num_cluster)] for i in range(num_points)])
            y = np.array([np.argmax(S_new[i]) for i in range(num_points)])
            Ys.append(y)
            Zs.append(Z_new)

            if(np.linalg.norm(S_old - S_new) < 1e-3):
                break
            
            Z_old = Z_new
            S_old = S_new
                # Plotting initial cluster
            if(init_lines_plot):
                for c in range(num_cluster):
                        xs, ys = get_points(Z_old[c], min_x, max_x ,min_y,max_y)
                        plt.plot(xs, ys, colors[c])
            
                plt.scatter(ordered_points[:,0],ordered_points[:,1],c=[colors[int(x)] for x in y])
                plt.savefig('pics/iter' + str(iter+1).zfill(4) + '.png')
                plt.show()

        
        print(f'Ended at Iter {iter + 1}')

    else: 
        # Update Algorithm
        new_X = kernelize(X, 'RBF')
        iter = 0
        for iter in tqdm(range(num_iters)):
            Z_new = []
            new_Xs, new_X_bars, Ss, S_bars = get_X_S(new_X, y, S_old, num_cluster)

            # Compute weights and biases for each cluster
            for c in range(num_cluster):
                H1 = E1 = np.concatenate([Ss[c] @ new_Xs[c], Ss[c] @ np.ones((Ss[c].shape[0], 1))], axis = 1)
                H2 = E2 = np.concatenate([S_bars[c] @ new_X_bars[c], S_bars[c] @ np.ones((S_bars[c].shape[0], 1))], axis = 1)
                F_sub = (E2 @ Z_old[c]).reshape(-1)
                G = F = np.diag(F_sub/np.linalg.norm(F_sub))
                m, n = H1.shape
                Z_new.append(C * np.linalg.inv((H1.T @ H1) + np.eye(n)*v + C*(H2.T @ H2)) @ H2.T @ G.T @ np.ones((G.shape[0])))
                # P = np.linalg.inv(v/C + (E2 @ E2.T)) 
                # V = (1/v)*(np.eye(E2.shape[1]) - (E2.T @ P @ E2))
                # R = np.linalg.inv(np.eye(E1.shape[0]) + (E1 @ V @ E1.T))
                # Z_new.append(C*(V - (V @ E1.T @ R @ E1 @ V) ) @ E2.T @ F.T @ np.ones((F.shape[0])))
            
            # Calculate new fuzzy matrix and cluster assignments
            S_new = np.array([[1/(1 + dist(new_X[i], Z_new[j])) for j in range(num_cluster)] for i in range(num_points)])
            y = np.array([np.argmax(S_new[i]) for i in range(num_points)])
            Ys.append(y)
            Zs.append(Z_new)

            if(np.linalg.norm(S_old - S_new) < 1e-3):
                break
            
            Z_old = Z_new
            S_old = S_new
            if(init_lines_plot):
                for c in range(num_cluster):
                        xs, ys = get_points(Z_old[c], min_x, max_x, min_y,max_y)
                        plt.plot(xs, ys, colors[c])
                plt.scatter(ordered_points[:,0],ordered_points[:,1],c=[colors[int(x)] for x in y])
                plt.savefig('pics/iter' + str(iter+1).zfill(4) + '.png')
                plt.show()
            
        print(f'Ended at Iter {iter + 1}')

    # Use this X since order is changed for this one
    return X, Ys, Zs
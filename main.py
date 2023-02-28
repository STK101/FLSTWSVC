import argparse
import sklearn
from FLSTWSVC import run_algorithm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='FLSTWSVC Experiments Parameters')
    parser.add_argument('--dataset',type=int,required=True, default=0)
    parser.add_argument('--num_cluster',type=int,required=False, default=3)
    parser.add_argument('--C',type=float,required=False, default=0.001)
    parser.add_argument('--v',type=float,required=False, default=0.001)
    parser.add_argument('--iter',type=int,required=False, default=1000)
    parser.add_argument('--get_line_plots',type=bool,required=False, default=False)
    parser.add_argument('--kernel', type = int, required = False, default = 0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    num_cluster = args.num_cluster
    C = args.C
    v = args.v
    iter = args.iter
    plots = args.get_line_plots
    ker = ""
    if (args.kernel == 0):
        ker = "linear"
    elif (args.kernel == 1):
        ker = "RBF"
    X = sklearn.datasets.make_blobs(50, 2, centers = num_cluster, cluster_std = 1,center_box=(-6.0, 6.0))[0]
    X, Ys, Zs = run_algorithm(X, num_cluster, C,v,iter,init_lines_plot = plots,kernel=ker)
    np.savetxt("cluster_assignments.csv", Ys[-1], delimiter=",")
    np.savetxt("Plane_weights.csv", Zs[-1], delimiter=",")
    print("Algorithm Succeeded")
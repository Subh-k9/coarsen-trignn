import torch
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_dense_adj, get_laplacian



def dirichlet_energy(P,original_edge_index,edge_index_corsen,edge_features_coarsen,original_X,coarsened_X):
    Lap =  get_laplacian(original_edge_index)
    Lap2 = get_laplacian(edge_index = edge_index_corsen, edge_weight= edge_features_coarsen)
    
    L_dense = to_dense_adj(edge_index= Lap[0], edge_attr= Lap[1])
    L_dense =torch.squeeze(L_dense)
    L_coarse = to_dense_adj(edge_index= Lap2[0], edge_attr= Lap2[1])
    L_coarse =torch.squeeze(L_coarse)
    
    k_max = 30
    L_org = csr_matrix(np.array(L_dense))
    l, U = sp.sparse.linalg.eigsh(L_org, k=k_max, which="LM", tol=1e-3)
    L_c = csr_matrix(np.array(L_coarse))
    lc, Uc = sp.sparse.linalg.eigsh(L_c, k=k_max, which="LM", tol=1e-3)
    
    
    #print(np.trace(coarsened_X.T@L_c@coarsened_X))
    #print(np.trace(original_X.T@L_org@original_X))
    
    #error=np.linalg.norm(original_X-(P@coarsened_X))
    error = np.abs(np.trace(coarsened_X.T@L_c@coarsened_X) - np.trace(original_X.T@L_org@original_X))
    return error



def reconstruction_error(num_nodes,P,original_edge_index,edge_index_corsen,edge_features_coarsen):
  Lap =  get_laplacian(original_edge_index)
  L_dense = to_dense_adj(edge_index= Lap[0], edge_attr= Lap[1])
  L_dense =torch.squeeze(L_dense)
  L = csr_matrix(np.array(L_dense))

  Lap2 = get_laplacian(edge_index = edge_index_corsen, edge_weight= edge_features_coarsen)
  L_coarse = to_dense_adj(edge_index= Lap2[0], edge_attr= Lap2[1])
  L_coarse =torch.squeeze(L_coarse)
  L_c = csr_matrix(np.array(L_coarse))

  L_lift=P.T@L_c@P
  LL=(L-L_lift)
  return np.log(pow(np.linalg.norm(LL),2)/num_nodes)



def hyperbolic_error(P,original_edge_index,edge_index_corsen,edge_features_coarsen,X):
  Lap =  get_laplacian(original_edge_index)
  L_dense = to_dense_adj(edge_index= Lap[0], edge_attr= Lap[1])
  L_dense =torch.squeeze(L_dense)
  L = csr_matrix(np.array(L_dense))

  Lap2 = get_laplacian(edge_index = edge_index_corsen, edge_weight= edge_features_coarsen)
  L_coarse = to_dense_adj(edge_index= Lap2[0], edge_attr= Lap2[1])
  L_coarse =torch.squeeze(L_coarse)
  L_c = csr_matrix(np.array(L_coarse))

  # Lifted Laplacian
  L_lift=P.T@L_c@P
  return np.arccosh(1+((pow(np.linalg.norm((L_lift-L)@X),2)*pow(np.linalg.norm(X),2))/(2*np.trace(X.T@L_lift@X)*np.trace(X.T@L@X))))




def eigen_error(original_edge_index, edge_index_corsen, edge_features_coarsen, k_max):
    Lap =  get_laplacian(original_edge_index)
    Lap2 = get_laplacian(edge_index = edge_index_corsen, edge_weight= edge_features_coarsen)
    
    L_dense = to_dense_adj(edge_index= Lap[0], edge_attr= Lap[1])
    L_dense =torch.squeeze(L_dense)
    L_coarse = to_dense_adj(edge_index= Lap2[0], edge_attr= Lap2[1])
    L_coarse =torch.squeeze(L_coarse)
    
    L_org = csr_matrix(np.array(L_dense))
    print(L_org.shape)
    print(k_max)
    l, U = sp.sparse.linalg.eigsh(L_org, k=k_max, which="LM", tol=1e-3)
    L_c = csr_matrix(np.array(L_coarse))
    lc, Uc = sp.sparse.linalg.eigsh(L_c, k=k_max, which="LM", tol=1e-3)
    
    errors = np.abs(l[:k_max] - lc[:k_max]) / l[:k_max]
    #print("original graph eigen values sum",l[:k_max])
    #print("coarsened graph eifen values sum",lc[:k_max])
    print("mean eigen error", np.mean(errors))
    return errors



def plot_most_significant_eigen_values(n,original_edge_index,edge_index_corsen,edge_features_coarsen,name):

  Lap =  get_laplacian(original_edge_index)
  L = to_dense_adj(edge_index= Lap[0], edge_attr= Lap[1])
  eigen_values,eigenvectors=np.linalg.eig(L) 
  s=np.sort(eigen_values)
  s_new=s.flatten()[-n:]
  

  Lap2 = get_laplacian(edge_index = edge_index_corsen, edge_weight= edge_features_coarsen)
  L_c = to_dense_adj(edge_index= Lap2[0], edge_attr= Lap2[1])
  eigen_value,eigenvector=np.linalg.eig(L_c)
  z=np.sort(eigen_value) 
  z_new=z.flatten()[-n:]
  temp=0
  # for j in range(len(s_new)):
  #   temp=temp+(abs(z_new[j]-s_new[j])/s_new[j])
  # eigenerror1=temp/len(s_new)
  #print(" eigen_error 1")
  #print(eigenerror1)

  plt.plot(s_new,label="Original")
  plt.plot(z_new,':', label="FACH")
  title = name.split('/')[-1]
  plt.title(title, x=0.5, y=0.9,weight="bold")
  plt.ylabel('Relative eigen-value error')
  plt.xlabel('Nth eigenvalue')
  plt.legend()
  temp = name + "_top_" + (str)(n) + "_eigen_values.png"
  plt.savefig(temp, dpi=1500)
  plt.show()

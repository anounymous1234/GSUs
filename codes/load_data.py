import numpy as np
import random as rd
import os
from scipy import sparse
from utils import *
from params import *


class Data(object):
    def __init__(self, train_directory, val_file, test_file):
        self.n_users, self.n_items = 0, 0
        self.train_file_list = os.listdir(train_directory)
        self.train_file_list.sort(key=lambda x:int(x.strip('.npz').split('_')[-1]))
        for ind in range(len(self.train_file_list)):
            self.train_file_list[ind] = os.path.join(train_directory, self.train_file_list[ind])

        # read one of train mat to get n_users and n_items
        one_ui_mat = sparse.load_npz(self.train_file_list[0])
        self.n_users, self.n_items = one_ui_mat.shape

        # load train items for each user
        self.train_items = {}
        last_mat = sparse.load_npz(self.train_file_list[-1]).toarray()
        for u in range(self.n_users):
            pos_items = np.nonzero(last_mat[u, :])[0]
            pos_items = pos_items.tolist()
            self.train_items[u] = pos_items

        self.validation_set = {}
        iStream_val_file = open(val_file, 'r')
        for eachline in iStream_val_file:
            line_parts = eachline.strip('\n').split(' ')
            uid = int(line_parts[0])
            val_item = int(line_parts[1])
            self.validation_set[uid] = val_item
        iStream_val_file.close()

        self.test_set = {}
        iStream_test_file = open(test_file,'r')
        for eachline in iStream_test_file:
            line_parts = eachline.strip('\n').split(' ')
            uid = int(line_parts[0])
            test_item = int(line_parts[1])
            self.test_set[uid] = test_item
        iStream_test_file.close()

        self.lastN_matrices = self.load_matrices()
        self.eig_values, self.eig_vectors = self.eigs()
        self.conscutive_matrice = [self.lastN_matrices[0]]
        for mat in self.lastN_matrices[1:]:
            temp = self.conscutive_matrice[-1] + mat
            self.conscutive_matrice.append(temp)

        self.last = self.conscutive_matrice[-1].toarray()
        self.last[self.last > 1] = 1
        # self.n_mats, _, _ = self.lastN_matrices.shape
        self.n_mats = len(self.lastN_matrices)
        print('Number of rating matrices:', self.n_mats)
        self.start = 0

    # load the last consecutive n_mat matrices as a 3 dimentional tensor [n_mat, n_users+n_items, n_users+n_items]
    def load_matrices(self):
        last_n_mat_file_list = self.train_file_list
        mats = []
        for eachfile in last_n_mat_file_list:
            one_mat = sparse.load_npz(eachfile)
            mats.append(one_mat)
        return mats

    # sample from loaded matrices
    def data_generator(self):
        time = len(self.lastN_matrices)
        self.start = []
        values, vectors = [], []
        for b in range(BATCH_SIZE):
            s = rd.choice(range(time - SEQ_LEN -2))
            self.start.append(s)
            #consec_n_mat_3darray = self.lastN_matrices[self.start:self.start + SEQ_LEN]#self.conscutive_matrice[self.start:self.start + SEQ_LEN]
                               #self.lastN_matrices[self.start+1:self.start + SEQ_LEN]
            #consec_n_mat_3darray = np.stack([mat.toarray().astype(np.bool) for mat in consec_n_mat_3darray])
            values.append(self.eig_values[s:s+SEQ_LEN])
            vectors.append(self.eig_vectors[s:s+SEQ_LEN])
        return np.asarray(values), np.asarray(vectors)


    def create_labels(self):
        labels = np.zeros((BATCH_SIZE, SAMPLE_SIZE, 1 + 1 + NEGATIVE_SAMPLE))
        for b in range(0, BATCH_SIZE):
            index = self.start[b]

            #start = self.conscutive_matrice[index+SEQ_LEN].toarray()
            #start[start > 1] = 1
            target = self.lastN_matrices[index+SEQ_LEN]#self.last - start
        
            users = np.nonzero(target)[0].tolist()
            items = np.nonzero(target)[1].tolist()

            for s in range(0, SAMPLE_SIZE):
                i = rd.choice(range(len(users)))
                u, pos = users[i], items[i]
                negs = rd.sample(list(set(set(range(self.n_items)) -
                        set(np.nonzero(self.last[u, :])[0]))), NEGATIVE_SAMPLE)
                labels[b, s, 0], labels[b, s, 1], labels[b, s, 2:] = u, pos, negs
        return labels

    def eigs(self):
        eig_values = []
        eig_vectors = []
        from scipy.sparse import linalg
        i = 0
        for R in self.lastN_matrices:
            adj = self.adjacient_matrix(R=R, self_connection=True)
            D = self.degree_matrix(A=adj)
            L = self.laplacian_matrix(D=D, A=adj)
            eigenvalues, eigenvectors = linalg.eigs(L, k=l)
            eig_values.append(eigenvalues)
            eig_vectors.append(eigenvectors)
            i += 1
        eig_values = np.stack(eig_values)
        eig_vectors = np.stack(eig_vectors)
        # np.save('eigen_values', eig_values)
        # np.save('eigen_vectors', eig_vectors)
        # eig_values = np.load('eigen_values.npy')
        # eig_values = [values.real for values in eig_values]
        # eig_vectors = np.load('eigen_vectors.npy')
        # eig_vectors = [vectors.real for vectors in eig_vectors]
        return eig_values, eig_vectors

    def adjacient_matrix(self, R, self_connection=False):
        A = np.zeros([self.n_users + self.n_items, self.n_users + self.n_items], dtype=np.float32)
        A[:self.n_users, self.n_users:] = R.toarray()
        A[self.n_users:, :self.n_users] = R.toarray().T
        if self_connection == True:
            return np.identity(self.n_users + self.n_items, dtype=np.float32) + A
        return A

    def degree_matrix(self, A):
        degree = np.sum(A, axis=1, keepdims=False)
        # degree = np.diag(degree)
        return degree

    def laplacian_matrix(self, D, A, normalized=False):
        if normalized == False:
            return D - A

        temp = np.dot(np.diag(np.power(D, -1)), A)
        return np.identity(self.n_users + self.n_items, dtype=np.float32) - temp


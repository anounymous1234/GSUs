import tensorflow as tf
from params import *
from utils import *

class GSU(object):
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.N = self.n_users + self.n_items
        self.C = C
        self.F = F

        # placeholder definition
        #self.R_mats = tf.placeholder(tf.bool, shape=(SEQ_LEN,self.n_users, self.n_items))
        self.eig_values = tf.placeholder(DTYPE, shape=(BATCH_SIZE, SEQ_LEN, l))
        self.eig_vectors = tf.placeholder(DTYPE, shape=(BATCH_SIZE, SEQ_LEN, self.N, l))

        self.labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE, SAMPLE_SIZE, 1 + 1 + NEGATIVE_SAMPLE))


        self.features = tf.Variable(
            tf.truncated_normal([self.N, self.F], mean=0.01, stddev=0.02, dtype=DTYPE))
        
        self.Wz = tf.Variable(
            tf.truncated_normal([self.F, self.F], mean=0.01, stddev=0.02, dtype=DTYPE))


        self.Wr = tf.Variable(
            tf.truncated_normal([self.F, self.F], mean=0.01, stddev=0.02, dtype=DTYPE))


        self.Wx = tf.Variable(
            tf.truncated_normal([self.F, self.F], mean=0.01, stddev=0.02, dtype=DTYPE))


        self.bias_Z = tf.Variable(tf.zeros([self.N, 1], dtype=DTYPE))
        self.bias_R = tf.Variable(tf.zeros([self.N, 1], dtype=DTYPE))
        self.bias_H = tf.Variable(tf.zeros([self.N, 1], dtype=DTYPE))

        self.total_loss = tf.zeros(1)

        self.opt = tf.train.AdamOptimizer(learning_rate=lr)

        self.forward()

        u_embeddings, i_embeddings = tf.split(self.final_H, [self.n_users, self.n_items])

        self.all_ratings = tf.matmul(u_embeddings, i_embeddings, transpose_a=False, transpose_b=True)
        
        self.updates = self.opt.minimize(self.total_loss)
        

    def cell(self, t, H_t):
        # transformation formula for gated GCN
        # OUTPUT: H_{t+1}
        # L = tf.subtract(tf.eye(self.N, dtype=DTYPE), tf.matmul(D, A))
        #coeff = L#tf.add(L, tf.matmul(L, L))
        def conv(signal, t, theta):
            values = tf.gather(self.eig_value, t)
            values = tf.add(tf.eye(l), tf.diag(values))

            vectors = tf.gather(self.eig_vector, t)
            coeff = tf.matmul(tf.matmul(vectors, values), tf.transpose(vectors))
            return tf.matmul(tf.matmul(coeff,signal),theta)


        # signal = tf.concat([self.user_signal, self.item_signal],axis=0)

        Z_t = tf.nn.sigmoid(tf.add(conv(H_t, t, self.Wz), self.bias_Z)) # tf.nn.sigmoid(tf.add(conv(H_t, coeff, self.Whz), self.bias_Z))#
        
        R_t = tf.nn.sigmoid(tf.add(conv(H_t, t, self.Wr), self.bias_R))
        H_t_hat = tf.nn.tanh(tf.add(conv(tf.multiply(R_t, H_t), t, self.Wx), self.bias_H))

        H_t1 = tf.add(tf.multiply(Z_t, H_t),
                      tf.multiply(tf.subtract(tf.ones((self.N, self.F), dtype=DTYPE), Z_t), H_t_hat))

        return H_t1
        

    def forward(self):
        losses = []
        for b in range(0, BATCH_SIZE):
            self.eig_value = tf.gather(self.eig_values, b)
            self.eig_vector = tf.gather(self.eig_vectors, b)
            H_t = self.features
            H = []
            for t in range(0, SEQ_LEN):
                H_t = self.cell(t=t, H_t=H_t)
                H.append(H_t)
       
            self.final_H = H[-1]#tf.concat(H, axis=1)
            labels = tf.gather(self.labels, b)
            loss = self.create_dynamic_loss(h=self.final_H, labels=labels) + DECAY * tf.nn.l2_loss(self.final_H)
            losses.append(loss)

        self.total_loss = tf.reduce_sum(losses)
        #self.total_loss = tf.add(self.total_loss, tf.multiply(tf.constant(DECAY), self.regularization()))


    def create_dynamic_loss(self, h, labels):
        user_embeddings, item_embeddings = tf.split(h, [self.n_users, self.n_items])

        losses = []
        for u in range(SAMPLE_SIZE):
            l = tf.gather(labels, u)
            user, pos_item, neg_item = l[0], l[1], l[2]
            u_embedding = tf.gather(user_embeddings, user)
            pos_i_embedding = tf.gather(item_embeddings, pos_item)
            pos_scores = tf.reduce_sum(tf.multiply(u_embedding, pos_i_embedding))

            neg_i_embedding = tf.gather(item_embeddings, neg_item)
            neg_scores = tf.reduce_sum(tf.multiply(u_embedding, neg_i_embedding))

            maxi = tf.log_sigmoid(tf.subtract(pos_scores, tf.reduce_sum(neg_scores)))
            losses.append(maxi)
        return tf.negative(tf.reduce_sum(losses))

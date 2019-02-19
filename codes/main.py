from GSU import *
from test import *
from validation import *

def main():
    model = GSU(n_users=data.n_users, n_items=data.n_items)

    config = tf.ConfigProto()
    #config = tf.ConfigProto(
    #    device_count={'GPU': 0}
    #)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print('Initializing variables...')
    sess.run(tf.global_variables_initializer())

    for epoch in range(N_EPOCHS):
        eig_values, eig_vectors = data.data_generator()
        labels = data.create_labels()

        _, loss = sess.run([model.updates, model.total_loss],
                           feed_dict={model.eig_values: eig_values,
                                      model.eig_vectors: eig_vectors,
                                      model.labels: labels})
        users_to_val = list(data.validation_set.keys())

        ret = batch_validation(sess, model, users_to_val)

        print('Epoch %d training loss %f' % (epoch, loss))
        print('Validation: ')
        print('hr 10 %f hr 20 %f hr 30 %f hr 40 %f hr 50 %f' % (ret[0], ret[1], ret[2], ret[3], ret[4]))
        print('ndcg 10 %f ndcg 20 %f ndcg 30 %f ndcg 40 %f ndcg 50 %f' % (ret[5], ret[6], ret[7], ret[8], ret[9]))

        users_to_test = list(data.test_set.keys())

        ret = batch_test(sess, model, users_to_test)

        print('Test: ')
        print('hr 10 %f hr 20 %f hr 30 %f hr 40 %f hr 50 %f' % (ret[0], ret[1], ret[2], ret[3], ret[4]))
        print('ndcg 10 %f ndcg 20 %f ndcg 30 %f ndcg 40 %f ndcg 50 %f' % (ret[5], ret[6], ret[7], ret[8], ret[9]))
        #import time
        #time.sleep(60)

if __name__ == '__main__':
    main()

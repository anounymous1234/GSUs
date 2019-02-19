from test import *

def val_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    training_items = data.train_items[u]
    #user u's items in the test set
    user_pos_val = data.validation_set[u]

    all_items = set(range(data.n_items))

    val_items = rd.sample(list((all_items - set(training_items)) - set([user_pos_val])), NUM_NEGATIVE_TEST)
    #val_items = (all_items - set(training_items)) - set([user_pos_val])
    val_items = list(set(val_items) | set([user_pos_val]))
    item_score = []
    for i in val_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i == user_pos_val:
            r.append(1)
        else:
            r.append(0)

    ht_10 = hit_ratio(r, 10)
    ht_20 = hit_ratio(r, 20)
    ht_30 = hit_ratio(r, 30)
    ht_40 = hit_ratio(r, 40)
    ht_50 = hit_ratio(r, 50)
    ndcg_10 = ndcg_at_k(r, 10)
    ndcg_20 = ndcg_at_k(r, 20)
    ndcg_30 = ndcg_at_k(r, 30)
    ndcg_40 = ndcg_at_k(r, 40)
    ndcg_50 = ndcg_at_k(r, 50)
    return np.array([ht_10,ht_20,ht_30,ht_40,ht_50,ndcg_10,ndcg_20,ndcg_30,ndcg_40,ndcg_50])


def batch_validation(sess, model, users_to_val):
    result = np.array([0.] * 10)
    pool = multiprocessing.Pool(cores)

    #all users needed to test
    val_users = users_to_val
    val_user_num = len(val_users)

    values_for_test = data.eig_values[-SEQ_LEN:]
    values_for_test = np.asarray([values_for_test] * BATCH_SIZE)
    vectors_for_test = data.eig_vectors[-SEQ_LEN:]
    vectors_for_test = np.asarray([vectors_for_test] * BATCH_SIZE)
    user_ratings = sess.run(model.all_ratings, feed_dict={model.eig_values: values_for_test,
                                                          model.eig_vectors: vectors_for_test})

    user_batch = val_users

    user_batch_rating_uid = zip(np.take(user_ratings, user_batch, axis=0).tolist(), user_batch)
    batch_result = pool.map(val_one_user, user_batch_rating_uid)


    for re in batch_result:
        result += re

    pool.close()
    ret = result / val_user_num
    ret = list(ret)
    return ret

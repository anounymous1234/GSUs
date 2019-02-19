from datetime import datetime
from datetime import timedelta
import copy
from scipy import sparse
import numpy as np
import os


user_id_list_file = 'user_id_list.txt'
user_id_list_lesspos_file = 'user_id_lesspos_list.txt'
item_id_list_file = 'item_id_list.txt'
date_num_inter_stat_file = 'date_num_inter_stat_file.txt'
train_adjacency_folder_str = 'train_adjacency_matrices_perday_ratiosplit'
validation_set_file = 'validation_set.txt'
test_set_file = 'test_set.txt'
dates_with_adj_index_file = 'dates_with_adj_index.txt'

binary_setting = True
is_accumulated = False

#if rating is less than or equal to rating threshold, it will be 0, else, 1
rating_threshold = 5
num_days_for_mat = 20000000000

#these three numbers accepted from running parameters
#define how many days per adjmatrix has, for example, we have 1040 days,
#then we have 1040/7 = 148 with reminder of 4, we should have 149 matrices.
#skip those matrices which have no difference between itself and its previous one.
days_per_mat = 1
num_items_validation = 1
num_items_test = 1
is_ratio_split = False
num_items_ratio_valid_test = 0.4

def load_movielensonemillion(rating_file_path):
    iStream_ratings = open(rating_file_path,'r')

    #key is user/item id, value is index
    user_id_positive_numratings = {}
    uniq_users_indataset = set()
    item_id_list = {}
    first_interact_date = datetime.max
    last_interact_date = datetime.min
    num_interact_eachdate = {}
    num_positive_eachdate = {}
    
    #key is user_id, value is another {} whose key is (item_id, rating) and value is datetime_obj
    all_ratings = {}
    item_id_ind = 0
    i = 0
    num_positive_ratings = 0
    for eachline in iStream_ratings:
        i += 1
        if i % 20000 == 0:
            print(i)
        allparts = eachline.strip().split('::')
        user_id = int(allparts[0])
        item_id = int(allparts[1])
        rating = float(allparts[2])
        if binary_setting:
            if rating < rating_threshold:
                rating = 0
            else:
                rating = 1
                num_positive_ratings += 1
        timestamp = int(allparts[3])
        one_date = datetime.fromtimestamp(timestamp)

        uniq_users_indataset.add(user_id)

        date_str = one_date.strftime('%Y-%m-%d')
        if date_str not in num_interact_eachdate:
            num_interact_eachdate[date_str] = 1
        else:
            num_interact_eachdate[date_str] += 1

        if item_id not in item_id_list:
            item_id_list[item_id] = item_id_ind
            item_id_ind += 1

        if one_date < first_interact_date:
            first_interact_date = one_date
        if one_date > last_interact_date:
            last_interact_date = one_date

        if rating == 1:
            if user_id not in user_id_positive_numratings:
                user_id_positive_numratings[user_id] = 1
            else:
                user_id_positive_numratings[user_id] += 1

            if date_str not in num_positive_eachdate:
                num_positive_eachdate[date_str] = 1
            else:
                num_positive_eachdate[date_str] += 1

            if user_id not in all_ratings:
                all_ratings[user_id] = {(item_id,rating):one_date}
            else:
                all_ratings[user_id][(item_id,rating)] = one_date


    iStream_ratings.close()
    user_id_list = {}
    user_id_list_lesspos = {}
    sorted_user_ids = sorted(user_id_positive_numratings.items(), key=lambda x: x[0])
    ind_user = 0
    for user_id, num_pos in sorted_user_ids:
        if user_id_positive_numratings[user_id] >=3:
            user_id_list[user_id] = ind_user
            ind_user += 1

    for user_id, num_pos in sorted_user_ids:
        if user_id_positive_numratings[user_id] > 0 and user_id_positive_numratings[user_id] < 3:
            user_id_list_lesspos[user_id] = ind_user
            ind_user += 1

    print('first_interact_date: ' + str(first_interact_date))
    print('last_interact_date: ' + str(last_interact_date))
    print('num_uniq_users in dataset: ' + str(len(uniq_users_indataset)))
    print('num_uniq_users with >= 3 positive ratings: ' + str(len(user_id_list)))
    print('num_uniq_users with >0 and <3 positive ratings: ' + str(len(user_id_list_lesspos)))
    print('num_uniq_items: ' + str(item_id_ind))
    print('num_positive_ratings:' + str(num_positive_ratings))

    oStream_user_id_list = open(user_id_list_file,'w')
    for user_id, u_index in user_id_list.items():
        oStream_user_id_list.write(str(user_id) + ' ' + str(u_index) + '\n')
    oStream_user_id_list.close()

    oStream_user_id_list_lesspos = open(user_id_list_lesspos_file,'w')
    for user_id, u_index in user_id_list_lesspos.items():
        oStream_user_id_list_lesspos.write(str(user_id) + ' ' + str(u_index) + '\n')
    oStream_user_id_list_lesspos.close()

    oStream_item_id_list = open(item_id_list_file,'w')
    for item_id, i_index in item_id_list.items():
        #because we are going to create adjancy matrix with user first and then item, so indices of items should be after the ones of users.
        #item_id_list[item_id] += len(user_id_list)
        oStream_item_id_list.write(str(item_id) + ' ' + str(item_id_list[item_id]) + '\n')
    oStream_item_id_list.close()

    oStream_date_stat = open(date_num_inter_stat_file, 'w')
    incre_date = timedelta(days=1)
    move_date = first_interact_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while(move_date <= last_interact_date):
        move_date_str = move_date.strftime('%Y-%m-%d')
        if move_date_str not in num_interact_eachdate:
            oStream_date_stat.write(move_date_str + ' 0 ')
            if move_date_str not in num_positive_eachdate:
                oStream_date_stat.write('0\n')
            else:
                oStream_date_stat.write(str(num_positive_eachdate[move_date_str]) + '\n')
        else:
            oStream_date_stat.write(move_date_str + ' ' + str(num_interact_eachdate[move_date_str]) + ' ')
            if move_date_str not in num_positive_eachdate:
                oStream_date_stat.write('0\n')
            else:
                oStream_date_stat.write(str(num_positive_eachdate[move_date_str]) + '\n')
        move_date = move_date + incre_date
    oStream_date_stat.close()

    return user_id_list, user_id_list_lesspos, item_id_list, first_interact_date, last_interact_date, num_interact_eachdate, all_ratings



def create_adjacency_mat(user_id_list, user_id_list_lesspos, item_id_list, first_interact_date, last_interact_date, all_ratings):

    first_interact_date = first_interact_date.replace(hour=0, minute=0, second=0, microsecond=0)
    last_interact_date = last_interact_date.replace(hour=0, minute=0, second=0, microsecond=0)
    #dimensions = len(user_id_list) + len(item_id_list)
    #array of scipy sparse matrices (in dok_matrix)
    #recurrent_rating_spmat = sparse.csr_matrix((dimensions, dimensions), dtype=np.float32)
    #recurrent_rating_fullmat = np.zeros((dimensions, dimensions), dtype=np.float32)
    recurrent_rating_fullmat = np.zeros((len(user_id_list)+len(user_id_list_lesspos), len(item_id_list)), dtype=np.float32)
    previous_rating_fullmat = copy.deepcopy(recurrent_rating_fullmat)

    #separate the training, validation and test set
    #here I use datetime as key in training set because it is more convenient to generate adjacency matrices
    training_set = {}
    validation_set = {}
    test_set = {}
    global num_items_test
    global num_items_validation
    for user_id, user_ratings in all_ratings.items():

        if user_id in user_id_list:
            sorted_ratings = sorted(user_ratings.items(), key=lambda x:x[1])
            #sorted_ratings will be (((item_id1, rating1),date_obj1), ((item_id2, rating2), date_obj2),...)
            if is_ratio_split:
                if 1.0 * len(sorted_ratings) * num_items_ratio_valid_test < 2:
                    num_items_validation = 0
                    num_items_test = 1
                else:
                    num_items_valid_test = int(len(sorted_ratings) * num_items_ratio_valid_test)
                    if num_items_valid_test %2 == 0:
                        num_items_validation = int(num_items_valid_test/2)
                        num_items_test = num_items_validation
                    else:
                        num_items_test = int(num_items_valid_test/2)+1
                        num_items_validation = int(num_items_valid_test/2)

            test_set[user_id] = sorted_ratings[(len(sorted_ratings) - num_items_test):len(sorted_ratings)]
            validation_set[user_id] = sorted_ratings[(len(sorted_ratings) - num_items_test - num_items_validation):(len(sorted_ratings) - num_items_test)]

            #get the training portion from user_ratings
            user_training_data = sorted_ratings[:(len(sorted_ratings) - num_items_test - num_items_validation)]

            #print((len(sorted_ratings),num_items_test, num_items_validation, len(sorted_ratings) - num_items_test - num_items_validation, len(test_set[user_id]), len(validation_set[user_id]), len(user_training_data)))
            for eachrating in user_training_data:
                item_id = eachrating[0][0]
                rating = eachrating[0][1]
                happen_date = eachrating[1]
                happen_date_str = happen_date.strftime('%Y-%m-%d')
                if happen_date_str not in training_set:
                    training_set[happen_date_str] = [[user_id, item_id, rating]]
                else:
                    training_set[happen_date_str].append([user_id, item_id, rating])
        elif user_id in user_id_list_lesspos:
            user_training_data = sorted(user_ratings.items(),key=lambda x:x[1])
            for eachrating in user_training_data:
                item_id = eachrating[0][0]
                rating = eachrating[0][1]
                happen_date = eachrating[1]
                happen_date_str = happen_date.strftime('%Y-%m-%d')
                if happen_date_str not in training_set:
                    training_set[happen_date_str] = [[user_id, item_id, rating]]
                else:
                    training_set[happen_date_str].append([user_id, item_id, rating])
    #save validation set to file
    write_ValiOrTest_tofile(validation_set_file, validation_set, user_id_list, item_id_list)
    
    #save test set to file
    write_ValiOrTest_tofile(test_set_file, test_set, user_id_list, item_id_list)

    incre_one_day = timedelta(days=1)
    move_date = first_interact_date
    num_ratings_in_train = 0
    mat_index = 0
    dates_with_adj = []

    if days_per_mat == 1:
        train_adjacency_folder = train_adjacency_folder_str + '/'
    else:
        train_adjacency_folder = train_adjacency_folder_str + '_' + str(days_per_mat) + 'dayspermat/'
    if not os.path.isdir(train_adjacency_folder):
        os.mkdir(train_adjacency_folder)

    while move_date <= last_interact_date:
        move_date_str = move_date.strftime('%Y-%m-%d')

        diff_days_from_last = (last_interact_date - move_date).days
        diff_days_from_first = (move_date - first_interact_date).days + 1

        if not is_accumulated:
            recurrent_rating_fullmat = np.zeros((len(user_id_list) + len(user_id_list_lesspos), len(item_id_list)),
                                            dtype=np.float32)
            previous_rating_fullmat = copy.deepcopy(recurrent_rating_fullmat)
            num_ratings_in_train = 0
        #the current move_date has no positive rating
        if move_date_str not in training_set:
            if diff_days_from_first % days_per_mat == 0 or (move_date == last_interact_date and diff_days_from_first % days_per_mat != 0):
                if np.sum(np.absolute(np.subtract(recurrent_rating_fullmat, previous_rating_fullmat))) > 0:
                    previous_rating_fullmat = copy.deepcopy(recurrent_rating_fullmat)
                    recurrent_rating_spmat = sparse.csr_matrix(recurrent_rating_fullmat)
                    sparse.save_npz(train_adjacency_folder + 'adjacency_mat_cumulative_' + str(mat_index) + '.npz',
                                    recurrent_rating_spmat)
                    dates_with_adj.append(move_date_str)
                    mat_index += 1
            print(move_date_str + ' has no positive rating')
            move_date += incre_one_day
            continue
        else:
            for eachrating in training_set[move_date_str]:
                user_id = eachrating[0]
                user_index = -1
                if user_id in user_id_list:
                    user_index = user_id_list[user_id]
                elif user_id in user_id_list_lesspos:
                    user_index = user_id_list_lesspos[user_id]
                item_id = eachrating[1]
                item_index = item_id_list[item_id]
                rating = eachrating[2]
                if rating != 0:
                    recurrent_rating_fullmat[user_index, item_index] = rating
                    #recurrent_rating_fullmat[item_index, user_index] = rating
                    num_ratings_in_train += 1
        num_u, num_i = recurrent_rating_fullmat.shape
        print(move_date_str + ' ' + str(num_ratings_in_train) + ' ' + str(num_ratings_in_train / (num_u * num_i)))

        if diff_days_from_last < num_days_for_mat:
            if diff_days_from_first % days_per_mat == 0 or (move_date == last_interact_date and diff_days_from_first % days_per_mat != 0):
                if np.sum(np.absolute(np.subtract(recurrent_rating_fullmat, previous_rating_fullmat))) > 0:
                    previous_rating_fullmat = copy.deepcopy(recurrent_rating_fullmat)
                    recurrent_rating_spmat = sparse.csr_matrix(recurrent_rating_fullmat)
                    sparse.save_npz(train_adjacency_folder + 'adjacency_mat_cumulative_' + str(mat_index) + '.npz', recurrent_rating_spmat)
                    dates_with_adj.append(move_date_str)
                    mat_index += 1


        move_date += incre_one_day

    oStream_dates_with_adj = open(dates_with_adj_index_file, 'w')
    for ind, date_str in enumerate(dates_with_adj):
        oStream_dates_with_adj.write(str(ind) + ' ' + date_str + '\n')
    oStream_dates_with_adj.close()





def write_ValiOrTest_tofile(file_path, dataset, user_id_list, item_id_list):
    #save test set to file
    oStream_file = open(file_path, 'w')
    for user_id, one_sorted_ratings in dataset.items():
        if len(one_sorted_ratings) == 0:
            continue
        oStream_file.write(str(user_id_list[user_id]))
        for eachrating in one_sorted_ratings:
            item_id = eachrating[0][0]
            rating = eachrating[0][1]
            happen_date = eachrating[1]
            happen_date_str = happen_date.strftime('%Y-%m-%d')
            #oStream_file.write(' ' + str(item_id_list[item_id]) + ' ' + str(rating) + ' ' + happen_date_str)
            oStream_file.write(' ' + str(item_id_list[item_id]))
        oStream_file.write('\n')
    oStream_file.close()



if __name__ == '__main__':
    ml_rate_file = 'ratings.dat'
    user_id_list, user_id_list_lesspos, item_id_list, first_interact_date, last_interact_date, num_interact_eachdate, all_ratings = load_movielensonemillion(ml_rate_file)
    create_adjacency_mat(user_id_list, user_id_list_lesspos, item_id_list, first_interact_date, last_interact_date, all_ratings)

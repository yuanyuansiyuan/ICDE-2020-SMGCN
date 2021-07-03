from utility.parser import parse_args
from utility.load_data import *
import multiprocessing

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def test(sess, model, users_to_test, test_group_list, drop_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
    test_users = users_to_test
    item_batch = range(ITEM_NUM)
    if drop_flag == False:
        rate_batch = sess.run(model.batch_ratings, {model.users: test_users,
                                                    model.pos_items: item_batch})
    else:
        rate_batch = sess.run(model.batch_ratings, {model.users: test_users,
                                                    model.pos_items: item_batch,
                                                    model.mess_dropout: [0.] * len(eval(args.layer_size))})

    user_batch_rating_uid = zip(test_users, rate_batch)
    user_rating_dict = {}

    index = 0
    for entry in user_batch_rating_uid:
        rating = entry[1]
        temp = [(i, float(rating[i])) for i in range(len(rating))]
        user_rating_dict[index] = temp
        index += 1
    print("#user_rating_dict\t", len(user_rating_dict))

    precision_n = np.zeros(len(Ks))
    recall_n = np.zeros(len(Ks))
    ndcg_n = np.zeros(len(Ks))
    topN = Ks

    gt_count = 0
    candidate_count = 0
    for index in range(len(test_group_list)):
        entry = test_group_list[index]

        v = entry[1]
        rating = user_rating_dict[index]

        candidate_count += len(rating)
        rating.sort(key=lambda x: x[1], reverse=True)
        gt_count += len(v)
        K_max = topN[len(topN) - 1]

        r = []
        for i in rating[:K_max]:
            herb = i[0]
            if herb in v:
                r.append(1)
            else:
                r.append(0)

        for ii in range(len(topN)):
            number = 0
            for i in rating[:topN[ii]]:
                herb = i[0]
                if herb in v:
                    number += 1
            precision_n[ii] = precision_n[ii] + float(number / topN[ii])
            recall_n[ii] = recall_n[ii] + float(number / len(v))
            ndcg_n[ii] = ndcg_n[ii] + ndcg_at_k(r, topN[ii])

    print('gt_count ', gt_count)
    print('candidate_count ', candidate_count)
    print('ideal candidate count ', len(test_group_list) * ITEM_NUM)
    for ii in range(len(topN)):
        result['precision'][ii] = precision_n[ii] / len(test_group_list)
        result['recall'][ii] = recall_n[ii] / len(test_group_list)
        result['ndcg'][ii] = ndcg_n[ii] / len(test_group_list)

    return result


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
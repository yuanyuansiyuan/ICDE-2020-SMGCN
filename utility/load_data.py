import numpy as np
import random as rd
import scipy.sparse as sp
from time import time


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        sym_pair_file = path + '/symPair-5.txt'
        herb_pair_file = path + '/herbPair-40.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0

        # herbs in train
        self.train_items = set()
        # herbs in test
        self.test_items = set()
        self.all_items = set()
        # prescriptions in train
        self.train_pres = list()

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    temp = l.strip().split('\t')
                    tempS = temp[0].split(" ")
                    tempH = temp[1].split(" ")
                    try:
                        uids = [int(i) for i in tempS]
                        items = [int(i) for i in tempH]
                        self.train_pres.append([uids, items])
                        for item in items:
                            self.train_items.add(item)
                            self.all_items.add(item)
                    except Exception:
                        continue
                    self.n_train += 1
                    self.n_users = max(self.n_users, max(uids))
                    self.n_items = max(self.n_items, max(items))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    temp = l.strip().split('\t')
                    tempS = temp[0].split(" ")
                    tempH = temp[1].split(" ")
                    uids = [int(i) for i in tempS]
                    try:
                        items = [int(i) for i in tempH]
                        for item in items:
                            self.test_items.add(item)
                            self.all_items.add(item)
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, max(uids))
                    self.n_test += 1

        self.n_items += 1
        self.n_users += 1
        self.print_statistics()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.test_group_set = list()
        self.test_users = np.zeros((self.n_test, self.n_users), dtype=float)
        self.item_weights = np.zeros((self.n_items, 1), dtype=float)
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    temp = l.strip().split('\t')
                    tempS = temp[0].split(' ')
                    tempH = temp[1].split(' ')
                    uids = [int(i) for i in tempS]
                    items = [int(i) for i in tempH]
                    for item in items:
                        self.item_weights[item][0] += 1
                    for user in uids:
                        for item in items:
                            self.R[user, item] = 1.

                print('item_weight ', len(self.item_weights))
                item_freq_max = self.item_weights.max()
                print('item_freq_max ', item_freq_max)
                print(self.item_weights.shape[0], ' ', self.item_weights.shape[1])
                for index in range(self.item_weights.shape[0]):
                    self.item_weights[index][0] = item_freq_max * 1.0 / self.item_weights[index][0]

                test_index = 0
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    temp = l.strip().split('\t')
                    tempS = temp[0].split(' ')
                    tempH = temp[1].split(' ')
                    try:
                        uids = [int(i) for i in tempS]
                        for uid in uids:
                            self.test_users[test_index][uid] = 1.
                        items = [int(i) for i in tempH]
                    except Exception:
                        continue
                    test_index += 1

                    uid, test_items = uids, items
                    user_index = ''
                    for user in uid:
                        user_index += str(user) + "_"
                    user_index = user_index[:-1]
                    self.test_group_set.append([user_index, test_items])

                print("#multi-hot for test users\t", len(self.test_users))
                print("#test\t", len(self.test_group_set))

        self.sym_pair = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
        self.herb_pair = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        sym_pair_count = 0
        with open(sym_pair_file) as f_sym_pair:
            for l in f_sym_pair.readlines():
                if len(l) == 0: break
                pair = l.strip().split(' ')
                sym1 = int(pair[0])
                sym2 = int(pair[1])
                # print('sym-pair ', sym1, ' ', sym2)
                self.sym_pair[sym1, sym2] = 1.
                self.sym_pair[sym2, sym1] = 1.
                sym_pair_count += 2

        print('#双向 sym pairs ', sym_pair_count)

        herb_pair_count = 0
        with open(herb_pair_file) as f_herb_pair:
            for l in f_herb_pair.readlines():
                if len(l) == 0: break
                pair = l.strip().split(' ')
                herb1 = int(pair[0])
                herb2 = int(pair[1])
                # print('herb ', herb1, ' ', herb2)
                self.herb_pair[herb1, herb2] = 1.
                self.herb_pair[herb2, herb1] = 1.
                herb_pair_count += 2

        print('#双向herb pairs ', herb_pair_count)

    def get_adj_mat(self, gcnversion):
        adj_mat, norm_adj_mat, mean_adj_mat, sym_pair_mat, herb_pair_mat = self.create_adj_mat(gcnversion)
        return adj_mat, norm_adj_mat, mean_adj_mat, sym_pair_mat, herb_pair_mat

    def create_adj_mat(self, gcnversion):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        # 双向的原始邻接矩阵
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, 'time:', time() - t1)
        t2 = time()

        sym_pair_adj_mat = self.sym_pair.tolil().todok()
        print('already create sym_pair adjacency matrix', sym_pair_adj_mat.shape, 'time:', time() - t2)
        t3 = time()

        herb_pair_adj_mat = self.herb_pair.tolil().todok()
        print('already create herb_pair adjacency matrix', herb_pair_adj_mat.shape, 'time:', time() - t3)

        # todo notice已经为user和item分别进行了行归一化，也就是已经取了平均
        def normalized_adj_single(adj):
            # 行归一化  每行的行sum列表
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            # 每一个元素都除上该行的sum
            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_bi(adj):
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_adj

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        print('already normalize adjacency matrix', time() - t2)

        print('sym_pair 和 herb_pair 有 self-connection, sum!!')
        sym_pair_adj_mat = sym_pair_adj_mat + sp.eye(sym_pair_adj_mat.shape[0])
        herb_pair_adj_mat = herb_pair_adj_mat + sp.eye(herb_pair_adj_mat.shape[0])

        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(), sym_pair_adj_mat.tocsr(), herb_pair_adj_mat.tocsr()

    def sample(self):
        sample_ids = [i for i in range(len(self.train_pres))]
        if self.batch_size <= len(sample_ids):
            pres_ids = rd.sample(sample_ids, self.batch_size)
        else:
            pres_ids = [rd.choice(sample_ids) for _ in range(self.batch_size)]

        users = []
        for pres_id in pres_ids:
            users.append(self.train_pres[pres_id])

        user_sets = np.zeros((len(users), self.n_users), dtype=float)
        item_sets = np.zeros((len(users), self.n_items), dtype=float)
        user_set = set()
        item_set = set()
        for index in range(len(users)):
            uids = users[index][0]
            items = users[index][1]
            for uid in uids:
                user_sets[index][int(uid)] = 1.
                user_set.add(int(uid))
            for item in items:
                item_sets[index][int(item)] = 1.
                item_set.add(int(item))

        return user_sets, list(user_set), item_sets, list(item_set)

    def print_statistics(self):
        print('symtom个数n_users=%d, herb个数n_items=%d' % (self.n_users, self.n_items))
        print('#train中的herb个数 train_items %d' % (len(self.train_items)))
        print('#test中的herb个数 test_items %d' % (len(self.test_items)))
        print('#总的herb个数 all_items %d' % (len(self.all_items)))
        print('#用于sample生成batch的train pres %d' % (len(self.train_pres)))
        print('#train中 症状组合-单个herb pair的个数 n_train %d, #test 行数 n_test %d' % (self.n_train, self.n_test))



import tensorflow as tf


import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utility.helper import *
from utility.batch_test import *
import datetime


import numpy as np


class SMGCN(object):
    def __init__(self, data_config, pretrain_data):
        self.model_type = 'SMGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.sym_pair_adj = data_config['sym_pair_adj']
        self.herb_pair_adj = data_config['herb_pair_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)


        self.fusion = args.fusion
        print('***********fusion method************ ', self.fusion)

        self.mlp_predict_weight_size = eval(args.mlp_layer_size)
        self.mlp_predict_n_layers = len(self.mlp_predict_weight_size)
        print('mlp predict weight ', self.mlp_predict_weight_size)
        print(' mlp_predict layer ', self.mlp_predict_n_layers)
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        print('regs ', self.regs )
        self.decay = self.regs[0]
        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.float32, shape=(None, self.n_users))
        self.user_set = tf.placeholder(tf.int32, shape=(None,))

        self.items = tf.placeholder(tf.float32, shape=(None, self.n_items))
        self.item_set = tf.placeholder(tf.int32, shape=(None,))

        self.pos_items = tf.placeholder(tf.int32, shape=(None,))

        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        self.item_weights = tf.placeholder(tf.float32, shape=(self.n_items, 1))

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights)
        """

        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['SMGCN']:
            self.ua_embeddings  = self._create_graphsage_user_embed()
            self.ia_embeddings  = self._create_graphsage_item_embed()


        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """

        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)


        """
        *********************************************************
        Inference for the testing phase. 

        """
        self.batch_ratings = self.create_batch_rating(self.users, self.pos_i_g_embeddings)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss  = self.create_set2set_loss(self.users, self.items, self.user_set)

        self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size
        pair_dimension = self.weight_size_list[len(self.weight_size_list) - 1]


        for k in range(self.n_layers):
            all_weights['W_gc_user_%d' % k] = tf.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_user_%d' % k)
            all_weights['b_gc_user_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_user_%d' % k)

            all_weights['W_gc_item_%d' % k] = tf.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_item_%d' % k)
            all_weights['b_gc_item_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_item_%d' % k)

            all_weights['Q_user_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k],  self.weight_size_list[k]]), name='Q_user_%d' % k)
            all_weights['Q_item_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k],  self.weight_size_list[k]]), name='Q_item_%d' % k)


        self.mlp_predict_weight_size_list = [self.mlp_predict_weight_size[len(self.mlp_predict_weight_size) - 1]] + self.mlp_predict_weight_size
        print('mlp_predict_weight_size_list ',self.mlp_predict_weight_size_list)
        for k in range(self.mlp_predict_n_layers):
            all_weights['W_predict_mlp_user_%d' % k] = tf.Variable(
                initializer([self.mlp_predict_weight_size_list[k], self.mlp_predict_weight_size_list[k + 1]]), name='W_predict_mlp_user_%d' % k)
            all_weights['b_predict_mlp_user_%d' % k] = tf.Variable(
                initializer([1, self.mlp_predict_weight_size_list[k + 1]]), name='b_predict_mlp_user_%d' % k)


        all_weights['M_user'] = tf.Variable(
            initializer([self.emb_dim, pair_dimension]), name='M_user')
        all_weights['M_item'] = tf.Variable(
            initializer([self.emb_dim, pair_dimension]), name='M_item')

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat


    def _create_graphsage_user_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        pre_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], pre_embeddings))
            embeddings = tf.concat(temp_embed, 0)

            embeddings = tf.nn.tanh(tf.matmul(embeddings, self.weights['Q_user_%d' % k]))
            embeddings = tf.concat([pre_embeddings, embeddings], 1)

            pre_embeddings = tf.nn.tanh(
                tf.matmul(embeddings, self.weights['W_gc_user_%d' % k]) + self.weights['b_gc_user_%d' % k])
            pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[k])

            norm_embeddings = tf.nn.l2_normalize(pre_embeddings, axis=1)
            all_embeddings = [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        temp = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.sym_pair_adj), self.weights['user_embedding'])
        user_pair_embeddings = tf.nn.tanh(tf.matmul(temp, self.weights['M_user']))

        if self.fusion in ['add']:
            u_g_embeddings = u_g_embeddings + user_pair_embeddings
        if self.fusion in ['concat']:
            u_g_embeddings =  tf.concat([u_g_embeddings, user_pair_embeddings], 1)
        return u_g_embeddings

    def _create_graphsage_item_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        pre_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], pre_embeddings))
            embeddings = tf.concat(temp_embed, 0)


            embeddings = tf.nn.tanh(tf.matmul(embeddings, self.weights['Q_item_%d' % k]))
            embeddings = tf.concat([pre_embeddings, embeddings], 1)


            pre_embeddings = tf.nn.tanh(
                tf.matmul(embeddings, self.weights['W_gc_item_%d' % k]) + self.weights['b_gc_item_%d' % k])

            pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[k])

            norm_embeddings = tf.nn.l2_normalize(pre_embeddings, axis=1)
            all_embeddings = [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)


        temp = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.herb_pair_adj),
                                             self.weights['item_embedding'])
        item_pair_embeddings = tf.nn.tanh(tf.matmul(temp, self.weights['M_item']))


        if self.fusion in ['add']:
            i_g_embeddings = i_g_embeddings + item_pair_embeddings

        if self.fusion in ['concat']:
            i_g_embeddings =  tf.concat([i_g_embeddings, item_pair_embeddings], 1)

        return i_g_embeddings


    def create_set2set_loss(self, users, items, user_set):
        sum_embeddings = tf.matmul(users, self.ua_embeddings)

        normal_matrix = tf.reciprocal(tf.reduce_sum(users, 1))

        normal_matrix = tf.expand_dims(normal_matrix, 1)

        extend_normal_embeddings = tf.tile(normal_matrix, (1, sum_embeddings.shape[1]))

        user_embeddings = tf.multiply(sum_embeddings, extend_normal_embeddings)
        all_user_embeddins = tf.nn.embedding_lookup(self.ua_embeddings, user_set)

        for k in range(0, self.mlp_predict_n_layers):
            user_embeddings = tf.nn.relu(
                tf.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k]) + self.weights['b_predict_mlp_user_%d' % k])

            user_embeddings = tf.nn.dropout(user_embeddings, 1 - self.mess_dropout[k])

        predict_probs = tf.nn.sigmoid(tf.matmul(user_embeddings, self.ia_embeddings, transpose_a=False,
                                  transpose_b=True))

        mf_loss = tf.reduce_sum(tf.matmul(tf.square((items - predict_probs), name=None), self.item_weights), 0)
        mf_loss = mf_loss / self.batch_size

        all_item_embeddins = self.ia_embeddings
        regularizer = tf.nn.l2_loss(all_user_embeddins) + tf.nn.l2_loss(all_item_embeddins)
        regularizer = regularizer / self.batch_size

        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32, [1])
        return mf_loss, emb_loss, reg_loss


    def create_batch_rating(self, users, pos_items):

        sum_embeddings = tf.matmul(users, self.ua_embeddings)

        normal_matrix = tf.reciprocal(tf.reduce_sum(users, 1))

        normal_matrix = tf.expand_dims(normal_matrix, 1)

        extend_normal_embeddings = tf.tile(normal_matrix, (1, sum_embeddings.shape[1]))

        user_embeddings = tf.multiply(sum_embeddings, extend_normal_embeddings)


        for k in range(0, self.mlp_predict_n_layers):

            user_embeddings = tf.nn.relu(tf.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k]) + self.weights['b_predict_mlp_user_%d' % k])

            user_embeddings = tf.nn.dropout(user_embeddings, 1 - self.mess_dropout[k])

        pos_scores = tf.nn.sigmoid(tf.matmul(user_embeddings, pos_items, transpose_a=False,
                                       transpose_b=True))
        return pos_scores

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    startTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start ', startTime)
    print('************SMGCN*************** ')
    print('result_index ', args.result_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, sym_pair_adj, herb_pair_adj = data_generator.get_adj_mat()


    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj


    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj


    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = SMGCN(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        mess_dr = '-'.join([str(l) for l in eval(args.mess_dropout)])
        weights_save_path = '%sweights-SMGCN/%s/%s/%s/%s/l%s_r%s_messdr%s' % (
        args.weights_path, args.dataset, model.model_type, layer, args.embed_size,
        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), mess_dr)
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    print("args.pretrain\t", args.pretrain)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining.')


    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger = [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss  = 0., 0., 0., 0.

        n_batch = data_generator.n_train  // args.batch_size + 1

        for idx in range(n_batch):
            users, user_set, items, item_set = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss  = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                feed_dict={model.users: users, model.user_set: user_set, model.items: items, model.item_set: item_set,
                           model.mess_dropout: eval(args.mess_dropout),
                           model.item_weights: data_generator.item_weights})

            loss += batch_loss

            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss


        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  ]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss )
                print(perf_str)
            continue

        t2 = time()

        group_to_test = data_generator.test_group_set
        ret = test(sess, model, list(data_generator.test_users), group_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f ]\n recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f],  ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss,   ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
            paras = str(args.lr) + "_" + str(args.regs) + "_" + str(args.mess_dropout) + "_" + str(args.embed_size) + "_" + str(
                args.adj_type) + "_" + str(args.alg_type)
            print("paras\t", paras)


        # cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['precision'][0], cur_best_pre_0,
        #                                                             stopping_step, expected_order='acc', flag_step=10)

        cur_best_pre_0, stopping_step, should_stop = no_early_stopping(ret['precision'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc')

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.

        if should_stop == True:
            print('early stopping')
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['precision'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_pres_0 = max(pres[:, 0])
    idx = list(pres[:, 0]).index(best_pres_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s],  ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result-SMGCN-%d' % (args.proj_path, args.dataset, model.model_type, args.result_index)
    ensureDir(save_path)
    f = open(save_path, 'a')

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.write(
        'time=%s, fusion=%s, embed_size=%d, lr=%s, layer_size=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\t'
        % (str(cur_time), args.fusion, args.embed_size,   str(args.lr), args.layer_size,
           args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()

    endTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end ', endTime)

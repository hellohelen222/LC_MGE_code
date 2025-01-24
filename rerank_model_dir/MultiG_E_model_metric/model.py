# coding: utf-8 -*-
from data_input import *
from tools import *
from config import *
from util import *
import numpy as np
import tensorflow as tf
from layers import *
import os
from collections import namedtuple
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DNN:
    def __init__(self):
        pass

    # env_feature = > dense_feature
    # cxr_feature = > screen_predict_feature
    # cat_feature = > screen_cate_feature
    # dense_feature = > screen_dense_feature

    def inter_attr_self_attention(self,input):

        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input,[-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output, self.inter_attr_self_attention_weights['weight_MLP'])

        return output

    def intra_attr_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V), (-1, POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        return output

    def _oam(self,input):

        self.intra_attr_output = tf.identity(input)
        intra_attr_output_each_poi = tf.identity(input)
        for i in range(FEATURE_CATE_NUM):
            self.tmp_intra_attr_input = tf.reshape(tf.gather(input,list(range(i,i+1)),axis=2),[-1,POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_intra_attr_output = self.intra_attr_self_attention(self.tmp_intra_attr_input)
            self.tmp_intra_attr_output_mlp = tf.matmul(self.tmp_intra_attr_input + self.tmp_intra_attr_output, self.intra_attr_self_attention_weights['weight_MLP'])
            if i == 0:
                self.intra_attr_output = tf.expand_dims(self.tmp_intra_attr_output_mlp,axis=1)
                intra_attr_output_each_poi = tf.expand_dims(tf.reshape(self.tmp_intra_attr_output,[-1,POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]]),axis=2)
            else:
                self.intra_attr_output = tf.concat([self.intra_attr_output,tf.expand_dims(self.tmp_intra_attr_output_mlp, axis=1)],axis=1)
                intra_attr_output_each_poi = tf.concat([intra_attr_output_each_poi,tf.expand_dims(tf.reshape(self.tmp_intra_attr_output, [-1, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]]), axis=2)],axis=2)

        inter_attr_output = self.inter_attr_self_attention(self.intra_attr_output)

        # inter_attr_output: -1 * 8
        # intra_attr_output_each_poi : -1 * POI_NUM * feanum * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]
        return inter_attr_output,intra_attr_output_each_poi

    def _user_interest_target_attention(self,target,behavior,mask):
        target = tf.tile(tf.expand_dims(target,axis=1),[1,tf.shape(behavior)[1],1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights,axis=1) # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights,axis=2) * behavior,axis=1)
        return user_interest # batch * 8

    def _user_interest_target_attention_topK(self,target,behavior,mask,k):
        target = tf.tile(tf.expand_dims(target,axis=1),[1,tf.shape(behavior)[1],1])
        behavior = tf.tile(behavior,[k,1,1])
        mask = tf.tile(mask,[k,1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights,axis=1) # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights,axis=2) * behavior,axis=1)
        return user_interest # batch * 8

    def _intra_poi_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input, [-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output,
                           self.intra_poi_self_attention_weights['weight_MLP'])

        return output

    
    def positional_encoding(self,zero_pad=True,scale=True):

        N = 1
        T = 5
        num_units = 8

        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # batch * 8

        position_enc = np.array([[pos / np.power(10000, 2. * i / num_units) for i in range(num_units)] for pos in range(T)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5



        return outputs


    def _PPML(self,features):
        position_encoding = features['position_encoding']
        position_encoding = tf.expand_dims(tf.expand_dims(position_encoding, axis=0),axis=0)
        self.ppml_page = tf.identity(self.behavior_cate_feature_embeddings)
        self.ppml_permuation = tf.identity(self.behavior_cate_feature_embeddings)
        # batch * 5 * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_behavior_input_embedding = tf.reshape(tf.gather(self.behavior_cate_feature_embeddings,list(range(i,i+1)),axis=3),[-1,PAGE_NUM, POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_behavior_input_embedding = tf.reduce_mean(self.tmp_behavior_input_embedding * position_encoding,axis=2) # batch * page * emb
            if i == 0:
                self.ppml_page = tf.expand_dims(self.tmp_behavior_input_embedding,axis=2)
            else:
                self.ppml_page = tf.concat([self.ppml_page,tf.expand_dims(self.tmp_behavior_input_embedding,axis=2)],axis=2)
        self.ppml_page = tf.reduce_mean(self.ppml_page,axis=2) # # batch * page * emb
        # batch * pe * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_permuation_input_embedding = tf.reshape(tf.gather(self.full_permuation_embeddings,list(range(i,i+1)),axis=3),[-1,PERMUATION_SIZE, POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_permuation_input_embedding = tf.reduce_mean(self.tmp_permuation_input_embedding * position_encoding,axis=2) # batch * page * emb
            if i == 0:
                self.ppml_permuation = tf.expand_dims(self.tmp_permuation_input_embedding,axis=2)
            else:
                self.ppml_permuation = tf.concat([self.ppml_permuation,tf.expand_dims(self.tmp_permuation_input_embedding,axis=2)],axis=2)
        self.ppml_permuation = tf.reduce_mean(self.ppml_permuation, axis=2)  # # batch * permuation * emb

    def _SimHash_TopOne(self,features):
        page_hash_vector = tf.expand_dims(features['hash_vector'],axis=0) # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector) # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature,axis=1),[1,PERMUATION_SIZE,1,1]) # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'],axis=0),axis=0) # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation,axis=2),[1,1,PAGE_NUM,1])
        self.permuation_hash_signature = tf.sign(self.ppml_permuation * permuation_hash_vector) # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),axis=3) # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(tf.expand_dims(features['disantce_weight'],axis=0) * hamming_distance,axis=2) # batch * permution
        top_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=1).indices # batch * k
        last_permutation_indicces = tf.nn.top_k(weighted_hamming_distance,k=1).indices
        return top_permutation_indicces,last_permutation_indicces

    def _SimHash(self,features): #yhl:120 select topk(20)
        page_hash_vector = tf.expand_dims(features['hash_vector'],axis=0) # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector) # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature,axis=1),[1,PERMUATION_SIZE,1,1]) # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'],axis=0),axis=0) # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation,axis=2),[1,1,PAGE_NUM,1])
        self.permuation_hash_signature = tf.sign(self.ppml_permuation * permuation_hash_vector) # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),axis=3) # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(tf.expand_dims(features['disantce_weight'],axis=0) * hamming_distance,axis=2) # batch * permution
        # # select top40
        # top_30_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=TOP_K + 10).indices # batch * k
        top_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=TOP_K).indices # batch * k
        last_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance,k=TOP_K).indices
        return top_k_permutation_indicces,last_k_permutation_indicces

    # 简化处理，使用全排列精排累积ctr的topK
    def BeamSearch(self):
        expose_rate = tf.expand_dims(tf.constant(EXPOSE_RATE_FOR_BEAM_SEARCH,tf.float32),axis=0)
        full_permuation_feat_predict = tf.reshape(self.full_permuation_feat_predict,[-1,PERMUATION_SIZE,POI_NUM]) * expose_rate
        cul_ctr = tf.reduce_sum(full_permuation_feat_predict,axis=2) # batch * permutation
        top_k_permutation_indicces = tf.nn.top_k(cul_ctr, k=TOP_K).indices  # batch * k
        return top_k_permutation_indicces

    def _FGSM(self,features):
        # get behavior and permuation emb
        self._PPML(features)
        self.top_k_permutation_indicces,last_k_permutation_indicces = self._SimHash(features)
        random_k_permutation_indicces = tf.tile(features['random_indices'],[tf.shape(self.top_k_permutation_indicces)[0],1])
        beam_k_permutation_indicces = self.BeamSearch()

        # top k
        self.top_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,self.top_k_permutation_indicces) # batch * k * 5 * 7 * 8
        # 在这个地方加scm,生成序列跟其他序列的距离
        self.top_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,self.top_k_permutation_indicces) # # batch * k * 5 * 1
        self.top_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,self.top_k_permutation_indicces) # # batch * k * 5 * 1
        self.top_k_permuation_embeddings = tf.stop_gradient(self.top_k_permuation_embeddings)
        self.top_k_permuation_predict_fea = tf.stop_gradient(self.top_k_permuation_predict_fea)
        self.top_k_permuation_mask = tf.stop_gradient(self.top_k_permuation_mask)

        # lask k
        self.last_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,last_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.last_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_embeddings = tf.stop_gradient(self.last_k_permuation_embeddings)
        self.last_k_permuation_predict_fea = tf.stop_gradient(self.last_k_permuation_predict_fea)
        self.last_k_permuation_mask = tf.stop_gradient(self.last_k_permuation_mask)

        # random k
        self.random_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,random_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.random_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,random_k_permutation_indicces)  # # batch * k * 5 * 1
        self.random_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,random_k_permutation_indicces)  # # batch * k * 5 * 1

        # beam search k
        self.beam_search_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,beam_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.beam_search_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,beam_k_permutation_indicces)  # # batch * k * 5 * 1
        self.beam_search_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,beam_k_permutation_indicces)  # # batch * k * 5 * 1
    
    #单点分布Pointwise(indp)
    def _Pointwise_generator_model(self, features, labels, mode, params):
        # get behavior and permuation emb
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)
        self._PPML(features)
        self.top_permutation_indicces,last_permutation_indicces = self._SimHash_TopOne(features)

        # top k
        self.top_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,self.top_permutation_indicces) # [bs, 1, 5, 7, 8]
        self.top_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,self.top_permutation_indicces) # # batch * 1 * 5 * 1
        self.top_permuation_mask = tf.batch_gather(self.full_permuation_mask,self.top_permutation_indicces) # # batch * k * 5 * 1
        target_page_predict_fea = tf.reshape(self.top_permuation_predict_fea,[-1, POI_NUM, 1])

        self.top_permuation_embeddings = tf.stop_gradient(self.top_permuation_embeddings)
        self.top_permuation_predict_fea = tf.stop_gradient(self.top_permuation_predict_fea)
        self.top_permuation_mask = tf.stop_gradient(self.top_permuation_mask)

        pointwise_ctr_out = tf.layers.dense(target_page_predict_fea, OUT_NUM, activation=None, name="pointwise_final_out_ctr",reuse=tf.AUTO_REUSE)


        if not self.train:
                pointwise_ctr_out = tf.nn.sigmoid(pointwise_ctr_out)

        self.pointwise_generator_out = tf.reshape(pointwise_ctr_out,[-1,POI_NUM]) # bs*5
        # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
        # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
        # 使用 tf.argsort 来获取排序后的索引
        self.pointwise_generator_permution_index = tf.argsort(self.pointwise_generator_out, axis=1, direction='DESCENDING') #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
        self.pointwise_generator_permution_index = tf.stop_gradient(self.pointwise_generator_permution_index)

        self.pointwise_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.pointwise_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.pointwise_generator_embedding = tf.expand_dims(self.pointwise_generator_embedding, axis=1)  # 形状变为 [bs, 1, 5,7,8]
        self.pointwise_generator_embedding = tf.stop_gradient(self.pointwise_generator_embedding)

        self.pointwise_generator_permution_index_expand = tf.expand_dims(self.pointwise_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
        self.pointwise_generator_permution_index_expand = tf.stop_gradient(self.pointwise_generator_permution_index_expand)

        self.pointwise_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.pointwise_generator_permution_index),[-1, POI_NUM, 1])
        self.pointwise_permuation_mask = tf.expand_dims(self.pointwise_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
        self.pointwise_permuation_mask = tf.stop_gradient(self.pointwise_permuation_mask)
        # self.Q_network_output = self.out 

    
    #位置分布ListWise
    def _Listwise_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            listwise_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.listwise_cur_page_oam_output,self.listwise_intra_attr_output_each_poi = self._oam(listwise_cate_feature_embeddings_stopped)


            self.listwise_behavior_page_oam_output = tf.identity(self.listwise_cur_page_oam_output)
            for i in range(PAGE_NUM):
                listwise_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(self.behavior_cate_feature_embeddings)
                self.listwise_tmp_behavior_input = tf.reshape(tf.gather(listwise_behavior_cate_feature_embeddings_stopped,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_behavior_oam_output,_ = self._oam(self.listwise_tmp_behavior_input)
                if i == 0:
                    self.listwise_behavior_page_oam_output = tf.expand_dims(self.listwise_tmp_behavior_oam_output,axis=1)
                else:
                    self.listwise_behavior_page_oam_output = tf.concat([self.listwise_behavior_page_oam_output,tf.expand_dims(self.listwise_tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            listwise_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.listwise_user_interest_emb = self._user_interest_target_attention(self.listwise_cur_page_oam_output,self.listwise_behavior_page_oam_output,listwise_mask_feature_stopped)
            
            #features['page_mask'] noindp

            self.listwise_poi_rep_emb = self.listwise_intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.listwise_tmp_poi_input = tf.reshape(tf.gather(self.listwise_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_poi_output = self._intra_poi_self_attention(self.listwise_tmp_poi_input)
                if i == 0:
                    self.listwise_poi_rep_emb = tf.expand_dims(self.listwise_tmp_poi_output, axis=1)
                else:
                    self.listwise_poi_rep_emb = tf.concat([self.listwise_poi_rep_emb, tf.expand_dims(self.listwise_tmp_poi_output, axis=1)], axis=1)

            self.listwise_cur_page_oam_output = tf.tile(tf.expand_dims(self.listwise_cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.listwise_user_interest_emb = tf.tile(tf.expand_dims(self.listwise_user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            listwise_fc_out = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Listwise_MLP_A" + str(i)
                listwise_fc_out = tf.layers.dense(listwise_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                listwise_fc_out = tf.nn.swish(listwise_fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            listwise_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            listwise_fc_input = tf.concat([self.listwise_cur_page_oam_output, self.listwise_user_interest_emb, self.listwise_poi_rep_emb, listwise_feat_predict_stopped,listwise_fc_out], axis=2)

            listwise_fc_out_ctr, listwise_fc_out_imp = tf.reshape(listwise_fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(listwise_fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "Listwise_MLP_B" + str(i)
                listwise_fc_out_ctr = tf.layers.dense(listwise_fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                listwise_fc_out_ctr = tf.nn.relu(listwise_fc_out_ctr)

            listwise_ctr_out = tf.layers.dense(listwise_fc_out_ctr, OUT_NUM, activation=None, name="listwise_final_out_ctr")

            if not self.train:
                listwise_ctr_out = tf.nn.sigmoid(listwise_ctr_out)

            self.listwise_generator_out = tf.reshape(listwise_ctr_out,[-1,POI_NUM]) # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.listwise_generator_permution_index = tf.argsort(self.listwise_generator_out, axis=1, direction='DESCENDING') #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.listwise_generator_permution_index = tf.stop_gradient(self.listwise_generator_permution_index)

            self.listwise_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.listwise_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.listwise_generator_embedding = tf.expand_dims(self.listwise_generator_embedding, axis=1)  # [bs, 1, 5,7,8]
            self.listwise_generator_embedding = tf.stop_gradient(self.listwise_generator_embedding)

            self.listwise_generator_permution_index_expand = tf.expand_dims(self.listwise_generator_permution_index, axis=1)  # [bs, 1, 5]
            self.listwise_generator_permution_index_expand = tf.stop_gradient(self.listwise_generator_permution_index_expand)

            self.listwise_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.listwise_generator_permution_index),[-1, POI_NUM, 1])
            self.listwise_permuation_mask = tf.expand_dims(self.listwise_permuation_mask, axis=1)  # [bs,1,5,1]
            self.listwise_permuation_mask = tf.stop_gradient(self.listwise_permuation_mask)
            # self.Q_network_output = self.out 
    
    #生成式ListWise
    def _headlayers(self, top_tensor_list, position_tensor):
        """
        top_tensor_list: [batch_size, 50 embedding_size]]
        position_tensor: [batch_size , embedding_size]
        output: [batch_size, predict_len, 1]
        """
        position_tensor = tf.tile(tf.expand_dims(position_tensor, axis=1), [1, POI_NUM, 1])
        output = tf.concat([top_tensor_list, position_tensor], axis=2)
        with tf.variable_scope("head_layers", reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(output, 64, activation=tf.nn.relu, name="head_1")
            output = tf.layers.dense(output, 16, activation=tf.nn.relu, name="head_2")
            output = tf.layers.dense(output, 1, activation=None, name="score")
        return output

    def _Generative_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.generative_cur_page_oam_output,self.generative_intra_attr_output_each_poi = self._oam(generative_cate_feature_embeddings_stopped)

            self.generative_behavior_page_oam_output = tf.identity(self.generative_cur_page_oam_output)
            for i in range(PAGE_NUM):
                generative_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(self.behavior_cate_feature_embeddings)
                self.generative_tmp_behavior_input = tf.reshape(tf.gather(generative_behavior_cate_feature_embeddings_stopped,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_behavior_oam_output,_ = self._oam(self.generative_tmp_behavior_input)
                if i == 0:
                    self.generative_behavior_page_oam_output = tf.expand_dims(self.generative_tmp_behavior_oam_output,axis=1)
                else:
                    self.generative_behavior_page_oam_output = tf.concat([self.generative_behavior_page_oam_output,tf.expand_dims(self.generative_tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            generative_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.generative_user_interest_emb = self._user_interest_target_attention(self.generative_cur_page_oam_output,self.generative_behavior_page_oam_output,generative_mask_feature_stopped)

            self.generative_poi_rep_emb = self.generative_intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.generative_tmp_poi_input = tf.reshape(tf.gather(self.generative_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_poi_output = self._intra_poi_self_attention(self.generative_tmp_poi_input)
                if i == 0:
                    self.generative_poi_rep_emb = tf.expand_dims(self.generative_tmp_poi_output, axis=1)
                else:
                    self.generative_poi_rep_emb = tf.concat([self.generative_poi_rep_emb, tf.expand_dims(self.generative_tmp_poi_output, axis=1)], axis=1)

            self.generative_cur_page_oam_output = tf.tile(tf.expand_dims(self.generative_cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.generative_user_interest_emb = tf.tile(tf.expand_dims(self.generative_user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            generative_fc_out = tf.reshape(generative_cate_feature_embeddings_stopped,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Generative_MLP_A" + str(i)
                generative_fc_out = tf.layers.dense(generative_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                generative_fc_out = tf.nn.swish(generative_fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            generative_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            generative_fc_input = tf.concat([self.generative_cur_page_oam_output, self.generative_user_interest_emb, self.generative_poi_rep_emb, generative_feat_predict_stopped,generative_fc_out], axis=2) # Batch_size * POI_NUM * FEAT_NUM
            generative_fc_input = tf.layers.dense(generative_fc_input, 64, activation=None, name="generative_fc_input_emb")

            # fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])
            batch_size      = tf.shape(generative_fc_input)[0]
            with tf.variable_scope("rnn_init_state", reuse=tf.AUTO_REUSE):
                state = tf.reduce_mean(generative_fc_input, axis=1)  # [B,64]
                state = tf.stop_gradient(state)
                # state = tf.layers.dense(state, STATE_EMBEDDING_SIZE, activation=None, name="layer_1")
                input_tensor = tf.zeros_like(state)
                # input_tensor = tf.get_variable('go', (1, STATE_EMBEDDING_SIZE))
                # input_tensor = tf.tile(input_tensor, [batch_size, 1])  # [batch_size, embedding_size]

            with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
                gru_cell = tf.nn.rnn_cell.GRUCell(num_units=64, name='gru_cell')  # 替换 tf.contrib.rnn.GRUCell
        
            if self.train:
                self.label = labels['ctr_label']
                self.generative_generator_loss = 0.0
                output_list = []
                label_list = []
                eval_weight_list = []
                generative_generator_out = []
                generative_generator_permution_index = []
                mask_tmp = tf.ones([batch_size, POI_NUM], dtype=tf.float32) # [B, 5]
                eval_ones = tf.ones([batch_size, POI_NUM], dtype=tf.float32) # [B, 5]

                for index in range(POI_NUM):
                    discount = 1.0 / np.log2(2.0 + index)
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor) #[B, 5,1], head_layer最后一层不用激活函数
                    head_score_origin = tf.reshape(head_score, [-1, POI_NUM]) #[B, 5]
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1.0 - mask_tmp, -1.0e9), head_score_origin)) #[B,5]
                    # lable
                    select_index = tf.tile(tf.expand_dims(tf.constant([index], dtype=tf.int32), axis=1), [tf.shape(head_score)[0], 1]) #[B, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32) #[B, 5]

                    # sample_reward = realshow_list * sample_weight
                    label_mask = self.label * select_onehot #[B, 5], 只有曝光位置有reward, 其他为 0
                    # loss
                    tmp_loss = -1.0 * tf.reduce_sum(label_mask * tf.log(tf.clip_by_value(head_score, 1e-9, 1.0)), axis=-1) # [B]
                    tmp_loss = tf.reduce_sum(tmp_loss)
                    self.generative_generator_loss = self.generative_generator_loss + tmp_loss
                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))
                    

                    output_list.append(head_score)
                    label_list.append(label_mask)
                    weight = mask_tmp * eval_ones
                    eval_weight_list.append(weight)

                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [B, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]
                    input_tensor = tf.stop_gradient(input_tensor)
            else:
                generative_generator_permution_index = []
                generative_generator_out = []
                mask_tmp = tf.ones([batch_size, PAGE_NUM], dtype=tf.float32) # [B, 5]
                ## 这边预测6个
                for index in range(POI_NUM):
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor) #[B, 50,1]
                    head_score = tf.reshape(head_score, [-1, POI_NUM])
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), head_score)) #[B,50]
   
                    select_index = tf.reshape(tf.argmax(head_score, axis=1, output_type=tf.int32), [-1, 1])  # [batch_size, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32)
                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [batch_size, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]

                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))
            


                

            self.generative_generator_out = tf.stack(generative_generator_out, axis=1) # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.generative_generator_permution_index = tf.stack(generative_generator_permution_index, axis=1) # bs*5 #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.generative_generator_permution_index = tf.stop_gradient(self.generative_generator_permution_index)

            self.generative_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.generative_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.generative_generator_embedding = tf.expand_dims(self.generative_generator_embedding, axis=1)  # 形状变为 [bs, 1, 5,7,8]
            self.generative_generator_embedding = tf.stop_gradient(self.generative_generator_embedding)

            self.generative_generator_permution_index_expand = tf.expand_dims(self.generative_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
            self.generative_generator_permution_index_expand = tf.stop_gradient(self.generative_generator_permution_index_expand)

            self.generative_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.generative_generator_permution_index),[-1, POI_NUM, 1])
            self.generative_permuation_mask = tf.expand_dims(self.generative_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
            self.generative_permuation_mask = tf.stop_gradient(self.generative_permuation_mask)
            # self.Q_network_output = self.out
    
    def _OCPM(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            self.cur_page_oam_output,self.intra_attr_output_each_poi = self._oam(self.cate_feature_embeddings)

            self.behavior_page_oam_output = tf.identity(self.cur_page_oam_output)
            for i in range(PAGE_NUM):
                self.tmp_behavior_input = tf.reshape(tf.gather(self.behavior_cate_feature_embeddings,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_behavior_oam_output,_ = self._oam(self.tmp_behavior_input)
                if i == 0:
                    self.behavior_page_oam_output = tf.expand_dims(self.tmp_behavior_oam_output,axis=1)
                else:
                    self.behavior_page_oam_output = tf.concat([self.behavior_page_oam_output,tf.expand_dims(self.tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            self.user_interest_emb = self._user_interest_target_attention(self.cur_page_oam_output,self.behavior_page_oam_output,features['page_mask'])

            self.poi_rep_emb = self.intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.tmp_poi_input = tf.reshape(tf.gather(self.intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_poi_output = self._intra_poi_self_attention(self.tmp_poi_input)
                if i == 0:
                    self.poi_rep_emb = tf.expand_dims(self.tmp_poi_output, axis=1)
                else:
                    self.poi_rep_emb = tf.concat([self.poi_rep_emb, tf.expand_dims(self.tmp_poi_output, axis=1)], axis=1)

            self.cur_page_oam_output = tf.tile(tf.expand_dims(self.cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.user_interest_emb = tf.tile(tf.expand_dims(self.user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            fc_out = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                fc_out = tf.nn.swish(fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat([self.cur_page_oam_output, self.user_interest_emb, self.poi_rep_emb, self.feat_predict,fc_out], axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_model_final_out_ctr")

            if not self.train:
                ctr_out = tf.nn.sigmoid(ctr_out)

            self.out = tf.reshape(ctr_out,[-1,POI_NUM])
            self.Q_network_output = self.out

    def _OCPM_TOP_K(self, features, labels, mode, params,k_page,k_predict,k,mask):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        target_page_embedding = tf.reshape(k_page,[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        target_page_predict_fea = tf.reshape(k_predict,[-1, POI_NUM, 1])

        with tf.name_scope('dnn_model'):
            cur_page_oam_output, target_intra_attr_output_each_poi = self._oam(target_page_embedding)

            behavior_page_oam_output = self.behavior_page_oam_output

            user_interest_emb = self._user_interest_target_attention_topK(cur_page_oam_output,behavior_page_oam_output,features['page_mask'],k)

            poi_rep_emb = target_intra_attr_output_each_poi # （batch * k） * poi * 8
            for i in range(POI_NUM):
                tmp_poi_input = tf.reshape(
                    tf.gather(target_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                tmp_poi_output = self._intra_poi_self_attention(tmp_poi_input)
                if i == 0:
                    poi_rep_emb = tf.expand_dims(tmp_poi_output, axis=1)
                else:
                    poi_rep_emb = tf.concat(
                        [poi_rep_emb, tf.expand_dims(tmp_poi_output, axis=1)], axis=1)

            cur_page_oam_output = tf.tile(tf.expand_dims(cur_page_oam_output, axis=1),[1, POI_NUM, 1])
            user_interest_emb = tf.tile(tf.expand_dims(user_interest_emb, axis=1), [1, POI_NUM, 1])

            fc_out = tf.reshape(target_page_embedding, [-1, POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name,reuse=tf.AUTO_REUSE)
                fc_out = tf.nn.swish(fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat(
                [cur_page_oam_output, user_interest_emb, poi_rep_emb, target_page_predict_fea, fc_out],axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input,[-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i],
                                             activation=None, name=dense_name,reuse=tf.AUTO_REUSE)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_topk_final_out_ctr",reuse=tf.AUTO_REUSE)
            ctr_out = tf.nn.sigmoid(ctr_out)

            topk_out = tf.reshape(ctr_out, [-1, k, POI_NUM])
            mask = tf.reshape(mask,[-1,k,POI_NUM])
            return topk_out * mask

    


    def _create_loss(self, labels):
        with tf.name_scope('loss'):
            self.label = labels['ctr_label']
            self.mask = labels['mask']
            # ctr_loss 
            self.pointwise_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.pointwise_generator_out,weights=self.mask))
            self.listwise_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.listwise_generator_out, weights=self.mask))
            # self.submodularity_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.submodularity_generator_out, weights=self.mask))

            self.evaluator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.max_list_scores, weights=self.mask))
            self.loss = self.pointwise_generator_loss + self.listwise_generator_loss  + self.evaluator_loss + self.generative_generator_loss
            # if USE_CONSTRATIVE_LOSS:
            #     self.loss -= CONSTRATIVE_LOSS_K * (tf.reduce_mean(self.topK_output,axis=[0,1,2]) - tf.reduce_mean(self.lastK_output,axis=[0,1,2]))

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
    
    def _create_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'embedding': tf.get_variable('embedding',
                                             shape=CATE_FEATURE_EMBEDDINGS_SHAPE,
                                             initializer=tf.zeros_initializer(),
                                            )
            }

            glorot = np.sqrt(2.0 / (4 + 1))

            self.intra_attr_self_attention_weights = {
                'weight_Q':tf.get_variable(name='intra_attr_self_attention_weights_Q',shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1],CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                           initializer=tf.random_normal_initializer(0.0, glorot),
                                           dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_attr_self_attention_weights_MLP',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32)
            }

            self.inter_attr_self_attention_weights = {
                'weight_Q': tf.get_variable(name='inter_attr_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='inter_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='inter_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='inter_attr_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

            self.intra_poi_self_attention_weights = {
                'weight_Q': tf.get_variable(name='intra_poi_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_poi_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_poi_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_poi_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

    def _process_features(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        cate_fea = tf.reshape(features['cate_feature'],[-1,POI_NUM * FEATURE_CATE_NUM])
        self.cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], cate_fea),
            [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        full_permuation_index = tf.reshape(features['full_permuation_index'],[PERMUATION_SIZE,POI_NUM])

        self.cate_feature_embeddings_for_permuation = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.full_permuation_index = tf.reshape(tf.tile(tf.expand_dims(full_permuation_index,axis=0),[tf.shape(self.cate_feature_embeddings_for_permuation)[0],1,1]),[-1,POI_NUM * PERMUATION_SIZE])
        self.full_permuation_embeddings = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.full_permuation_index),[-1,PERMUATION_SIZE,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        
        # self.input_embedding = tf.concat(
        #     [self.cate_feature_embeddings], axis=2)
        self.feat_predict = features['dense_feature']
        self.feat_predict_for_permuation = tf.reshape(self.feat_predict,[-1,POI_NUM,1])

        self.full_permuation_feat_predict = tf.reshape(tf.batch_gather(self.feat_predict_for_permuation,self.full_permuation_index),[-1,PERMUATION_SIZE,POI_NUM,1])


        self.permuation_mask = tf.reshape(features['permuation_mask'],[-1,POI_NUM,1])
        self.full_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.full_permuation_index),[-1, PERMUATION_SIZE, POI_NUM, 1])


        behavior_cate_fea = tf.reshape(features['behavior_cate_feature'],[-1,PAGE_NUM * POI_NUM * FEATURE_CATE_NUM])
        self.behavior_cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], behavior_cate_fea),
            [-1, PAGE_NUM , POI_NUM, FEATURE_CATE_NUM ,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        self.behavior_input_embedding = tf.concat([self.behavior_cate_feature_embeddings], axis=3)
        self.behavior_feat_predict = features['behavior_dense_feature']

        return features
    
    #两两之间的逆序对
#    def _count_inversions(self, L1, L2):
#        inversions = 0
        
#        # 遍历所有可能的元素对 (i, j) 且 i < j
#        for i in range(POI_NUM):
#            for j in range(i + 1, POI_NUM):
#                # L1 中 i 元素和 j 元素的顺序
#                L1_i, L1_j = L1[i], L1[j]
#                # L2 中 i 元素和 j 元素的顺序
#                L2_i = L2.index(L1_i)
#                L2_j = L2.index(L1_j)
#                
#                # 如果 L1 中 i < j，但 L2 中 i > j，则计为逆序对
#                if L2_i > L2_j:
#                    inversions += 1
#
#        return inversions
    def _count_inversions(self, L1, L2):
        """
        使用 TensorFlow 计算两个列表 L1 和 L2 之间的逆序对数量
        :param L1: 形状为 [n] 的张量
        :param L2: 形状为 [n] 的张量
        :return: 逆序对数量
        """
        inversions = 0

        # 通过 tf.argsort 获取 L2 中元素的索引顺序
        L2_indices = tf.argsort(L2)  # 返回 L2 中元素的排序顺序
        L2_sorted = tf.gather(L2, L2_indices)  # 按照索引顺序重新排列 L2

        # 遍历所有可能的元素对 (i, j) 且 i < j
        for i in range(POI_NUM):
            for j in range(i + 1, POI_NUM):
                L1_i = L1[i]
                L1_j = L1[j]

                # 使用 tf.argmax 找到 L1_i 和 L1_j 在 L2_sorted 中的索引
                L2_i = tf.argmax(tf.cast(tf.equal(L2_sorted, L1_i), tf.int32))
                L2_j = tf.argmax(tf.cast(tf.equal(L2_sorted, L1_j), tf.int32))

                # 如果 L1 中 i < j，但 L2 中 i > j，则计为逆序对
                inversions += tf.reduce_sum(tf.cast(L2_i > L2_j, tf.float32))

        return inversions
    # 计算逆序对LC
    def _total_inversions_multiple_lists(self, lists, generator_num):
        total_inversions = 0  # 用于存储所有列表逆序对的总和
        
        # 遍历每对列表 (Li, Lj) 进行两两比较
        for i in range(generator_num):
            for j in range(i + 1, generator_num):
                inversions = self._count_inversions(lists[i,:], lists[j,:])
                total_inversions += inversions
        
        return total_inversions

    def _calculate_total_inversions(self,batch_lists, generator_num):
        """
        处理形状为 [bs, k, 5] 的输入，计算每个样本的逆序对总和
        """
        # bs = tf.shape(batch_lists)[0]
        # k = tf.shape(batch_lists)[1]
        batch_inversions = []
        
        for b in range(BATCH_SIZE):
            # 取出第 b 个 batch 的 k 个 list
            lists = batch_lists[b,:,:]
            # 计算该样本中 k 个 list 之间的逆序对总和
            inversions = self._total_inversions_multiple_lists(lists, generator_num)
            batch_inversions.append(inversions)
        
        return batch_inversions
    
        # #计算逆序对GC 新增list跟原本多个list之间的gc增益
    def _inversions_with_new_list(self, lists, new_list, generator_num):
        # bs = tf.shape(lists)[0]
        # k = tf.shape(lists)[1]
        total_inversions = []

        # 计算每个批次中，新的列表与每个现有列表之间的逆序对
        for i in range(BATCH_SIZE):
            batch_inversions = 0
            # 对于每个现有的列表，计算与新列表的逆序对
            for j in range(generator_num):
                existing_list = lists[i, j, :]  # 取出第 i 个批次的第 j 个列表
                new_list_i = new_list[i, 0, :]  # 取出第 i 个批次的新的列表 (shape: [5])
                inversions = self._count_inversions(existing_list, new_list_i)
                batch_inversions += inversions
            total_inversions.append(batch_inversions)
        
        return total_inversions
    
    # def _count_inversions_tensor(self, L1, L2):
    #     """
    #     使用 TensorFlow 计算两个列表 L1 和 L2 之间的逆序对数量
    #     L1 和 L2 都是形状 [bs, k, 5] 的 tensor
    #     """
    #     n = tf.shape(L1)[-1]  # 获取列表长度 (这里是5)
    #     # 生成所有可能的 i, j 对，其中 i < j
    #     i = tf.tile(tf.reshape(tf.range(n), [1, 1, n]), [tf.shape(L1)[0], tf.shape(L1)[1], 1])
    #     j = tf.tile(tf.reshape(tf.range(1, n), [1, 1, n-1]), [tf.shape(L1)[0], tf.shape(L1)[1], 1])
        
    #     # 将 i 和 j 扩展以便广播
    #     i = tf.expand_dims(i, axis=-1)
    #     j = tf.expand_dims(j, axis=-1)
        
    #     # 取出 L1 和 L2 中对应的元素
    #     L1_i = tf.gather(L1, i, batch_dims=2)
    #     L1_j = tf.gather(L1, j, batch_dims=2)
    #     L2_i = tf.gather(L2, i, batch_dims=2)
    #     L2_j = tf.gather(L2, j, batch_dims=2)
        
    #     # 判断是否为逆序对：L1 中 i < j 且 L2 中 i > j
    #     inversions = tf.cast(tf.greater(L2_i, L2_j), tf.float32) * tf.cast(tf.less(L1_i, L1_j), tf.float32)
        
    #     # 返回每对 (i, j) 的逆序对数量
    #     return tf.reduce_sum(inversions, axis=-1)  # 返回逆序对数量
    # def _total_inversions_multiple_lists(self, lists):
    #     """
    #     计算多个列表之间的逆序对数量
    #     :param lists: 形状为 [bs, k, 5] 的 Tensor
    #     :return: 返回每个批次中的逆序对总数
    #     """
    #     bs = tf.shape(lists)[0]
    #     k = tf.shape(lists)[1]
        
    #     # 使用 tf.combinations 获取所有 (i, j) 对 (i < j)
    #     combinations = tf.combinations(tf.range(k), 2)
        
    #     # 初始化存储逆序对总数的变量
    #     batch_inversions = tf.zeros([bs], dtype=tf.float32)
        
    #     # 遍历所有 (i, j) 对并计算逆序对数量
    #     for pair in combinations:
    #         i, j = pair[0], pair[1]
    #         # 获取当前的 L1 和 L2
    #         L1 = lists[:, i, :]  # 选择第 i 个列表
    #         L2 = lists[:, j, :]  # 选择第 j 个列表
            
    #         # 计算逆序对数量
    #         inversions = self._count_inversions_tensor(L1, L2)
            
    #         # 将逆序对数量加到 batch_inversions 上
    #         batch_inversions += inversions
        
    #     return batch_inversions
    # def _calculate_inversions(self,pair):
    #     L1, L2 = pair
        
    #     # 获取列表长度
    #     n = tf.shape(L1)[-1]
        
    #     # 创建索引对 (i, j) 且 i < j
    #     i, j = tf.meshgrid(tf.range(n), tf.range(n))
    #     mask = tf.cast(i < j, tf.float32)
        
    #     # 取出L1和L2中的索引对应的元素
    #     L1_i = tf.gather(L1, i)
    #     L1_j = tf.gather(L1, j)
    #     L2_i = tf.gather(L2, i)
    #     L2_j = tf.gather(L2, j)
        
    #     # 计算逆序对数量：在L1中顺序，但在L2中逆序的对
    #     inversions = tf.cast((L1_i < L1_j) & (L2_i > L2_j), tf.float32)
        
    #     # 返回加权的逆序对数量
    #     return tf.reduce_sum(inversions * mask)

    # def calculate_batch_inversions(self, batch_lists):
    #     # 获取批次大小和list的数量
    #     bs = tf.shape(batch_lists)[0]
    #     k = tf.shape(batch_lists)[1]

    #     # 生成有效的list配对 (L1, L2)
    #     valid_pairs = [(batch_lists[:, i, :], batch_lists[:, j, :]) for i in range(k) for j in range(i + 1, k)]

    #     # 计算每对list之间的逆序对
    #     inversions_per_pair = tf.map_fn(self._calculate_inversions, valid_pairs, dtype=tf.float32)

    #     # 合并所有pair的逆序对
    #     batch_inversions = tf.reduce_sum(inversions_per_pair, axis=0)
        
    #     return batch_inversions
    
    # def _inversions_with_new_list(self, lists, new_list, generator_num):
    #     """
    #     计算新 list 与已有 list 之间的逆序对
    #     """
    #     bs = tf.shape(lists)[0]
    #     k = tf.shape(lists)[1]
    #     # 扩展 new_list，使其形状为 [bs, k, 5]，以便与已有 list 对比
    #     expanded_new_list = tf.tile(new_list, [1, k, 1])  # 形状变为 [bs, k, 5]
        
    #     # 计算新 list 和每个已有 list 之间的逆序对
    #     inversions = self._count_inversions_tensor(expanded_new_list, lists)
        
    #     return inversions


    
    # def _dcg_at_k(self, ranking, k=5):
    #     """计算 DCG 值"""
    #     gain = 2 ** ranking - 1  # 计算增益，通常可以是相关度值
    #     discount = np.log2(np.arange(2, k+2))  # 位置上的折扣因子，log2(i+1)
    #     return np.sum(gain / discount)

    # def _ndcg_at_k(self,evaluator_out_argsort, label, k=5):
    #     """计算 NDCG"""
    #     batch_size = tf.shape(evaluator_out_argsort)[0]
    #     ndcg_scores = []
        
    #     for i in range(batch_size):
    #         # 获取当前样本的排序索引和标签
    #         sorted_indices = evaluator_out_argsort[i]  # shape = [5]
    #         actual_labels = label[i]  # shape = [5]
            
    #         # 根据标签计算理想的排序（即标签值的降序排列）
    #         ideal_sorted_labels = np.argsort(actual_labels)[::-1]  # 由高到低排序的索引
            
    #         # 计算实际的 DCG 和理想的 DCG
    #         dcg = self._dcg_at_k(actual_labels[sorted_indices], k)
    #         idcg = self._dcg_at_k(actual_labels[ideal_sorted_labels], k)
            
    #         # 计算 NDCG，避免除以 0
    #         ndcg = dcg / idcg if idcg > 0 else 0.0
    #         ndcg_scores.append(ndcg)
        
    #     return np.mean(ndcg_scores)
    
    def _ndcg_at_k(self, evaluator_out_argsort, label, k=5):
        """计算 NDCG"""
        # 获取批次大小
        batch_size = tf.shape(evaluator_out_argsort)[0]
        
        # 获取排序索引
        sorted_indices = evaluator_out_argsort  # shape = [bs, k]
        actual_labels = label  # shape = [bs, k]
        
        # 根据标签计算理想的排序（即标签值的降序排列）
        ideal_sorted_labels = tf.argsort(actual_labels, axis=-1, direction='DESCENDING')  # shape = [bs, k]
        
        # 使用向量化的方式计算 DCG 和 IDCG
        def dcg_at_k(labels, indices, k):
            sorted_labels = tf.gather(labels, indices, axis=1)  # 根据给定的排序索引，获取排序后的标签
            gain = tf.pow(2.0, sorted_labels) - 1  # 使用加权函数计算 gain
            discount = tf.math.log(tf.cast(tf.range(1, k + 1), tf.float32) + 1.0) / tf.math.log(2.0)  # 对应位置的折扣
            return tf.reduce_sum(gain / discount, axis=-1)  # 计算 DCG
        
        # 计算实际的 DCG 和理想的 DCG
        dcg = dcg_at_k(actual_labels, sorted_indices, k)
        idcg = dcg_at_k(actual_labels, ideal_sorted_labels, k)
        
        # 计算 NDCG，避免除以 0
        ndcg = tf.where(idcg > 0, dcg / idcg, tf.zeros_like(dcg))
        
        # 返回每个批次的 NDCG 平均值
        return tf.reduce_mean(ndcg)
    
    def _compute_pairwise_distances(self,embedding):
        # 1. 将最后的 7x8 展平为 56
        #bs, k, _, _, _ = embedding.shape
        batch_size = tf.shape(embedding)[0]
        listnum = tf.shape(embedding)[1]
        embedding_flat = tf.reshape(embedding, [batch_size, listnum, POI_NUM, -1])  # [bs, k, 5, 56]
        
        # 2. 扩展维度以计算 pairwise 距离
        embedding_expanded_1 = tf.expand_dims(embedding_flat, axis=2)  # [bs, k, 1, 5, 56]
        embedding_expanded_2 = tf.expand_dims(embedding_flat, axis=1)  # [bs, 1, k, 5, 56]
        
        # 3. 计算 pairwise 差异
        pairwise_diff = embedding_expanded_1 - embedding_expanded_2  # [bs, k, k, 5, 56]
        
        # 4. 计算 pairwise 欧氏距离 (平方和再开方)
        pairwise_distance = tf.sqrt(tf.reduce_sum(tf.square(pairwise_diff), axis=-1) + 1e-8)  # [bs, k, k, 5]
        
        # 5. 计算每对列表之间的平均距离 (对第 3 维的 5 个元素取平均)
        pairwise_distance_mean = tf.reduce_mean(pairwise_distance, axis=-1)  # [bs, k, k]
        
        # 6. 排除与自身的距离 (将对角线元素置为无穷大)
        mask = tf.eye(listnum, batch_shape=[batch_size], dtype=tf.bool)  # 对角线为 True，其余为 False
        valid_pairwise_distances = tf.where(mask, tf.zeros_like(pairwise_distance_mean), pairwise_distance_mean)  # 对角线设为 0
    
        # 7. 计算每个批次下的最大距离和平均距离
        max_distance = tf.reduce_max(valid_pairwise_distances, axis=[1, 2])  # [bs]
        mean_distance = tf.reduce_mean(valid_pairwise_distances, axis=[1, 2])  # [bs]
        
        return max_distance, mean_distance
    

    def _compute_cosine_distances(self,embedding):
        # 1. 将最后的 7x8 展平为 56 维
        batch_size = tf.shape(embedding)[0]
        listnum = tf.shape(embedding)[1]
        embedding_flat = tf.reshape(embedding, [batch_size, listnum, POI_NUM, -1])  # [bs, k, 5, 56]
        
        # 2. 计算嵌入向量的 L2 范数
        embedding_norm = tf.norm(embedding_flat, axis=-1)  # [bs, k, 5]
        
        # 3. 扩展维度以计算 pairwise 余弦相似度
        embedding_expanded_1 = tf.expand_dims(embedding_flat, axis=2)  # [bs, k, 1, 5, 56]
        embedding_expanded_2 = tf.expand_dims(embedding_flat, axis=1)  # [bs, 1, k, 5, 56]
        
        # 4. 扩展 L2 范数以匹配 pairwise 的维度
        norm_expanded_1 = tf.expand_dims(embedding_norm, axis=2)  # [bs, k, 1, 5]
        norm_expanded_2 = tf.expand_dims(embedding_norm, axis=1)  # [bs, 1, k, 5]
        
        
        # 5. 计算 pairwise dot product
        dot_product = tf.reduce_sum(embedding_expanded_1 * embedding_expanded_2, axis=-1)  # [bs, k, k, 5]
        
        # 6. 计算 pairwise cosine similarity
        cosine_similarity = dot_product / (norm_expanded_1 * norm_expanded_2 + 1e-8)  # [bs, k, k, 5]
        
        # 7. 计算 pairwise cosine distance (1 - cosine similarity)
        cosine_distance = 1 - cosine_similarity  # [bs, k, k, 5]
        
        # 8. 对每对列表之间的 5 个元素的余弦距离求平均值
        pairwise_distance_mean = tf.reduce_mean(cosine_distance, axis=-1)  # [bs, k, k]
        
        # 9. 将对角线元素（自身与自身的比较）设为无穷大
        mask = tf.linalg.diag(tf.ones([batch_size, listnum], dtype=tf.bool))
        valid_pairwise_distances = tf.where(mask, tf.zeros_like(pairwise_distance_mean), pairwise_distance_mean)
        
        # 10. 计算每个批次中的最大距离和平均距离
        max_distance = tf.reduce_max(valid_pairwise_distances, axis=[1, 2])  # [bs]
        mean_distance = tf.reduce_mean(valid_pairwise_distances, axis=[1, 2])  # [bs]
        
        return max_distance, mean_distance

    def _create_indicator(self, labels):
        ctr_out = self.out
        ctr_out = tf.reduce_mean(tf.nn.sigmoid(ctr_out))

        # All gradients of loss function wrt trainable variables
        '''
        grads = tf.gradients(self.loss, tf.trainable_variables())
        for grad, var in list(zip(grads, tf.trainable_variables())):
            tf.summary.histogram(var.name + '/gradient', grad)
        '''
        def format_log(tensors):
            def convert_to_scalar(tensor):
                if isinstance(tensor, np.ndarray):
                    return float(tensor.item()) if tensor.ndim == 0 else tensor
                elif isinstance(tensor, tf.Tensor):
                    return float(tensor.numpy().item()) if tensor.ndim == 0 else tensor.numpy()
                return float(tensor)

            total_lc_num = convert_to_scalar(tensors["total_lc_num"])
            PT_GC_num = convert_to_scalar(tensors["PT_GC_num"])
            NAR_GC_num = convert_to_scalar(tensors["NAR_GC_num"])
            GFN_GC_num = convert_to_scalar(tensors["GFN_GC_num"])
            NDCG_Value = convert_to_scalar(tensors["NDCG_Value"])

            #pointwise_ctr_out_avg = np.mean(tensors["pointwise_ctr_out"])
            #listwise_ctr_out_avg = np.mean(tensors["listwise_ctr_out"])
            #generative_ctr_out_avg = np.mean(tensors["generative_ctr_out"])

            #print(f"total_lc_num: {total_lc_num}, type: {type(total_lc_num)}")
            #print(f"PT_GC_num: {PT_GC_num}, type: {type(PT_GC_num)}")
            #print(f"NAR_GC_num: {NAR_GC_num}, type: {type(NAR_GC_num)}")
            #print(f"GFN_GC_num: {GFN_GC_num}, type: {type(GFN_GC_num)}")
            #print(f"NDCG_Value: {NDCG_Value}, type: {type(NDCG_Value)}")
            #print(f"pointwise_ctr_out_avg: {pointwise_ctr_out_avg}, type: {type(pointwise_ctr_out_avg)}")
            #print(f"listwise_ctr_out_avg: {listwise_ctr_out_avg}, type: {type(listwise_ctr_out_avg)}")
            #print(f"generative_ctr_out_avg: {generative_ctr_out_avg}, type: {type(generative_ctr_out_avg)}")

            log0 = "train info: step {}, loss={:.4f}, pointwise_loss={:.4f}, listwise_generator_loss={:.4f}, generative_generator_loss={:.4f}, evaluator_loss={:.4f}, " \
                   "pointwise_ctr_out_avg={:.4f}, listwise_ctr_out_avg={:.4f}, generative_ctr_out_avg={:.4f}, ctr_out={:.4f}, avg_ctr_topk={:.4f}, avt_ctr_lastk={:.4f}, mean_o_distance={:.6f}, max_o_distance={:.6f}, mean_cosine_distance={:.6f}, max_cosine_distance={:.6f}, total_lc_num={:.6f}, PT_GC_num={:.6f}, NAR_GC_num={:.6f}, GFN_GC_num={:.6f}, NDCG_Value={:.6f}".format(
                    tensors["step"], tensors["loss"], tensors["pointwise_loss"], tensors["listwise_generator_loss"], tensors["generative_generator_loss"], tensors["evaluator_loss"],
                    tensors["pointwise_ctr_out"], tensors["listwise_ctr_out"], tensors["generative_ctr_out"], tensors["ctr_out"],
                    tensors["avg_ctr_topk"], tensors["avt_ctr_lastk"], tensors["mean_o_distance"], tensors["max_o_distance"], tensors["mean_cosine_distance"], tensors["max_cosine_distance"],
                    total_lc_num, PT_GC_num, NAR_GC_num, GFN_GC_num, NDCG_Value
            )
            return log0
        self.logging_hook = tf.train.LoggingTensorHook({"step": tf.train.get_global_step(),
                                                        "loss": self.loss,
                                                        "pointwise_loss": self.pointwise_generator_loss,
                                                        "listwise_generator_loss": self.listwise_generator_loss,
                                                        "generative_generator_loss":self.generative_generator_loss,
                                                        "evaluator_loss":self.evaluator_loss,
                                                        "pointwise_ctr_out":tf.reduce_mean(self.pointwise_generator_out),
                                                        "listwise_ctr_out":tf.reduce_mean(self.listwise_generator_out),
                                                        "generative_ctr_out":tf.reduce_mean(self.generative_generator_out),
                                                        "ctr_out": ctr_out,
                                                        "avg_ctr_topk" : tf.reduce_mean(self.topK_output,axis=[0,1,2]),
                                                        "avt_ctr_lastk" : tf.reduce_mean(self.lastK_output,axis=[0,1,2]),
                                                        "mean_o_distance" : tf.reduce_mean(self.o_final_mean_distance),
                                                        "max_o_distance" : tf.reduce_max(self.o_final_max_distance),
                                                        "mean_cosine_distance" : tf.reduce_mean(self.c_final_mean_distance),
                                                        "max_cosine_distance" : tf.reduce_max(self.c_final_max_distance),
                                                        "total_lc_num" :tf.reduce_mean(self.total_lc_num),
                                                        "PT_GC_num" : tf.reduce_mean(self.PT_GC_num),
                                                        "NAR_GC_num" : tf.reduce_mean(self.NAR_GC_num),
                                                        "GFN_GC_num" : tf.reduce_mean(self.GFN_GC_num),
                                                        "NDCG_Value" : self.ndcg_value,
                                                        },
                                                       every_n_iter=5,
                                                       formatter=format_log)

    def model_fn_estimator(self, features, labels, mode, params):
        self._create_weights()
        self._process_features(features)
        self._FGSM(features)
        #pointwise输出一个序列
        self._Pointwise_generator_model(features, labels, mode, params)
        #位置分布输出一个序列
        self._Listwise_generator_model(features, labels, mode, params)
        #GRN输出一个序列
        self._Generative_generator_model(features, labels, mode, params)
        
        generator_num = 3
        self._OCPM(features, labels, mode, params)
        self.generator_permuation_embeddings = tf.concat([self.pointwise_generator_embedding,self.listwise_generator_embedding,self.generative_generator_embedding],axis =1) # [bs, k, 5,7,8]
        self.generator_permuation_embeddings = tf.stop_gradient(self.generator_permuation_embeddings)

        self.pointwise_generator_out_expand = tf.expand_dims(self.pointwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.listwise_generator_out_expand = tf.expand_dims(self.listwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.generative_generator_out_expand = tf.expand_dims(self.generative_generator_out, axis=1)  # 形状变为 [bs, 1, 5]

        self.generator_permuation_predict_fea = tf.concat([self.pointwise_generator_out_expand,self.listwise_generator_out_expand,self.generative_generator_out_expand],axis =1) # [bs, k, 5]
        self.generator_permuation_predict_fea = tf.stop_gradient(self.generator_permuation_predict_fea)

        self.generator_permuation_index = tf.concat([self.pointwise_generator_permution_index_expand,self.listwise_generator_permution_index_expand,self.generative_generator_permution_index_expand],axis =1)
        self.generator_permuation_index = tf.stop_gradient(self.generator_permuation_index)

        self.generator_permuation_mask = tf.concat([self.pointwise_permuation_mask,self.listwise_permuation_mask,self.generative_permuation_mask],axis =1) # [bs, k, 5,1]
        self.generator_permuation_mask = tf.stop_gradient(self.generator_permuation_mask)


        self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.generator_permuation_embeddings, self.generator_permuation_predict_fea,generator_num,self.generator_permuation_mask)
        self.lastK_output = self._OCPM_TOP_K(features, labels, mode, params, self.last_k_permuation_embeddings, self.last_k_permuation_predict_fea, TOP_K,self.last_k_permuation_mask)
        self.randK_output = self._OCPM_TOP_K(features, labels, mode, params, self.random_k_permuation_embeddings, self.random_k_permuation_predict_fea, TOP_K,self.random_k_permuation_mask)
        self.beamK_output = self._OCPM_TOP_K(features, labels, mode, params, self.beam_search_k_permuation_embeddings, self.beam_search_k_permuation_predict_fea, TOP_K,self.beam_search_k_permuation_mask)
        self.allK_output = self._OCPM_TOP_K(features, labels, mode, params, self.full_permuation_embeddings, self.full_permuation_feat_predict, PERMUATION_SIZE,self.full_permuation_mask)

        #获取evaluator的top1输出
        list_sums = tf.reduce_sum(self.topK_output, axis=-1)  # shape: [bs, 4]
        max_list_indices = tf.argmax(list_sums, axis=-1)  # shape: [bs], 表示每个批次中最高的 list 的索引
        max_list_indices = tf.expand_dims(max_list_indices, axis=-1)
        self.max_list_scores = tf.batch_gather(self.topK_output, max_list_indices)  #[bs,1,5]
        self.max_list_scores = tf.squeeze(self.max_list_scores, axis=1)  # shape: [bs, 5]
        


        # 输出avg-ctr最高的序列
        self.avg_ctr_of_topK = tf.reduce_mean(self.topK_output,axis=2)
        max_ctr_index_in_topk = tf.nn.top_k(self.avg_ctr_of_topK,k=1).indices # batch * 1
        self.final_rerank_output_index = tf.batch_gather(self.generator_permuation_index,max_ctr_index_in_topk) # bs,1,5
        # self.max_ctr_index = tf.batch_gather(self.top_k_permutation_indicces,self.max_ctr_index_in_topk) # batch * 1
        # full_permuation_index = tf.reshape(features['full_permuation_index'], [PERMUATION_SIZE, POI_NUM])
        # full_permuation_index = tf.tile(tf.expand_dims(full_permuation_index, axis=0),[tf.shape(self.cate_feature_embeddings_for_permuation)[0], 1, 1]) # batch * 120 * 5

        #输出embedding欧式距离
        self.o_final_max_distance, self.o_final_mean_distance = self._compute_pairwise_distances(self.generator_permuation_embeddings)  # [bs]

        #余弦距离计算
        self.c_final_max_distance, self.c_final_mean_distance = self._compute_cosine_distances(self.generator_permuation_embeddings)  # [bs]

        #计算各个模型生成的序列之间的index的LC
        self.total_lc_num = self._calculate_total_inversions(self.generator_permuation_index,generator_num)
        self.PT_GC_num = self._inversions_with_new_list(self.generator_permuation_index, self.pointwise_generator_permution_index_expand, generator_num) # 输出每个批次的逆序对总数
        self.NAR_GC_num = self._inversions_with_new_list(self.generator_permuation_index, self.listwise_generator_permution_index_expand, generator_num) # 输出每个批次的逆序对总数
        self.GFN_GC_num = self._inversions_with_new_list(self.generator_permuation_index, self.generative_generator_permution_index_expand, generator_num) # 输出每个批次的逆序对总数
        

        #NDCG
        batch_size = tf.shape(self.out)[0]
        evaluator_out_argsort = tf.argsort(self.out, axis=1, direction='DESCENDING') #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
        ndcg_label =  tf.tile(tf.expand_dims(tf.range(POI_NUM - 1, -1, -1, dtype=tf.float32), axis=0), [batch_size, 1])
        self.ndcg_value = self._ndcg_at_k(evaluator_out_argsort, ndcg_label)


        if self.train:
            self._create_loss(labels)
            self._create_optimizer()
            self._create_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op, training_hooks=[self.logging_hook])
        else:
            if 'save_model' in list(params.keys()):
                outputs = {
                    "Q_network_output": tf.identity(self.Q_network_output, "Q_network_output"),
                    "out": tf.identity(self.out, "out"),
                    'output_index':tf.identity(self.final_rerank_output_index,"out_index")
                    }
            else:
                ctr_out = self.out
                # gmv
                outputs = {'out': self.out,
                           'mask': features['mask'],
                           'ctr_out': ctr_out,
                           'ctr_label': features['ctr_label'],
                           'q_out': self.Q_network_output,
                           'cxr_feature': features['dense_feature'],
                           "ctr_topk": self.topK_output,
                           "ctr_lastk": self.lastK_output,
                           "ctr_randk": self.randK_output,
                           "ctr_all": self.allK_output,
                           "ctr_beamk": self.beamK_output,
                           'out_index':self.final_rerank_output_index
                         }
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                      tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)


    def tf_print(self, var, varStr='null'):
        return tf.Print(var, [var], message=varStr, summarize=100)

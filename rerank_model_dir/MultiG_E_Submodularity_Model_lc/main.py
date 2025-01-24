# -*- coding: utf-8 -*-
from config import *
from model import *
from sklearn import metrics
import numpy as np

def create_estimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        tf_random_seed=RANDOM_SEED,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        model_dir=MODEL_SAVE_PATH,
        keep_checkpoint_max=2,
        log_step_count_steps=1000,
        session_config=session_config)
    nn_model = DNN()
    estimator = tf.estimator.Estimator(model_fn=nn_model.model_fn_estimator, config=config)
    return estimator, nn_model


def save_model_pb_with_estimator(estimator, params, export_dir_base):
    estimator._params['save_model'] = params['save_model']

    def _serving_input_receiver_fn():
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature
        receiver_tensors = {
            # ctr cvr gmv预估值 && bid
            'screen_predict_feature': tf.placeholder(tf.float32, [None, POI_NUM, FEATURE_CXR_NUM],
                                                name='screen_predict_feature'),
            # dense 特征 (价格，评分)
            'screen_dense_feature': tf.placeholder(tf.float32, [None, POI_NUM, FEATURE_DENSE_NUM],
                                                name='screen_dense_feature'),
            # 离散特征(品类)
            'screen_cate_feature': tf.placeholder(tf.int64, [None, POI_NUM, FEATURE_CATE_NUM],
                                                name='screen_cate_feature'),
            # 环境特征（是否有铂金）
            'dense_feature': tf.placeholder(tf.float32, [None, DENSE_FEAT_NUM],
                                                name='dense_feature')
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)

    export_dir = estimator.export_saved_model(export_dir_base=export_dir_base,
                                              serving_input_receiver_fn=_serving_input_receiver_fn)
    estimator._params.pop('save_model')
    return export_dir.decode()


#def dcg_at_k(scores, k):
#    """
#    计算 DCG 值 (Discounted Cumulative Gain)
#    scores: 实际结果的评分
#    k: 截止位置
#    """
#    scores = np.asfarray(scores)[:k]  # 取前 k 个结果
#    if scores.size == 0:
#        return 0.0
#    # 使用公式：sum((2^relevance - 1) / log2(rank + 1))
#    return np.sum((2**scores - 1) / np.log2(np.arange(1, scores.size + 1) + 1))

#def ndcg_at_k(actual, predicted, k):
#    """
#    计算 NDCG 值 (Normalized Discounted Cumulative Gain)
#    actual: 实际的评分或标签
#    predicted: 预测结果的排序索引
#    k: 截止位置
#    """
#    ideal_sorted = sorted(actual, reverse=True)  # 理想的排序（从高到低）
#    dcg = dcg_at_k([actual[i] for i in predicted], k)  # 使用预测排序计算 DCG
#    idcg = dcg_at_k(ideal_sorted, k)  # 使用理想排序计算 IDCG
#    return dcg / idcg if idcg > 0 else 0.0

def dcg_at_k(scores, k):
    """
    计算给定排序情况下的 DCG 值
    :param scores: 真实排序后的标签值
    :param k: 计算前 k 个位置的 DCG
    :return: DCG 值
    """
    scores = np.asfarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2**scores - 1) / np.log2(np.arange(1, scores.size + 1) + 1))

def ndcg_at_k(prediction, label, k):
    """
    计算 NDCG@k
    :param prediction: 预测分数，形状为[batch_size, k]
    :param label: 理想排序后的标签，形状为[batch_size, k]
    :param k: 计算前 k 个位置的 NDCG
    :return: NDCG@k 值
    """
    batch_size = prediction.shape[0]
    ndcg_scores = np.zeros(batch_size)

    for i in range(batch_size):
        pred_scores = prediction[i]
        true_labels = label[i]

        # 按照预测的得分排序
        pred_ranks = np.argsort(-pred_scores)  # 降序排列

        # 计算 DCG
        dcg = dcg_at_k(true_labels[pred_ranks], k)

        # 计算 IDCG (理想的排序）
        idcg = dcg_at_k(true_labels, k)

        # 计算 NDCG
        if idcg > 0:
            ndcg_scores[i] = dcg / idcg
        else:
            ndcg_scores[i] = 0.0

    return np.mean(ndcg_scores)

def calculate_result(result_generator):

    y_ctr, pred_ctr, ctr = [], [], []
    topk_ctr = []
    beamK_ctr = []
    all_topk_ctr = []
    randomk_ctr = []
    best_topk_ctr = []
    best_beamK_ctr = []
    best_all_topk_ctr = []
    best_randomk_ctr = []
    for result in result_generator:
        cxr_feature = result['cxr_feature']
        mask = result['mask']
        #out_index = result['out_index']
        #print(out_index)
        ctr_out = result['ctr_out']
        ndcg_label = np.arange(POI_NUM - 1, -1, -1, dtype=np.float32)
        ndcg_label = np.expand_dims(ndcg_label, axis=0)
        ndcg_label = np.tile(ndcg_label, (1024, 1))
        ndcg_value = ndcg_at_k(ctr_out,ndcg_label,k=POI_NUM)
        
        #print("ctr_out:",ctr_out)
        #print("ctr_out.shape:",ctr_out.shape)
        #before = result['ctr_out'].reshape(-1).tolist()
        #print("before:",before)
        # ctr_label
        idx = np.where(mask.reshape(-1) == 1)
        listctr = result['ctr_out'].reshape(-1)[idx].tolist()
        #print("listctr:",listctr)
        #print("listctr:",len(listctr))
        y_ctr += result['ctr_label'].reshape(-1)[idx].tolist()
        pred_ctr += result['ctr_out'].reshape(-1)[idx].tolist()
        ctr += cxr_feature[:, :, 0].reshape(-1)[idx].tolist()

        tmp_ctr_topk = np.mean(result['ctr_topk'],axis=2).reshape(-1)
        tmp_ctr_randk = np.mean(result['ctr_randk'],axis=2).reshape(-1)
        tmp_ctr_all = np.mean(result['ctr_all'],axis=2).reshape(-1)
        tmp_ctr_all_topk = tmp_ctr_all[np.argsort(tmp_ctr_all * -1)].reshape(-1)
        tmp_ctr_beamk = np.mean(result['ctr_beamk'],axis=2).reshape(-1)

        topk_ctr += tmp_ctr_topk.tolist()
        #print("tmp_ctr_topk.tolist():",tmp_ctr_topk.tolist())
        beamK_ctr += tmp_ctr_beamk.tolist()
        randomk_ctr += tmp_ctr_randk.tolist()
        all_topk_ctr += tmp_ctr_all_topk.tolist()

        best_topk_ctr.append(np.max(tmp_ctr_topk))
        #print("np.max(tmp_ctr_topk)",np.max(tmp_ctr_topk))
        best_beamK_ctr.append(np.max(tmp_ctr_beamk))
        best_all_topk_ctr.append(np.max(tmp_ctr_all_topk))
        best_randomk_ctr.append(np.max(tmp_ctr_randk))

    #print("best_topk_ctr:",best_topk_ctr)
    #print("topk_ctr:",topk_ctr)
    #print("pred_ctr:",pred_ctr)
    #print("shape best_topk_ctr:",len(best_topk_ctr))
    #print("shape topk_ctr:",len(topk_ctr))
    #print("shape pred_ctr:",len(pred_ctr))
   
    #evaluator_out_argsort = np.argsort(pred_ctr)[::-1]
    #ndcg_value = ndcg_at_k(ndcg_label, evaluator_out_argsort, POI_NUM)
    ndcg_value = np.mean(ndcg_value)
     

    ctr_auc, ctr_ndcg, ctr_auc_jp, ctr_cb, ctr_cb_jp = metrics.roc_auc_score(y_ctr, pred_ctr), ndcg_value, metrics.roc_auc_score(y_ctr, ctr), np.sum(pred_ctr) / np.sum(y_ctr),  np.sum(ctr) / np.sum(y_ctr)
    print("ctr_auc:{}, ctr_ndcg:{}, ctr_auc_jp:{}, ctr_cb:{}, ctr_cb_jp:{}".format(ctr_auc, ctr_ndcg, ctr_auc_jp, ctr_cb, ctr_cb_jp))
    print("topk_ctr:{}, beamK_ctr:{}, randomk_ctr:{}, all_topk_ctr:{}".format(np.mean(topk_ctr), np.mean(beamK_ctr), np.mean(randomk_ctr), np.mean(all_topk_ctr)))
    print("best_topk_ctr:{}, best_beamK_ctr:{}, best_randomk_ctr:{}, best_all_topk_ctr:{}".format(np.mean(best_topk_ctr), np.mean(best_beamK_ctr), np.mean(best_randomk_ctr), np.mean(best_all_topk_ctr)))

if __name__ == '__main__':

    estimator, nn_model = create_estimator()

    with tick_tock("DATA_INPUT") as _:
        valid_input_fn = input_fn_maker(VALID_FILE, False, batch_size=1024, epoch=1)
        test_input_fn = input_fn_maker(TEST_FILE, False, batch_size=1024, epoch=1)

    if TRAIN_MODE == 1:
        for i in range(EPOCH):
            for idx, data in enumerate(TRAIN_FILE):
                with tick_tock("DATA_INPUT") as _:
                    train_input_fn = input_fn_maker([data], True, batch_size=BATCH_SIZE, epoch=1)
                with tick_tock("TRAIN") as _:
                    estimator.train(train_input_fn)
                if MODEL_SAVE_PB_EPOCH_ON:
                    export_dir = save_model_pb_with_estimator(estimator, params={'save_model': 'listwise'},
                                                              export_dir_base=MODEL_SAVE_PB_EPOCH_PATH)
                    ep_insert_index = i * len(TRAIN_FILE) + idx
                    target_dir = export_dir + "/../ep" + str(ep_insert_index)
                    while os.path.exists(target_dir):
                        target_dir = export_dir + "/../ep" + str(ep_insert_index)
                    shutil.move(export_dir, target_dir)
                    print(time.strftime("%m-%d %H:%M:%S ",
                                        time.localtime(time.time())) + "export model PB: " + target_dir)
                #with tick_tock("PREDICT") as _:
                    #result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                    #calculate_result(result_generator)

    elif TRAIN_MODE == 2:
        with tick_tock("PREDICT") as _:
            result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
            calculate_result(result_generator)

    elif TRAIN_MODE == 3:
        for i in range(EPOCH):
            for idx, data in enumerate(TRAIN_FILE):
                with tick_tock("DATA_INPUT") as _:
                    train_input_fn = input_fn_maker([data], True, batch_size=BATCH_SIZE, epoch=1)
                with tick_tock("TRAIN") as _:
                    estimator.train(train_input_fn)
                with tick_tock("PREDICT") as _:
                    result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                    print("valid_data")
                    calculate_result(result_generator)
                    #result_generator = estimator.predict(input_fn=test_input_fn, yield_single_examples=False)
                    print("train_data")
                    #calculate_result(result_generator)
                    # save pb

    
    elif TRAIN_MODE == 4:
        export_dir = save_model_pb_with_estimator(estimator, params={'save_model': 'listwise'},
                                                    export_dir_base=MODEL_SAVE_PB_EPOCH_PATH)
        ep_insert_index = 0
        target_dir = export_dir + "/../ep" + str(ep_insert_index)
        while os.path.exists(target_dir):
            target_dir = export_dir + "/../ep" + str(ep_insert_index)
        shutil.move(export_dir, target_dir)
        print(time.strftime("%m-%d %H:%M:%S ",
                            time.localtime(time.time())) + "export model PB: " + target_dir)
    

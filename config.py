import datetime
import os
import threading


class Config(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if not Config._init_flag:
            Config._init_flag = True
            self.base_path = ''
            self._init_train_config()


    def _init_train_config(self):
        self.label_list = []
        self.use_gpu = True
        self.device = "cuda"
        self.model = 'bert_crf'  # 可选['bert_bilstm_crf','bert_crf','FRLN'], 与ckpts中的文件夹名相对应

        # 输入数据集、日志、输出目录
        self.train_file = os.path.join(self.base_path, 'data/train.txt')
        self.test_file = os.path.join(self.base_path, 'data/test.txt')
        self.dev_file = os.path.join(self.base_path, 'data/test.txt')
        self.log_path = os.path.join(self.base_path, 'logs')
        self.output_path = os.path.join(self.base_path, 'output', self.model)
        self.trained_model_path = os.path.join(self.base_path, 'ckpts', self.model, 'checkpoints\checkpoints_epoch_19\checkpoints')
        self.model_name_or_path = self.trained_model_path # 如果想测试训练过的模型，改为 self.trained_model_path

        # 以下是模型训练参数
        self.do_train = True
        self.do_eval = False
        self.need_birnn = False
        self.do_lower_case = True
        self.rnn_dim = 128
        self.max_seq_length = 128
        self.batch_size = 16
        self.num_train_epochs = 10
        self.ckpts_epoch = 5
        self.gradient_accumulation_steps = 1
        self.learning_rate = 3e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.logging_steps = 50
        self.remove_O = False
        self.freeze_epoch = 5

    def _init_train_config_for_CLUENER2020(self):
        self.label_list = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        self.use_gpu = True
        self.device = "cuda"
        self.checkpoints = True  # 使用预训练模型时设置为False
        self.model = 'bert_cnn_another_dataset'  # 可选['bert_bilstm_crf','bilstm_crf','bilstm','crf','hmm']

        # 输入数据集、日志、输出目录
        self.train_file = os.path.join(self.base_path, 'data/cluener_public/train.json')
        self.test_file = os.path.join(self.base_path, 'data/cluener_public/test.json')
        self.dev_file = os.path.join(self.base_path, 'data/cluener_public/dev.json')
        self.log_path = os.path.join(self.base_path, 'logs')
        # self.output_path = os.path.join(self.base_path, 'output', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        self.output_path = os.path.join(self.base_path, 'output', self.model)
        self.trained_model_path = os.path.join(self.base_path, 'ckpts', self.model)
        self.model_name_or_path = 'bert-base-chinese'
        # os.path.join(self.base_path, 'ckpts', 'bert-base-chinese') if not self.checkpoints \
        #     else self.trained_model_path

        # 以下是模型训练参数
        self.do_train = True
        self.do_eval = False
        self.need_birnn = True
        self.do_lower_case = True
        self.rnn_dim = 128
        self.max_seq_length = 128
        self.batch_size = 16
        self.num_train_epochs = 20
        self.ckpts_epoch = 5
        self.gradient_accumulation_steps = 1
        self.learning_rate = 3e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.logging_steps = 50
        self.remove_O = False
        self.freeze_epoch = 5
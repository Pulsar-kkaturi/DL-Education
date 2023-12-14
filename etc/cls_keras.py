from typing import Any
import numpy as np
import pandas as pd
import os, matplotlib, random, math, shutil, time
from matplotlib import pyplot as plt
import skimage, h5py, logging, argparse
from tqdm import tqdm
from skimage import io as skio
from skimage import transform as skit

import tensorflow as tf
from tensorflow import keras as tfk

# 텐서 모델
class TensorSequence(tf.keras.utils.Sequence):
  def __init__(self, paths, batch_size, y_list, train=True, shuffle=True):
        self.paths = paths
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle
        self.y_list = y_list
        if self.shuffle:
            random.shuffle(self.paths)
        else:
            pass

  def __len__(self):
      return math.ceil(len(self.paths) / self.batch_size)

  def __getitem__(self, idx):
      low = idx * self.batch_size
      # Cap upper bound at array length; the last batch may be smaller
      # if the total number of items is not a multiple of batch size.
      high = min(low + self.batch_size, len(self.paths))
      batch_paths = self.paths[low:high]
      if self.shuffle:
          random.shuffle(batch_paths)
      else:
          pass
      batch_x, batch_y = [], []
      for path in batch_paths:
          image = skio.imread(path)
          image = (image-np.min(image))/(np.max(image)-np.min(image))
          batch_x.append(image)
          if self.train:
              label = [0]*len(self.y_list)
              label[self.y_list.index(os.path.basename(path).split('_')[-2])] = 1
              batch_y.append(label)
          else:
              pass
      if self.train:
          return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)
      else:
          return np.array(batch_x, dtype=np.float32)


class CLS_Keras():
    def __init__(self, params) -> None:
        self.params = params

    def preprocessing(self, input_path, output_path):
        # Log Setting
        os.makedirs(output_path, exist_ok=True)
        log_name = 'Prep_{}_{}'.format(os.path.basename(output_path), time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))
        logger = self._logger_setting(output_path, log_name)
        logger.info('### Pre-processing ###')

        # 폴더 확인
        logger.info(f' - 데이터셋 폴더 내부 확인(train/test) = {os.listdir(input_path)}')
        train_data_path = os.path.join(input_path, 'train')
        test_data_path = os.path.join(input_path, 'test')
        logger.info(f' - 학습 데이터셋 폴더 내부 확인 = {os.listdir(train_data_path)}')
        logger.info(f' - 테스트 데이터셋 폴더 내부 확인 = {os.listdir(test_data_path)}')

        # 레이블 확인
        label_list = sorted(os.listdir(train_data_path))
        logger.info(f' - 레이블 목록 = {label_list}')

        # Path loading
        train_paths, val_paths = [], []
        for k in sorted(os.listdir(train_data_path)):
            fol_path = os.path.join(train_data_path, k)
            num = 0
            for i in sorted(os.listdir(fol_path)):
                file_path = os.path.join(fol_path, i)
                if num < self.params['val_num']:
                    val_paths.append(file_path)
                else:
                    if self.params['train_num']:
                        if num < self.params['val_num'] + self.params['train_num']:
                            train_paths.append(file_path)
                        else:
                            pass
                    else:
                        train_paths.append(file_path)
                num += 1
        logger.info(f' - 학습 데이터셋(Train) 숫자 = {len(train_paths)}')
        logger.info(f' - 검증 데이터셋(Validation) 숫자 = {len(val_paths)}')

        test_paths = []
        for k in sorted(os.listdir(test_data_path)):
            fol_path = os.path.join(test_data_path, k)
            for i in sorted(os.listdir(fol_path)):
                file_path = os.path.join(fol_path, i)
                test_paths.append(file_path)
        logger.info(f' - 테스트 데이터셋(Test) 숫자 = {len(test_paths)}')

        # Save (train)
        logger.info(' - train dataset save')
        self._prep_csv_making('train', output_path, train_paths, self.params['input_size'])
        logger.info(' - valdiation dataset save')
        self._prep_csv_making('val', output_path, val_paths, self.params['input_size'])
        logger.info(' - test dataset save')
        self._prep_csv_making('test', output_path, test_paths, self.params['input_size'])

    def train_cls(self, input_path, output_path):
        output_path = output_path + f'_{self.params["model_name"]}'
        # Log Setting
        os.makedirs(output_path, exist_ok=True)
        log_name = 'Train_{}_{}'.format(os.path.basename(output_path), time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))
        log_path = os.path.join(output_path, 'log')
        os.makedirs(log_path, exist_ok=True)
        logger = self._logger_setting(log_path, log_name)
        logger.info('### Train CLS ###')

        # Tensor loading
        train_paths, val_paths, label_list = [], [], []
        for i_ in sorted(os.listdir(os.path.join(input_path, 'train'))):
            file_path = os.path.join(input_path, 'train', i_)
            file_label = i_.split('_')[-2]
            train_paths.append(file_path)
            label_list.append(file_label)
        for i_ in sorted(os.listdir(os.path.join(input_path, 'val'))):
            file_path = os.path.join(input_path, 'val', i_)
            val_paths.append(file_path)
        label_list = list(sorted(set(label_list)))
        logger.info(f' - Label = {label_list}')
        logger.info(f' - Train Number = {len(train_paths)}')
        logger.info(f' - Validation Number = {len(val_paths)}')

        # GPU Setting
        tf.random.set_seed(self.params['seed'])
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Model Build
            if self.params['model_name'] == 'vgg16':
                conv_base = tfk.applications.VGG16(input_shape=self.params['input_size'], include_top=False, pooling='avg', weights=None)
            elif self.params['model_name'] == 'vgg19':
                conv_base = tfk.applications.VGG19(input_shape=self.params['input_size'], include_top=False, pooling='avg', weights=None)
            elif self.params['model_name'] == 'resnet50':
                conv_base = tfk.applications.ResNet50V2(input_shape=self.params['input_size'], include_top=False, pooling='avg', weights=None)
            elif self.params['model_name'] == 'resnet152':
                conv_base = tfk.applications.ResNet152V2(input_shape=self.params['input_size'], include_top=False, pooling='avg', weights=None)
            else:
                print('!ERROR! please insert correct model name! (vgg16, vgg19, resnet50)')
            input = conv_base.input
            dense = tfk.layers.Dropout(self.params['drop_rate'])(conv_base.output)
            for d in self.params['dens_filters']:
                dense = tfk.layers.Dense(d, activation=None)(dense)
                dense = tfk.layers.BatchNormalization()(dense)
                dense = tfk.layers.Activation('relu')(dense)
                dense = tfk.layers.Dropout(self.params['drop_rate'])(dense)
            output = tfk.layers.Dense(len(label_list), activation='softmax')(dense)
            model = tfk.Model(input, output)
            model.summary(print_fn=logger.info)

            # Model Compile
            model.compile(loss=tfk.losses.CategoricalCrossentropy(), 
                          optimizer=self.optim_build(self.params['optimizer'], self.params['learning_rate']),
                          metrics=['accuracy'])
            
            callback_list = [tfk.callbacks.EarlyStopping(monitor='val_loss', patience=int(0.3*self.params['epoch'])),
                             tfk.callbacks.ModelCheckpoint(filepath=os.path.join(output_path, 'model.h5'),
                                                           monitor='val_accuracy', save_best_only=True),
                             tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(0.1*self.params['epoch']))]
            
            # Model history
            history = model.fit(TensorSequence(train_paths, self.params['batch_size'], label_list),
                                epochs=self.params['epoch'], callbacks=callback_list,
                                validation_data=TensorSequence(val_paths, self.params['batch_size'], label_list))
            doc_path = os.path.join(output_path, 'doc')
            os.makedirs(doc_path, exist_ok=True)

            # History Save
            df_history = pd.DataFrame(history.history)
            logger.info('<Train History>')
            logger.info(df_history)
            df_history.to_csv(os.path.join(doc_path, log_name + '_history.csv'), index=False)
            self.loss_met_graph(doc_path, log_name, self.params['model_name'])
            logger.info(f' - Network Model {self.params["model_name"]} Train is Finished!')

            # Config Save
            config_dic = {'Key': [], 'Value': []}
            for k in self.params.keys():
                config_dic['Key'].append(k)
                config_dic['Value'].append(self.params[k])
            df_config = pd.DataFrame(config_dic)
            df_config.to_csv(os.path.join(doc_path, log_name + '_config.csv'), index=False)


    def _gpu_check(self):
        print('\n### GPU Check ###')
        print(" - 현재 tensorflow 버전은 무엇인가? : %s" %(tf.__version__))
        device_list = tf.config.list_physical_devices('GPU')
        print(" - 사용 가능한 GPU가 존재하는가? (True or False): ", bool(device_list))
        if device_list:
            print(" - 현재 사용 가능한 GPU의 수는 {}개 입니다.".format(len(device_list)))
            print(" - GPU 목록은 아래와 같습니다.")
            for device in device_list:
                print(device)
        else:
            print(" - 사용 가능한 GPU가 존재하지 않습니다. 혹은 GPU를 Tensorflow가 찾지 못하고 있습니다.")
    
    def _norm_array(self, array):
        norm_arr = (array-np.min(array))/(np.max(array)-np.min(array))
        return (255*norm_arr).astype(np.uint8)


    def _logger_setting(self, path, name):
        """ Log Setting
        "Logger Environment setting"
        Args:
            path: log file save path (string)
            name: logger title (string)
        Return: logger
        """
        log_file_path = os.path.join(path, 'log_%s.txt' %(name))

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger
    
    def _prep_csv_making(self, set_type, output_path, data_paths, prep_size):
        data_dic = {'name': [], 'class': [], 'org_name': [], 'org_shape': [], 'prep_shape': []}
        error_dic = {'file': [], 'message': []}
        data_out_path = os.path.join(output_path, set_type)
        os.makedirs(data_out_path, exist_ok=True)
        pbar = tqdm(total=len(data_paths))
        for i, i_ in enumerate(data_paths):
            try:
                # Image Loading
                org_array = skio.imread(i_)
                new_array = skit.resize(org_array, prep_size)
                data_label = i_.split('/')[-2]
                data_name = f'img_{str(1001+i)[1:]}_{data_label}_{set_type}' 
                out_file_path = os.path.join(data_out_path, f'{data_name}.png')
                # Image Saving
                skio.imsave(out_file_path, self._norm_array(new_array))
                # info save
                data_dic['name'].append(data_name)
                data_dic['class'].append(data_label)
                data_dic['org_name'].append(os.path.basename(i_))
                data_dic['org_shape'].append(org_array.shape)
                data_dic['prep_shape'].append(new_array.shape)
            except Exception as err:
                error_dic['file'].append(i_)
                error_dic['message'].append(err)
            pbar.set_description(f'{data_name}: {os.path.basename(i_)}, {data_label}')
            pbar.update(1)
        pbar.close()
        df_data = pd.DataFrame(data_dic)
        df_data.to_csv(os.path.join(output_path, f'{set_type}_info_dic.csv'), index=False)
        df_error = pd.DataFrame(error_dic)
        df_error.to_csv(os.path.join(output_path, f'{set_type}_error_dic.csv'), index=False)

    def optim_build(self, mode, lr=0.01):
        if mode == 'adam':
            return tfk.optimizers.Adam(learning_rate=lr)
        elif mode == 'adamw':
            return tfk.optimizers.AdamW(learning_rate=lr)
        elif mode == 'rmsprop':
            return tfk.optimizers.RMSprop(learning_rate=lr)
        
    def loss_met_graph(self, history_save_path, log_name, model_name):
        df_history = pd.read_csv(os.path.join(history_save_path, log_name+'_history.csv'))
        loss = list(df_history['loss'])
        val_loss = list(df_history['val_loss'])
        epochs = range(len(loss))
        lr = list(df_history['lr'])
        # metric graph
        tr_met = list(df_history['accuracy'])
        val_met = list(df_history['val_accuracy'])
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, tr_met, 'b', label='Training {} = {}'.format('accuracy', round(np.max(tr_met), 3)))
        plt.plot(epochs, val_met, 'r', label='Validation {} = {}'.format('accuracy', round(np.max(val_met), 3)))
        plt.title('{} {} (Total Epoch = {})'.format(model_name, 'accuracy', len(tr_met)), fontsize=15, y=1.02)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.tick_params(axis='both', which='major', length=10, width=1, direction='in')
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(history_save_path, f'{log_name}_accuracy_fig.png'))
        # Loss graph
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, loss, 'b', label='Training loss = {}'.format(round(np.min(loss), 3)))
        plt.plot(epochs, val_loss, 'r', label='Validation loss = {}'.format(round(np.min(val_loss), 3)))
        plt.title('{} Loss (Total Epoch = {})'.format(model_name, len(loss)), fontsize=15, y=1.02)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.tick_params(axis='both', which='major', length=10, width=1, direction='in')
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(history_save_path, log_name+'_loss_fig.png'))
        # Learning rate graph
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, lr, 'g', label='Learning Rate = {}'.format(np.around(np.min(lr), decimals=3)))
        plt.title('{} Learning Rate (Total Epoch = {})'.format(model_name, len(lr)), fontsize=15, y=1.02)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.tick_params(axis='both', which='major', length=10, width=1, direction='in')
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(history_save_path, log_name + '_learning_fig.png'))



    
def config_dic(model_name):
    return {'seed': 999, 
            'model_name': model_name, # 신경망 모델 이름: vgg16, vgg19, resnet50
            'input_size': (256, 256, 3), # 입력 영상 크기: (32,32,3), (64, 64, 3), (128, 128, 3), (256, 256, 3)
            'class_num': 2, # 분류할 클래스 수
            'train_num': None, # 클래스 당 Train number
            'val_num': 10, # 클래스 당 validation number
            'batch_size': 32, # Batch Size: 4, 8, 16
            'epoch': 200,
            'drop_rate': 0.5,
            'dens_filters': [1000, 500, 100],
            'optimizer': 'adam', # adam, adamw, rmsprop
            'learning_rate': 1e-3, 
            }

if __name__ == '__main__': 
    # Parameter Setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/home/student/Datasets/jhjeong/VUNO', help='Base PATH')
    parser.add_argument('-g', '--gpu', type=str, default='0,1', help='GPU Number')
    parser.add_argument('-t', '--title', type=str, default='fruit_veg', help='Dataset Title')
    parser.add_argument('-p', '--process', type=str, default='train', help='Process Mode = prep,train,test')
    parser.add_argument('--model', type=str, default='vgg16', help='Model Name (vgg16, vgg19, resnet50)')
    args = parser.parse_args()
    print("### args = ", args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    configs = config_dic(args.model)
    print('\n### Configureation = ', configs)
    run_cls = CLS_Keras(configs)
    
    run_cls._gpu_check()
    for p in args.process.split(','):
        if p == 'prep':
            run_cls.preprocessing(os.path.join(args.base_path, 'raw_data', args.title), os.path.join(args.base_path, 'prep_data', args.title))
        elif p == 'train':
            run_cls.train_cls(os.path.join(args.base_path, 'prep_data', args.title), os.path.join(args.base_path, 'post_data', args.title))
        else:
            pass
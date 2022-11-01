
# This work was inspired by Xue et al. http://arxiv.org/abatch_size/1706.01805 as well as the excellent "Deep Learning for coders" tought by Jeremy Howard and Rachel Thomas on http://course.fast.ai/

from keras.engine import Model
from keras.layers import Lambda
from keras.layers import Dropout, LeakyReLU, Input, Activation, BatchNormalization, Concatenate, multiply, Flatten
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l1, l2
from keras.losses import mae
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from functions import *
from metrics import *
import imageio 
os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config = config)

K.set_image_data_format('channels_last')
channel_axis = -1

#########################
# setting
norm = ''#sys.argv[1]
D_loss = ''#sys.argv[2]
G_loss = ''#sys.argv[3]
opt = ''#sys.argv[4]
batch_size = 1#int(sys.argv[5])

path = '/scratch/ra130/Projects/DeepGlobe/Data2'
train_path = path + '/Data/train/'
test_path = path + '/Data/test/'
statistics_path = path + '/Data/'

train_X_path = train_path + '/**.tiff'#'/**_sat.jpg'
train_y_path = train_path + '/**.png'#'/**_mask.png'

test_X_path = test_path + '/**.tiff'#'/**_sat.jpg'
test_y_path = test_path + '/**.png'#'/**_mask.png'

results_path = path +'/Results'
os.mkdir(results_path) if not os.path.exists(results_path) else None

subpath = results_path + '/seGANoss_k'
os.mkdir(subpath) if not os.path.exists(subpath) else None

models_path = subpath + '/models'
os.mkdir(models_path) if not os.path.exists(models_path) else None

images_path = subpath + '/images'
os.mkdir(images_path) if not os.path.exists(images_path) else None

train_X = load_data(train_X_path)
train_y = load_data(train_y_path)

test_X = load_data(test_X_path)
test_y = load_data(test_y_path)

train_data = list(zip(sorted(train_X), sorted(train_y)))
train_batch = get_miniBatch(train_data,norm, statistics_path)


test_data = list(zip(sorted(test_X), sorted(test_y)))
test_batch = get_miniBatch(test_data,norm, statistics_path)

#####################################################################################

def dropout(x, p):
    return Dropout(p)(x)

def bnorm(x):
    return BatchNormalization()(x)

def relu(x):
    return Activation('relu')(x)

def conv_l1(x, nb_filters, kernel, stride=(1, 1)):
    return Convolution2D(nb_filters, kernel, padding='same',
                         kernel_initializer='he_uniform',
                         kernel_regularizer=l1(0.01), strides=(stride, stride))(x)

def convl1_lrelu(x, nb_filters, kernel, stride):
    x = conv_l1(x, nb_filters, kernel, stride)
    return LeakyReLU()(x)

def convl1_bn_lrelu(x, nb_filters, kernel, stride):
    x = conv_l1(x, nb_filters, kernel, stride)
    x = bnorm(x)
    return LeakyReLU()(x)

def shared_convl1_lrelu(shape, nb_filters, kernel, stride=(1, 1), **kwargs):
    # i = Input(shape)
    c = Convolution2D(nb_filters, kernel, padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l1(0.01), strides=(stride, stride), input_shape=shape)
    l = LeakyReLU()
    return Sequential([c, l], **kwargs)

def shared_convl1_bn_lrelu(shape, nb_filters, kernel, stride=(1, 1), **kwargs):
    c = Convolution2D(nb_filters, kernel, padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l1(0.01), strides=(stride, stride), input_shape=shape)
    b = BatchNormalization()
    l = LeakyReLU()
    return Sequential([c, b, l], **kwargs)

def upsampl_block(x, nb_filters, kernel, stride, size):
    x = UpSampling2D(size=size)(x)
    x = conv_l1(x, nb_filters, kernel, stride)
    x = bnorm(x)
    return relu(x)

def upsampl_conv(x, nb_filters, kernel, stride, size):
    x = UpSampling2D(size=size)(x)
    return conv_l1(x, nb_filters, kernel, stride)

def upsampl_softmax(x, nb_filters, kernel, stride, size):
    x = UpSampling2D(size=size)(x)
    x = conv_l1(x, nb_filters, kernel, stride)
    x = Lambda(hidim_softmax, name='softmax')(x)
    x = Lambda(lambda x: K.max(x, axis=-1, keepdims=True), name='MaxProject')(x)
    return x

def level_block(previous_block, nb_filters, depth, filter_inc_rate, p):
    print('Current level block depth {}'.format(depth))
    # curr_block = previous_block
    curr_block = convl1_bn_lrelu(previous_block, nb_filters, 4, 2)
    print('Shape prev {}, shape curr {} and depth {} before recursion'.format(previous_block.shape, curr_block.shape,
                                                                              depth))
    curr_block = dropout(curr_block, p) if p else curr_block
    if depth > 0:  # Call next recursion level
        curr_block = level_block(curr_block, int(filter_inc_rate * nb_filters), depth - 1, filter_inc_rate, p)
    print('Shape prev {}, shape curr {} and depth {} after recursion'.format(previous_block.shape, curr_block.shape,
                                                                             depth))
    curr_block = upsampl_block(curr_block, nb_filters, 3, 1, 2)
    print('Shape prev {}, shape curr {} and depth {} after upsampling'.format(previous_block.shape, curr_block.shape,
                                                                              depth))
    curr_block = Concatenate(axis=3)([curr_block, previous_block])
    print('Shape curr {} and depth {} before return'.format(curr_block.shape, depth))
    return curr_block

def hidim_softmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply hidim_softmax to a tensor that is 1D')

class SeGAN(object):
    def __init__(self, imgs, gt, start_filters=64, filter_inc_rate=2, out_ch=1, depth=2,
                 optimizer=RMSprop(2e-5), loss=mae, softmax=False, crop=False):
    
        self.imgs = imgs  # Images
        self.gt = gt  # images cropped with ground-truth mask
        self.n = imgs.shape[0]  # number of images
        self.source_shape = imgs.shape[1:] 
        self.target_shape = gt.shape[1:]  
        self.optimizer = optimizer
        self.loss = loss
        self.softmax = softmax
        self.crop = crop
        
        self.disc_loss, self.gene_loss, self.disc_loss_real, self.disc_loss_fake = [], [], [], []
        self.gen_iterations = 0  # Counter for generator training iterations
        
        self.generator = self.build_generator(start_filters, filter_inc_rate, out_ch, depth)
        self.discriminator = self.build_discriminator()
        print('G:', self.generator.summary())
        print('D:', self.discriminator.summary())
        
        X = Input(shape=self.source_shape)
        y_real = Input(shape=self.target_shape)
        y_fake = self.generator(X)
        
        v_real = self.discriminator([X,y_real])
        v_fake = self.discriminator([X,y_fake])
        self.combined_model = Model(input=[X, y_real], outputs = [v_fake, y_fake])
        
        self.discriminator.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        self.combined_model.compile(loss=['mae',dice_coef_loss], loss_weights=[1, 10], optimizer=optimizer)
                              
    def build_generator(self, start_filters=64, filter_inc_rate=2, out_ch=1, depth=2):
      
        X = Input(shape=self.source_shape)
        first_block = convl1_lrelu(X, start_filters, 4, 2)
        middle_blocks = level_block(first_block, int(start_filters * 2), depth=depth,
                                    filter_inc_rate=filter_inc_rate, p=0.1)
        if self.softmax:
            last_block = upsampl_softmax(middle_blocks, out_ch+1, 3, 1, 2, self.max_project) # out_ch+1, because softmax needs crossentropy
        else:
            last_block = upsampl_conv(middle_blocks, out_ch, 3, 1, 2)
        if self.crop:
            out = multiply([X, last_block])  # crop input with predicted mask
            return Model([X], [out], name='segmentor_net')
     
        return Model([X], [last_block], name='segmentor_net')
        
    def build_discriminator(self):

        X = Input(shape=self.source_shape)
        y = Input(shape=self.target_shape)
		
		
        shared_1 = shared_convl1_lrelu(self.source_shape, 64, 4, 2, name='shared_1_conv_lrelu')
        shared_2 = shared_convl1_bn_lrelu((16, 16, 64), 128, 4, 2, name='shared_2_conv_bn_lrelu')
        shared_3 = shared_convl1_bn_lrelu((8, 8, 128), 256, 4, 2, name='shared_3_conv_bn_lrelu')
        shared_4 = shared_convl1_bn_lrelu((4, 4, 256), 512, 4, 2, name='shared_4_conv_bn_lrelu')
        
        X_M = multiply([X, y])
        x1_S = shared_1(X_M)
        x2_S = shared_2(x1_S)
        x3_S = shared_3(x2_S)
        x4_S = shared_4(x3_S)
        features = Concatenate(name='features_S')(
            [Flatten()(X_M), Flatten()(x1_S), Flatten()(x2_S), Flatten()(x3_S), Flatten()(x4_S)]
        )
    
        return Model([X, y], features, name='critic_net')

    def rand_idx(self, n, nb_samples):
        return np.random.randint(0, n, size=nb_samples)

    def supervised_convert_X_Y_mask(self, nb_samples):
        idx = self.rand_idx(self.n, nb_samples)
        X_real = self.imgs[idx]; y_real = self.gt[idx]
        
        while len(X_real.shape)<4: X_real = np.expand_dims(X_real, axis=-1)
        y_fake = self.generator.predict(X_real)
        
       	v_real = self.discriminator.predict([X_real,y_real])
        v_fake = self.discriminator.predict([X_real,y_fake])
       
      	# We return a gt cropped image multi-level features of predicted mask
        # Negative sign, because critic needs to maximize loss
        return X_real, y_real, y_fake, -v_real, -v_fake  
      
	
    def unsupervised_convert_X_Y_mask(self, nb_samples):
        idx = self.rand_idx(self.n, nb_samples)
        X_real = self.imgs[idx]
        while len(X_real.shape) < 4: X_real = np.expand_dims(X, axis=-1)
        y_fake = self.generator.predict(X_real)
        v_fake = self.discriminator.predict([X_real, y_fake])
      
        return X_real, y_fake, -v_fake
	
    def convert_X_Y_mask(self, nb_samples, prop_groundtruth=0.3):
        """
        Experimental method for feeding labeled and unlabeled data during training,
        i.e. semi-supervised learning
        
        Parameters
        -----------
        
        prop_groundtruth: relative proportion of ground truth data for learning
        
        """
        # Feeds prop_groundtruth gt_images + (1-prop_groundtruth) uncropped images to C
        gt_samples = np.int(np.round(nb_samples*prop_groundtruth))
        non_gt_samples = np.int(np.round(nb_samples*(1-prop_groundtruth)))
        X_real, y_real, y_fake, v_real, v_fake = self.supervised_convert_X_Y_mask(gt_samples)
        if non_gt_samples:
            X_real, y_fake, v_fake = self.unsupervised_convert_X_Y_mask(non_gt_samples)
            X = np.concatenate((X_real, v_fake))
            y = np.concatenate((y_real, v_fake))
        else:
            X = X_real
            y = y_real
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        y = y[randomize]
        pred_y = y_fake[randomize]
        valid_real = v_real[randomize]; valid_fake = v_fake[randomize]
        return X, y, pred_y, valid_real, valid_fake


    def predicted_X_y_mask(self, nb_samples):
        idx = self.rand_idx(self.n, nb_samples); X = self.imgs[idx]; y = self.gt[idx]; print(X.shape, y.shape)
        while len(X.shape) < 4: X = np.expand_dims(X, axis=-1)
        y_fake = self.generator.predict(X)
        v_real = self.discriminator.predict([X,y]) 
        v_fake = self.discriminator.predict([X,y_fake]) 
        return X, y, y_fake, v_real, v_fake  

    def plot_losses(self):
        fig, ax = plt.subplots(nrows=1, ncols=4)
        #fig = plt.figure(figsize=(14.,14.))
        styles = ['bo', 'g+', 'r-', 'y.']
        loss_tuple = (self.disc_loss, self.gene_loss, self.disc_loss_real, self.disc_loss_fake)
        if not loss_tuple is tuple:  # if not tuple make it a tuple
            loss_tuple = (loss_tuple,)
        for s, l in enumerate(loss_tuple):
            print(ax.shape)
            ax[s].plot(l, styles[s])
        #ax.legend
        plt.show()
        plt.savefig('loss.png')

    def make_trainable(self, net, val):
        """
        Sets layers in keras model trainable or not
        
        val: trainable True or False
        """
        net.trainable = val
        for l in net.layers: l.trainable = val

    def train(self, nb_epoch=5000, batch_size=128, first=True, prop_groundtruth=0.3, trainC_iters = 5):
        # dl..average loss on discriminator
        # gl..loss on generator
        # rl..discriminator loss on real data
        # fl..discriminator loss on fake data
        # prop_groundtruth..proportion of ground truth images for training. Fully supervised if prop_groundtruth=1.0
        # trainC_iters..how many iterations the critic gets trained for per epoch of segmentor training

        for e in range(nb_epoch):
            i = 0  # batch counter
            while i < batch_size:
                self.make_trainable(self.discriminator, True)
                #self.make_trainable(self.combined_model.layers[-1], True) # make critic trainable
                d_iters = (100 if first and (self.gen_iterations < 25) or self.gen_iterations % 500 == 0
                           else trainC_iters)
                j = 0
                print("iterations:", d_iters)
                while j < d_iters and i < batch_size:
                    print('d_iter {}, epoch {}'.format(j,i))
                    j += 1
                    i += 1
                    X, y, pred_y, valid_real, valid_fake = self.convert_X_Y_mask(batch_size, prop_groundtruth)

                    self.disc_loss_real.append(self.discriminator.train_on_batch([X, y], valid_real))
                    self.disc_loss_fake.append(self.discriminator.train_on_batch([X, pred_y], valid_fake))
               

                    self.disc_loss.append(np.mean([self.disc_loss_real[-1], self.disc_loss_fake[-1]])) # average latest value
                  

                self.make_trainable(self.discriminator, False)
               
                # netC needs to maximize loss
                X_real, y_real, y_fake, v_real, v_fake  = self.predicted_X_y_mask(batch_size); 
                self.gene_loss.append(self.combined_model.train_on_batch([X_real,y_real], [v_real,y_real]))
                self.gen_iterations += 1
                
            if e % 5 == 0:
            	self.combined_model.save(models_path+'/epochs'+str(e)+'.h5'); print(self.disc_loss[-1], self.disc_loss_real[-1], self.disc_loss_fake[-1], self.gene_loss[-1])
                	#tqdm.write(
                     #   'G_Iters: {}, Loss_D: {:06.2f}, Loss_D_real: {:06.2f}, Loss_D_fake: {:06.2f}, Loss_G: {:06.2f} \n'.format(
                     #       self.gen_iterations, self.disc_loss[-1], self.disc_loss_real[-1], self.disc_loss_fake[-1], self.gene_loss[-1]))
        #self.plot_losses()			
        return self.disc_loss, self.gene_loss, self.disc_loss_real, self.disc_loss_fake

    def show_masks(self, out_layer=-2):
        return Model(self.generator.input, self.generator.layers[out_layer].output)
    
    def test(self,end_epoch):
        X = []
        y_index, y_pred_index = [], []
        y_rgb, y_pred_rgb = [], []
        file_index = 0
        for index in range(0,len(test_data)):
            iteration, test_X_rgb, test_X_norm, test_y_rgb, test_y_norm, file_name = next(test_batch)
            test_y_index = convert_rgb2index(test_y_rgb[0])
            if len(np.unique(test_y_index)) > 1:
                
                pred_y_fake = self.generator.predict(test_X_norm); #print("Pred:", pred_y_fake.shape)
                test_y_index_fake, test_y_rgb_fake = get_prediction_label(pred_y_fake[0]); #print(test_X_rgb.shape, test_y_rgb.shape, test_y_rgb_fake.shape)

                fig, ax = plt.subplots(1, 3, figsize=(10, 3))
                ax[0].set_title('X'); ax[1].axis('off'); ax[0].imshow(test_X_rgb[0])
                ax[1].set_title('True y'); ax[1].axis('off'); ax[1].imshow(test_y_rgb[0])
                ax[2].set_title('Pred y'); ax[2].axis('off'); ax[2].imshow(test_y_rgb_fake)
                plt.savefig(images_path+'/'+str(file_name)); plt.close(); file_index=file_index+1
                file_index = file_index+1
                X.append(test_X_rgb)
                y_index.append(test_y_norm);y_pred_index.append(pred_y_fake)
                y_rgb.append(test_y_rgb);y_pred_rgb.append(test_y_rgb_fake)
        
        X = np.array(X)
        y_index = np.array(y_index); y_pred_index = np.array(y_pred_index)
        y_rgb = np.array(y_rgb); y_pred_rgb = np.array(y_pred_rgb)

        np.savez(images_path+'/'+'results'+str(end_epoch)+'.npz', X=X, y_index=y_index, y_pred_index=y_pred_index, y_rgb=y_rgb, y_pred_rgb=y_pred_rgb)
        
if __name__ == '__main__':
    data_list = []; label_list = []
    for index in range(0,len(train_data)):
             iteration, X_rgb, X, y_rgb, y, file_name = next(train_batch)
             y = y[:,:,:,1]; y = y[:,:,:,np.newaxis]
             imageio.imsave('rgb.png', X_rgb[0])
             imageio.imsave('building.png', np.where(y[0] > 0, 255, 0)) 
             print('shape:', index, X.shape, y.shape)
             data = X[0]; label = y[0]; print(np.min(data), np.max(data)); print(np.min(label), np.max(label), np.unique(label))
             data_list.append(data)
             label_list.append(label)
    data_list = np.array(data_list); label_list = np.array(label_list)
    print(data_list.shape); print(label_list.shape)
    gan = SeGAN(imgs=data_list,gt=label_list)
    gan.test(1000)
    
    #gan.train(nb_epoch=5001, batch_size=32, first=True, prop_groundtruth=1, trainC_iters = 2)

    

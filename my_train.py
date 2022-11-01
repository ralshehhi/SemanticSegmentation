from functions import *
import tensorflow as tf

os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config = config)

#num_samples = 803
#dim = 2448
norm_method = sys.argv[1]
model = sys.argv[2]

path = '/scratch/ra130/Projects/DeepGlobe/Data2'
data_path = path + '/Data/train'

#X_path = data_path + '/**_sat.jpg'
#y_path = data_path + '/**_mask.png'
X_path = data_path + '/**.tiff'
y_path = data_path + '/**.png'


results_path = path +'/Results/'
os.mkdir(results_path) if not os.path.exists(results_path) else None

model_path = results_path + model + '_adam_binarycrossentropy_6classes/'
os.mkdir(model_path) if not os.path.exists(model_path) else None

list_X = load_data(X_path)
list_y = load_data(y_path)

list_X_y = list(zip(sorted(list_X), sorted(list_y)))

print('Reading')
train_batch = get_miniBatch(list_X_y, norm_method)

print('Building Networks')
generator = Generator(ISIZE=512, NC_IN=source_dimz, NC_OUT=target_dimz)
discriminator= Discriminator(ISIZE=512, NC_IN=source_dimz, NC_OUT=target_dimz)
#print('D:', discriminator.summary())
#print('G:', generator.summary())

X_real = generator.input
y_fake = generator.output
y_real = discriminator.inputs[1]

y_disc_real = discriminator([X_real, y_real])
y_disc_fake = discriminator([X_real, y_fake])

loss_fn = lambda pred_y, true_y : -K.mean(K.log(pred_y+1e-12)*true_y+K.log(1-pred_y+1e-12)*(1-true_y))

loss_disc_real = loss_fn(y_disc_real, K.ones_like(y_disc_real))
loss_disc_fake = loss_fn(y_disc_fake, K.zeros_like(y_disc_fake))

loss_gene_fake = loss_fn(y_disc_fake, K.ones_like(y_disc_fake))
loss_l1_yfake_yreal = K.mean(K.abs(y_fake-y_real))
lam = 100

loss_disc = loss_disc_real + loss_disc_fake
loss_gene = loss_gene_fake + lam * loss_l1_yfake_yreal

training_updates_disc = Adam(lr = 2e-4, beta_1 = 0.5).get_updates(discriminator.trainable_weights, [], loss_disc)
discriminator_train = K.function([X_real, y_real], [loss_disc/2.0], training_updates_disc)

training_updates_gene = Adam(lr = 2e-4, beta_1 = 0.5).get_updates(generator.trainable_weights, [], loss_gene)
generator_train = K.function([X_real, y_real], [loss_gene_fake, loss_l1_yfake_yreal], training_updates_gene)

t00 = time.time()
index = 0
error_gen = 0
error_l1 = 0
sum_error_l1 = 0
sum_error_gene = 0
sum_error_disc_real = 0
sum_error_disc_fake = 0
num_epoches = 200
num_checkpoints = 10

#epoch_index = 1000
#model_name =  model_path + 'epoch' + str(epoch_index) + '.h5'
#generator = load_model(model_name)
print('Training')
while index <= num_epoches:
	t0 = time.time()
	iteration, train_X_rgb, train_X, train_y_rgb, train_y, fn = next(train_batch)
	error_disc_real,  = discriminator_train([train_X, train_y])
	sum_error_disc_real += error_disc_real

	error_disc_fake,  = discriminator_train([train_X, y_fake])
	sum_error_disc_fake += error_disc_fake
	d_loss = 0.5 * np.add(sum_error_disc_real, sum_error_disc_fake)
	
	error_gene, error_l1 = generator_train([train_X, train_y])
	sum_error_gene += error_gene
	sum_error_l1 += error_l1
	index += 1
	iteration += 1
	t1 = time.time()
	print ("[Epoch %d/%d] [D loss: %f] [D loss: %f, acc: %3d%%] time: %s" % (index, num_epoches, sum_error_disc_real,d_loss, sum_error_gene, sum_error_l1, (t1-t0)))
	if index % num_checkpoints ==0:
		model = model_path+'/epoch'+'%d'%index+'.h5'
		generator.save(model)
t11 = time.time()
print('Time for one epoch:', (t11-t00))
	

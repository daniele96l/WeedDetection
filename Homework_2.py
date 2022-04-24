#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ## SETUP

# In[ ]:


from keras.layers import *
from keras.models import Model
from keras import losses
from keras import utils
from keras.utils import to_categorical
from tensorflow.python.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
from PIL import Image
from matplotlib import pyplot
import json
from datetime import datetime

import math
import random

import itertools

import tensorflow as tf


# In[ ]:


out_shape = [512, 512] #TODO
bs = 4
SEED = 2347


# In[ ]:


dataset_dir = '/content/drive/My Drive/ANNDL_Homework2/'

training_dir = os.path.join(dataset_dir, "training")
assert(os.path.exists(dataset_dir))

test_dir = os.path.join(dataset_dir, "Test_Dev")
assert(os.path.exists(test_dir))

# DATA PRE-PROCESSED
s = '_' + str(out_shape[0]) + 'x' + str(out_shape[1])
precomputed_data = os.path.join(training_dir, "precomputed_data")
if (not os.path.exists(precomputed_data)):
  os.makedirs(precomputed_data)

train_img = os.path.join(precomputed_data, 'train_img'+s)
train_lab = os.path.join(precomputed_data, 'train_lab'+s)
val_img = os.path.join(precomputed_data, 'val_img'+s)
val_lab = os.path.join(precomputed_data, 'val_lab'+s)

models_dir = os.path.join(dataset_dir, "models")
if (not os.path.exists(models_dir)):
  os.makedirs(models_dir)

prediction_dir = os.path.join(dataset_dir, "predictions")
if (not os.path.exists(prediction_dir)):
  os.makedirs(prediction_dir)

ckpt_dir = os.path.join(dataset_dir, "checkpoints")
if (not os.path.exists(ckpt_dir)):
  os.makedirs(ckpt_dir)

tb_dir = os.path.join(dataset_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)


# ## UTILS

# ### Image manipulation

# 
# #### Labelling
# 
# | RGB        | TARGET         |
# | ------------- |-------------:|
# | 0 0 0 |  0 (background)|
# |216 124 18 | 0 (background)|
# | 255 255 255 | 1 (crop) |
# | 216 67 82 | 2 (weed) |

# In[ ]:


# Convert from RGB image to labels

def read_rgb_mask(mask_img):
    
    mask_arr = np.array(mask_img)

    new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)

    # Use RGB dictionary in 'RGBtoTarget.txt' to convert RGB to target
    new_mask_arr[np.where(np.all(mask_arr == [216, 124, 18], axis=-1))] = 0
    new_mask_arr[np.where(np.all(mask_arr == [255, 255, 255], axis=-1))] = 1
    new_mask_arr[np.where(np.all(mask_arr == [216, 67, 82], axis=-1))] = 2

    return new_mask_arr


# #### Image size "wise"-reduction
# 
# > Indented block
# 
# 

# In[ ]:


def reduce_img(img, out_shape):
  imgs = []
  for r in range(0, img.shape[0]-out_shape[0], out_shape[0]):
    for c in range(0, img.shape[1]-out_shape[1], out_shape[1]):
      imgs.append(img[r:r + out_shape[0], c: c+ out_shape[1]])
  return imgs


# ### Generation of the dataset for a given team

# In[ ]:


class CustomDataset(tf.keras.utils.Sequence):

  def __init__(self, images, masks, which_set):

    if (which_set == 'training'):
      self.img_data_gen = ImageDataGenerator(rotation_range=10,
                                      width_shift_range=10,
                                      height_shift_range=10,
                                      zoom_range=0.3,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      fill_mode='reflect')

      self.mask_data_gen = ImageDataGenerator(rotation_range=10,
                                       width_shift_range=10,
                                       height_shift_range=10,
                                       zoom_range=0.3,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='reflect')
    else:
      self.img_data_gen = ImageDataGenerator()
      self.mask_data_gen = ImageDataGenerator()

    self.images = images
    self.masks = masks


  def __len__(self):
    return images.shape[0]

  def __getitem__(self, index):
    image = self.images[index]
    mask = np.expand_dims( self.masks[index], axis=-1 )

    img_t = self.img_data_gen.get_random_transform(img_arr.shape, seed=SEED) # TODO maybe the seeds shouldn't be here
    mask_t = self.mask_data_gen.get_random_transform(mask_arr.shape, seed=SEED)

    img_arr = self.img_generator.apply_transform(img_arr, img_t)

    out_mask = np.zeros_like(mask)
    for c in np.unique(mask):
      if c > 0:
        curr_class_arr = np.float32(mask_arr == c)
        curr_class_arr = self.mask_generator.apply_transform(curr_class_arr, mask_t)
        
        curr_class_arr = np.uint8(curr_class_arr)
        # recover original class
        curr_class_arr = curr_class_arr * c 
        out_mask += curr_class_arr

    return image, np.float32(out_mask)


# In[ ]:


def get_dataset(team):

  print("Getting dataset for team " + team)
  
  train_file_im = os.path.join(train_img, team + ".npy")
  val_file_im = os.path.join(val_img, team + ".npy")
  train_file_mask = os.path.join(train_lab, team + ".npy")
  val_file_mask = os.path.join(val_lab, team + ".npy")

  print("Loading images")
  # Load the pre-processed images
  t_img = np.load(train_file_im)
  v_img = np.load(val_file_im)

  t_mask = np.load(train_file_mask)
  v_mask = np.load(val_file_mask)

  print("Cutting arrays")
  # If the dataset created is too big, remove some images
  cut = 1 #TODO modify according to needs
  limit_data_t = min( int(t_img.shape[0] * cut), 1000 )#TODO  
  limit_data_v = min( int(v_img.shape[0] * cut), 200 )#TODO

  t_img = t_img[:limit_data_t]
  v_img = v_img[:limit_data_v]
  t_mask = t_mask[:limit_data_t]
  v_mask = v_mask[:limit_data_v]

  # coherence check
  assert(t_img.shape[0:3] == t_mask.shape)
  assert(v_img.shape[0:3] == v_mask.shape)

  train_dataset = CustomDataset(t_img, t_mask, 'training')
  val_dataset = CustomDataset(v_img, v_mask, 'validation')

  img_h = out_shape[0]
  img_w = out_shape[1]
  
  print("Create the dataset")
  train_dataset = tf.data.Dataset.from_generator( lambda: train_dataset,
                                                 output_types=(tf.float32, tf.float32),
                                                output_shapes=([img_h, img_w, 3], [img_h, img_w, 1]))
  
  val_dataset = tf.data.Dataset.from_generator( lambda: val_dataset,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([img_h, img_w, 3], [img_h, img_w, 1]))
  
  train_dataset = train_dataset.batch(bs)
  train_dataset = train_dataset.repeat()

  val_dataset = val_dataset.batch(bs)
  val_dataset = val_dataset.repeat()

  return train_dataset, val_dataset, limit_data_t, limit_data_v
  


# ### Run-lenght encoding

# In[ ]:


def rle_encode(img):
    
    #img: numpy array, 1 - foreground, 0 - background
    #Returns run length as string formatted

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# ### Model

# #### Trivial model from lecture

# In[ ]:




def create_trivial_model(im_shape, depth, start_f, num_classes):

    model = tf.keras.Sequential()
    
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=im_shape)

    for layer in vgg.layers:
      layer.trainable = False

    model.add(vgg)
    
    # Encoder
    # -------
    
    start_f = 256
        
    # Decoder
    # -------
    for i in range(depth):
        model.add(tf.keras.layers.UpSampling2D(2, interpolation='bilinear'))
        model.add(tf.keras.layers.Conv2D(filters=start_f,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same'))
        model.add(tf.keras.layers.ReLU())

        start_f = start_f // 2
    
    # Prediction Layer
    # ----------------
    model.add(tf.keras.layers.Conv2D(filters=num_classes,
                                     kernel_size=(1, 1),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='softmax'))
    
    return model


# #### Unet

# In[ ]:


# Convolution Block

def __conv_block(input_tensor, num_filters, kernel_size=3):
    # First layer
    encoder = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same')(input_tensor)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    # Second layer
    encoder = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)

    return encoder


# Encoder block

def __encoder_block(input_tensor, num_filters, pool_size=2, strides=2):
    # Encoder
    encoder = __conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(strides, strides))(encoder)

    return encoder_pool, encoder


# Decoder block

def __decoder_block(input_tensor, concat_tensor, num_filters, kernel_size=3, transpose_kernel_size=2, strides=2):
    # Conv2DTranspose AKA Deconvolution
    decoder = Conv2DTranspose(filters=num_filters, kernel_size=(transpose_kernel_size, transpose_kernel_size),
                              strides=(strides, strides), padding='same')(input_tensor)
    # Concatenate
    decoder = concatenate([concat_tensor, decoder], axis=-1)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    return decoder


# BUILD MODEL

def get_unet_model(img_shape):

    base_n_filters = 32

    # Input
    inputs = Input(shape=img_shape)

    # Encoders
    encoder0_pool, encoder0 = __encoder_block(inputs, base_n_filters) 
    encoder1_pool, encoder1 = __encoder_block(encoder0_pool, base_n_filters*2) 
    encoder2_pool, encoder2 = __encoder_block(encoder1_pool, base_n_filters*4) 
    encoder3_pool, encoder3 = __encoder_block(encoder2_pool, base_n_filters*8)
    encoder4_pool, encoder4 = __encoder_block(encoder3_pool, base_n_filters*16)
    encoder5_pool, encoder5 = __encoder_block(encoder4_pool, base_n_filters*32)

    # Center
    center = __conv_block(encoder4_pool, base_n_filters*64) 

    # Decoders
    decoder5 = __decoder_block(center, encoder5, base_n_filters*32)
    decoder4 = __decoder_block(center, encoder4, base_n_filters*16)
    decoder3 = __decoder_block(decoder4, encoder3, base_n_filters*8)
    decoder2 = __decoder_block(decoder3, encoder2, base_n_filters*4)
    decoder1 = __decoder_block(decoder2, encoder1, base_n_filters*2)

    #TODO just a test
    decoder0 = __decoder_block(decoder1, encoder0, base_n_filters)

    # Output
    outputs = Conv2D(3, (1, 1), activation='softmax')(decoder0)  # 3 classes

    # Create model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


# #### Metrics

# In[ ]:


# Optimization params
# -------------------

# Loss
# Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
loss = tf.keras.losses.SparseCategoricalCrossentropy() 

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Here we define the intersection over union for each class in the batch.
# Then we compute the final iou as the weighted mean over classes
def meanIoU(y_true, y_pred):
    # get predicted class from softmax
    y_pred = tf.argmax(y_pred, -1)

    per_class_iou = []

    for i in range(1,3):
      # Get prediction and target related to only a single class (i)
      class_pred = tf.cast(tf.where(y_pred == i, 1, 0), tf.float32)
      class_true = tf.cast(tf.where(y_true == i, 1, 0), tf.float32)
      intersection = tf.reduce_sum(class_true * class_pred)
      union = tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) - intersection
    
      iou = ( (intersection + 1e-7) / (union + 1e-7) ) * i/3  # weight more the weeds, to improve their recognition
      per_class_iou.append(iou)

    return tf.reduce_mean(per_class_iou)

# Validation metrics
# ------------------
metrics = ['accuracy', meanIoU]
# ------------------


# In[ ]:


# Model checkpoint
# ----------------
callbacks = []

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

# Visualize Learning on Tensorboard
# ---------------------------------
    
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=0)  # if 1 shows weights histograms
callbacks.append(tb_callback)

# Early Stopping
# --------------

es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
callbacks.append(es_callback)


# ## PREPROCESSING DATA 
# 
# ### Prepare datasets
# 
# Save them for convenience

# Training and validation

# In[ ]:


validation_split = 0.8

# If the dataset has not been created yet
if ( not os.path.exists(os.path.abspath(train_img)) or not os.path.exists(os.path.abspath(val_img)) or not os.path.exists(os.path.abspath(train_lab)) or not os.path.exists(os.path.abspath(val_lab))):

  os.makedirs(train_img)
  os.makedirs(val_img)
  os.makedirs(train_lab)
  os.makedirs(val_lab)

  for team in os.listdir(test_dir):
    team_path = os.path.join(training_dir, team)

    img_arr = []
    mask_arr = []

    for crop in os.listdir(team_path):
      crop_path_gen = os.path.join(team_path, crop)
      crop_path_im = os.path.join(crop_path_gen, "Images")
      crop_path_mask = os.path.join(crop_path_gen, "Masks")

      # For every training images #TODO may be too much
      for file in os.listdir(crop_path_im):
        print("Elaborating image " + file)

        # Get image...
        img_path = os.path.join(crop_path_im, file)
        img = Image.open(img_path)

        # And corresponding mask (RGB)
        mask_path = os.path.join(crop_path_mask, os.path.splitext(file)[0] + '.png')
        assert(os.path.exists(mask_path))
        mask = Image.open(mask_path)

        img = np.array(img)
        mask = np.array(mask)

        # Get small images from the big picture to allow faster (and especially doable...) training
        imgs = reduce_img(img, out_shape)
        masks = reduce_img(mask, out_shape)
        
        # If the images generated are too many, pick some
        if(len(imgs) > 6):
          step = 2
          start = np.random.randint(2)
        else:
          step = 1
          start = 0

        # Add the tiles to a list
        for i in range(start, len(imgs), step):
          # get labels array
          new_mask = read_rgb_mask(masks[i])

          # append to a list
          img_arr.append(np.array(imgs[i]))
          mask_arr.append(np.array(new_mask))

    # Shuffle all the data
    random.Random(SEED).shuffle(img_arr)
    random.Random(SEED).shuffle(mask_arr)

    # separate train and validation
    bound = math.floor(len(img_arr)*validation_split)
    t_im = np.array(img_arr[: bound])
    v_im = np.array(img_arr[bound :])

    t_m = np.array(mask_arr[: bound])
    v_m = np.array(mask_arr[bound :])

    np.save(os.path.join(train_img, team), t_im)
    np.save(os.path.join(val_img, team), v_im)

    np.save(os.path.join(train_lab, team), t_m)
    np.save(os.path.join(val_lab, team), v_m)


# ## TRAINING
# 

# In[ ]:


# Train a model for each team
im_shape = [None, None, 3] # We want to train on small images but predict on big ones

now = datetime.now().strftime('%b%d_%H-%M-%S')
print(str(now))

teams = ["Bipbip", "Pead", "Weedelec", "Roseau"]

for team in teams:
  # Get Unet model
  model = get_unet_model(im_shape)

  # Get the dataset
  x, y, lenx, leny = get_dataset(team)
  print("Got the generator - " + str(lenx) + " images for training and " + str(leny) + " for validation")
  
  # Compile Model
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  # Uncomment if you have a checkpoint to start with
  if (team == "Bipbip"):
    model_name = os.path.join(ckpt_dir, "cp_16.ckpt")
    model.load_weights(model_name)

  assert(model)
  print("Model is ready")

  epochs = 30 #TODO

  # Train
  model.fit(x, 
            epochs=epochs,
            steps_per_epoch=lenx/bs,
            validation_data=y,
            validation_steps=leny/bs,
            callbacks=callbacks,
            verbose=1)

  model_name = os.path.join(models_dir, str(now) + "--" + team)

  model.save(model_name)


# # PREDICTION

# In[ ]:


model_basename = "Dec21_18-18-52"  # Select a model


# Predict

# In[ ]:


pred_dir = os.path.join(prediction_dir, os.path.basename(model_basename))

for team in os.listdir(test_dir):

  print("TEAM " + team)

  # Load the right model
  model_name = os.path.join(models_dir, model_basename + "--" + team) # If the models have the same basename!
  model = tf.keras.models.load_model(model_name, custom_objects={'meanIoU':meanIoU})

  # The predictions will be saved
  pred_dir_team = os.path.join(pred_dir, team)
  team_path = os.path.join(test_dir, team)

  for crop in os.listdir(team_path):

    crop_path = os.path.join(team_path, crop)
    crop_path = os.path.join(crop_path, "Images")

    pred_dir_crop = os.path.join(pred_dir_team, crop)

    if(not os.path.exists(pred_dir_crop)):
      os.makedirs(pred_dir_crop)

    for img_file in os.listdir(crop_path):
      # Get the image
      print(img_file)
      img = np.array(Image.open(crop_path + "/" + img_file))

      orig_im_w = img.shape[0]
      orig_im_h = img.shape[1]
      
      # Resize images to adapt to unet
      # Make the image divisible in patches as big as the training ones for coherence
      new_size = [math.ceil(orig_im_w/out_shape[0])*out_shape[0], math.ceil(orig_im_h/out_shape[1]) * out_shape[1], 3]
      # Add padding
      upscaled_img = np.zeros(new_size)
      upscaled_img[:orig_im_w, :orig_im_h, :] = img

      # Predict on the tiles and then attach the results
      i = 0
      out_sigm = np.zeros_like(upscaled_img)
      for r in range(0, new_size[0]+1-out_shape[0], out_shape[0]):
        for c in range(0, new_size[1]+1-out_shape[1], out_shape[1]):
          i += 1
          prediction = (np.squeeze(
                        model.predict(
                        np.expand_dims(
                        upscaled_img[r:r+out_shape[0], c:c+out_shape[1]], axis=0), verbose=False), axis=0))
          out_sigm[r:r+out_shape[0], c:c+out_shape[1]] = prediction
          
      # For better precision, take other tiles and average the results
      for r in range(int(new_size[0]/2), new_size[0]+1-int(out_shape[0]/2), out_shape[0]):
        for c in range(int(new_size[1]/2), new_size[1]+1-int(out_shape[1]/2), out_shape[1]):
          prediction = (np.squeeze(
                        model.predict(
                        np.expand_dims(
                        upscaled_img[r:r+out_shape[0], c:c+out_shape[1]], axis=0), verbose=False), axis=0))
          
          out_sigm[r:r+out_shape[0], c:c+out_shape[1]] = (prediction + out_sigm[r:r+out_shape[0], c:c+out_shape[1]]) / 2

      predict = tf.argmax(out_sigm, -1)
      # cut away the padding
      predict = predict[: orig_im_w, : orig_im_h]

      save_path = os.path.join(pred_dir_crop, img_file)
      np.save(save_path, predict)


# In[ ]:


fig=pyplot.figure(figsize=(10, 500))
i = 1
col = 3
row = 100

pred_dir = os.path.join(prediction_dir, os.path.basename(model_basename))

for team in os.listdir(pred_dir):
  print("TEAM " + team)
  tp = os.path.join(pred_dir, team)

  to = os.path.join(test_dir, team)

  for crop in os.listdir(tp):
    print("CROP " + crop)
    cp = os.path.join(tp, crop)

    co = os.path.join(to, crop)
    co = os.path.join(co, "Images")

    for pred in os.listdir(cp)[:1]:
      print(pred)
      #show prediction and original image
      p = np.load(os.path.join(cp, pred))
      o = Image.open(os.path.join(co, os.path.splitext(pred)[0]))

      fig.add_subplot(row, col, i)
      pyplot.imshow(o)

      fig.add_subplot(row, col, i+1)
      pyplot.imshow(p==1, cmap=pyplot.cm.gray)
      fig.add_subplot(row, col, i+2)
      pyplot.imshow(p==2, cmap=pyplot.cm.gray)
      
      i +=3


# In[ ]:


pred_dir = os.path.join(prediction_dir, model_basename)
submission_dict = {}

for team in os.listdir(pred_dir):
  print("TEAM " + team)
  tp = os.path.join(pred_dir, team)

  for crop in os.listdir(tp):
    print("CROP " + crop)
    cp = os.path.join(tp, crop)

    for pred in os.listdir(cp):
      print(pred)
      mask = np.load(os.path.join(cp, pred))

      name = os.path.splitext(pred)[0]
      name = os.path.splitext(name)[0]

      submission_dict[name] = {}
      submission_dict[name]['shape'] = mask.shape
      submission_dict[name]['team'] = team
      submission_dict[name]['crop'] = crop
      submission_dict[name]['segmentation'] = {}

      # RLE encoding
      # crop
      rle_encoded_crop = rle_encode(mask == 1)
      # weed
      rle_encoded_weed = rle_encode(mask == 2)

      submission_dict[name]['segmentation']['crop'] = rle_encoded_crop
      submission_dict[name]['segmentation']['weed'] = rle_encoded_weed


# Finally, save the results into the submission.json file
delivery_folder = os.path.join(prediction_dir, "output-" + model_basename)
if (not os.path.exists(delivery_folder)):
  os.makedirs(delivery_folder)

with open(os.path.join(delivery_folder, "submission.json"), 'w') as f:
  json.dump(submission_dict, f)

print("File saved as " + os.path.abspath(os.path.join(delivery_folder, "submission.json")))


# In[ ]:


def rle_decode(rle, shape):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

with open('/content/drive/My Drive/ANNDL_Homework2/predictions/output-' + model_basename + '/submission.json', 'r') as f:
    submission_dict = json.load(f)


img_name = 'Bipbip_mais_im_04121'
img_shape = submission_dict[img_name]['shape']

rle_encoded_crop = submission_dict[img_name]['segmentation']['crop']
rle_encoded_weed = submission_dict[img_name]['segmentation']['weed']

# Reconstruct crop and weed binary masks
crop_mask = rle_decode(rle_encoded_crop, shape=img_shape)
weed_mask = rle_decode(rle_encoded_weed, shape=img_shape)

# Reconstruct original mask
# weed_mask * 2 allows to convert ones into target 2 (weed label)
reconstructed_mask = crop_mask + (weed_mask * 2)

print(img_shape)
print(np.count_nonzero(reconstructed_mask==2))
pyplot.imshow(reconstructed_mask)


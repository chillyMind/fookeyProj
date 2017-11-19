import sys
import os
os.getenv('PYTHONPATH')
import numpy as np
import scipy.io
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
import VGG as vgg

##
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

cwd = os.getcwd()
cwd = cwd + "/public/fookey1"

imgsize = [192, 192]     # The reshape size
data_name = "data4vgg"  # Save name
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
# ------------------------------------------------------------------- #

category_name = sys.argv[2]
category_path = cwd + "/FOOD/"+ category_name
categories = sorted(os.listdir(category_path))
nclass = len(categories)

## wrong Order So change up and down
fc_num = 99
imgsize = [192, 192]     # The reshape size
if category_name == "rice":
    fc_num = 99
    imgsize = [192, 192] 
elif category_name == "soup":
    fc_num = 99
    imgsize = [192, 192] 
elif category_name == "dish":
    fc_num = 95
    imgsize = [192, 192] 
elif category_name == "side":
    fc_num = 85
    imgsize = [192, 192] 

fc_num = 99

##IMG_PATH = cwd + "/test/testimage.jpg"
IMG_PATH = cwd + "/" + sys.argv[1]

img = imread(IMG_PATH)
currimg = imread(IMG_PATH)


# Crop
cw = currimg.shape[0] / 2
ch = currimg.shape[1] / 2
Len = cw
if ch < Len:
    Len = ch
#hw = imgsize[0] / 2
#hh = imgsize[1] / 2
currimg = currimg[cw - Len: cw + Len, ch - Len: ch + Len, :]
# Resize
currimg = imresize(currimg, [imgsize[0], imgsize[1]])/255.

# Reshape
currimg = currimg.reshape((-1,) + currimg.shape)   #(h, w, nch) => (1, h, w, nch)


# PARAMETERS
n_output = nclass
fwid = imgsize[0] / 16
fhei = imgsize[1] / 16

tf.reset_default_graph()
with tf.device("/cpu:0"):
    cam_weights = {
        'wc': tf.Variable(tf.truncated_normal([fwid, fhei, 512, 512], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([512, n_output], stddev=0.1))
    }
    cam_biases = {
        'bc': tf.Variable(tf.random_normal([512], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
    # NETWORK
    def cam(_x, _W, _b, _kr):
        conv = tf.nn.conv2d(_x, _W['wc'], strides=[1, 1, 1, 1], padding='SAME')
        conv_relu = tf.nn.relu(tf.nn.bias_add(conv, _b['bc']))
        conv_dr = tf.nn.dropout(conv_relu, _kr)
        
        gap = tf.nn.avg_pool(conv_dr, ksize=[1, fwid, fhei, 1], strides=[1, fwid, fhei, 1], padding='SAME')
        gap_dr = tf.nn.dropout(gap, _kr)
        gap_vec = tf.reshape(gap_dr, [-1, _W['out'].get_shape().as_list()[0]])
        out = tf.add(tf.matmul(gap_vec, _W['out']), _b['out'])
        ret = {'gap': gap, 'gap_dr': gap_dr, 'gap_vec': gap_vec, 'out':out}
        return ret
    
VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"
y = tf.placeholder(tf.float32, [None, nclass])
keepratio = tf.placeholder(tf.float32)
with tf.device("/cpu:0"):
    img_placeholder = tf.placeholder(tf.float32, shape=(None, imgsize[0], imgsize[1], 3))
    camnet, _ = vgg.net(VGG_PATH, img_placeholder)
    cam_pred = cam(camnet['relu5_4'], cam_weights, cam_biases, keepratio)['out']
    
    cam_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cam_pred, labels=y))
    cam_optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cam_cost)
    cam_corr = tf.equal(tf.arg_max(cam_pred, 1), tf.argmax(y, 1))
    cam_accr = tf.reduce_mean(tf.cast(cam_corr, 'float'))
    cam_init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(cam_init)

#fc_num = 99
netname = cwd + "/nets/" + category_name + "_cam.ckpt-" + str(fc_num)
saver.restore(sess, netname)

# Feed forward
feature_map, my_pred = sess.run([camnet, cam_pred], feed_dict={img_placeholder:currimg, keepratio: 1.})

# Predict label
# my_label = np.argmax(my_pred)
# #print("This image is %s(%d)" %(categories[my_label], my_label))
# #category_name = categories[my_label]
# #print(categories);
   

# res = ""
# for i, cate in enumerate(categories):
#     res = res + cate
#     if i is (len(categories) - 1):
#         continue
#     res = res + ";"
# print (res)

res = ""
cat_cnt = len(categories) ## category value
#mypred -> weight

def pred(tup):
    return -tup[1]
result = zip(categories, my_pred[0])
result.sort(key = pred)
for i, val in enumerate(result):
    res = res + val[0]
    if i is (len(result) - 1):
        continue
    res = res + ";"
sys.stdout.write(res)

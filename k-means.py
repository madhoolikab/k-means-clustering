import pickle
import numpy as np

# Data Pre-processing
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names
        
        
#Directory where dataset is stored        
cifar_10_dir = 'data\cifar-10-batches-py'

train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

# Generating DataMatrix(Each data point is represented by a feature matrix)
"Converting CIFAR 10 DATA SET INTO Grayscale"
train_data_final = []
for train_img in train_data:
    train_data_final.append(cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY))

test_data_final = []
for test_img in test_data:
    test_data_final.append(cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY))
    
image = train_data_final[0]
  
def bins_grayscale(image):
    vec_image=image.flatten()
    #bin 0-25
    x1=len(np.where(vec_image<=25)[0])
    #bin 25-50
    x2=len(np.where(vec_image<=50)[0])-len(np.where(vec_image<=25)[0])
    #bin 50-75
    x3=len(np.where(vec_image<=75)[0])-len(np.where(vec_image<=50)[0])
    #bin 75-100
    x4=len(np.where(vec_image<=100)[0])-len(np.where(vec_image<=75)[0])
    #bin 100-125
    x5=len(np.where(vec_image<=125)[0])-len(np.where(vec_image<=100)[0])
    #bin 125-150
    x6=len(np.where(vec_image<=150)[0])-len(np.where(vec_image<=125)[0])
    #bin 150-175
    x7=len(np.where(vec_image<=175)[0])-len(np.where(vec_image<=150)[0])
    #bin 175-200
    x8=len(np.where(vec_image<=200)[0])-len(np.where(vec_image<=175)[0])
    #bin 200-225
    x9=len(np.where(vec_image<=225)[0])-len(np.where(vec_image<=200)[0])
    #bin 225-255
    x10=len(np.where(vec_image<=255)[0])-len(np.where(vec_image<=225)[0])
    temp=[]
    temp=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
    temp=np.array(temp).reshape(1,10)
    return temp

# creating a data matrix (50000 * 10)
feature_matrix=np.zeros((1,10))
for i in train_data_final:
    temp = bins_grayscale(i)
    feature_matrix = np.vstack((feature_matrix,temp))
    
feature_matrix=feature_matrix[1:50001,:]
feature_matrix=feature_matrix.transpose()
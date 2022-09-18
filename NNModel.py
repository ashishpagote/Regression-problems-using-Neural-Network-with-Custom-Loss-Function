

print("Importing libraries")
from keras import backend as K
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import datetime as dt
import numpy as np
import keras
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import BatchNormalization
import scipy
from sklearn.decomposition import TruncatedSVD

error_1_weight=2
error_2_weight=1

def data_preprocessing(df,numeric_col_list,n_components,target):
    one_hot = pd.get_dummies(df.drop([numeric_col_list],axis = 1))
    df2 = pd.concat([df,one_hot],axis=1)
    csr_sparse_matrix = scipy.sparse.csr_matrix(df2.values)
    pca = TruncatedSVD(n_components=n_components)
    principalComponents = pca.fit_transform(csr_sparse_matrix)
    print(pca.explained_variance_ratio_.sum())
    principalDf = pd.DataFrame(data = principalComponents)
    df2 = pd.concat([principalDf, df2[[target]]], axis = 1)
    return df2


def customLossCreator(error_1_weight, error_2_weight):
    def custom_loss(y_true, y_pred):
        diff =  math_ops.sub(y_true , y_pred)
        diff_error_1 = tf.math.multiply(error_1_weight,tf.math.multiply(diff,diff))
        diff_error_2 = tf.math.multiply(error_2_weight,tf.math.multiply(diff,diff))
        diff4 = tf.where(tf.greater(diff, 0), diff_error_1, diff_error_2)
        loss = K.mean(diff4, axis=-1) #mean over last dimension
        return loss
    return custom_loss


def createModel(X_train):
    return keras.Sequential([
                   keras.layers.Dense(32, activation='relu', input_shape=(1,X_train.shape[1]), dtype='float32'),
                   BatchNormalization(),
                   keras.layers.Dense(16, activation='relu'), 
                   BatchNormalization(),
                   keras.layers.Dense(8, activation='relu'),
                   BatchNormalization(),
                   keras.layers.Dense(1)])


def fetchNumbers(model, X_train,y_train,test_main_X,test_main_y,model_name):
    y_pred = model.predict(X_train)
    print('MSE:',mean_squared_error(y_train, y_pred))
    y_pred_round = [np.round(x) for x in y_pred]
    print('Rounded MSE:',mean_squared_error(y_train, y_pred_round))
    y_train_arr = np.array(y_train)
    accuracy = [1  for i in range(0,len(y_train)) if (y_train_arr[i] == y_pred_round[i])]
    error_1 = [1  for i in range(0,len(y_train)) if y_train_arr[i] > y_pred_round[i]]
    error_2 = [1  for i in range(0,len(y_train)) if y_train_arr[i] < y_pred_round[i] ]
    print('Train BAE Nos:',np.sum(accuracy)/len(y_train),np.sum(error_1)/len(y_train),np.sum(error_2)/len(y_train))
    
    
    y_pred_test = model.predict(test_main_X)
    print('MSE:',mean_squared_error(test_main_y, y_pred_test))
    y_pred_round_test = [np.round(x) for x in y_pred_test]
    print('Rounded MSE:',mean_squared_error(test_main_y, y_pred_round_test))
    y_test_arr = np.array(test_main_y)
    accuracy = [1  for i in range(0,len(test_main_y)) if (y_test_arr[i] == y_pred_round_test[i])]
    error_1 = [1  for i in range(0,len(test_main_y)) if y_test_arr[i] > y_pred_round_test[i]]
    error_2 = [1  for i in range(0,len(test_main_y)) if y_test_arr[i] < y_pred_round_test[i]]
    print('Test BAE Nos:',np.sum(accuracy)/len(test_main_y),np.sum(error_1)/len(test_main_y),np.sum(error_2)/len(test_main_y))
    
    model.save( 'reg_model_' + str(dt.today().date()) + model_name + '.h5')
    X_train['actual'] = y_train
    X_train['predicted'] = y_pred_round
    return X_train


def createAndSaveModel(main_X, main_y,test_main_X,test_main_y,model_name, breaches_weight, earliness_weight):
   
    tf.config.run_functions_eagerly(True)
    X_train = main_X.copy()
    y_train = main_y.copy()
    model = createModel(X_train)
    model.compile(loss=customLossCreator(error_1_weight,error_2_weight), optimizer='sgd')
    plot_model(model, to_file=model_name + '.png', show_shapes=True, show_layer_names=True)
    model.fit(X_train, y_train, batch_size=256, epochs=8)
    return fetchNumbers(model, X_train, y_train, test_main_X,test_main_y,model_name)

def data_split(df,time_interval,target):
    train_data = df[:time_interval].sample(frac=1)
    test_data=df[time_interval:]
    main_X=train_data.drop([target],axis = 1)
    main_y = train_data[target].astype('float')
    test_main_X=test_data.drop([target],axis = 1)
    test_main_y=test_data[target].astype('float')
    return main_X, main_y, test_main_X, test_main_y

choice_set=[(error_1_weight,error_2_weight)]

print('read your dataframe')
print('define numerical col, target and number of principle components')

'''
1.pass the dataframe through :
   a.data_preprocessing  
   b.data_split
   
2.pass the main_X, main_y, test_main_X, test_main_y found in above through createAndSaveModel function

3.Run the below line of codes:
    result_set=[createAndSaveModel(main_X, main_y,test_main_X,test_main_y, 'Model-v0.1'+str(i)+str(j), i,j) for i,j in choice_set]
    res = result_set[0]
    res['predicted'] = res['predicted'].map(lambda x: x[0])
    df.join(res[['actual','predicted']]).to_csv('Results_'+ str(dt.today().date()) + '.csv.gz', compression = 'gzip')



'''
##result_set=[createAndSaveModel(main_X, main_y,test_main_X,test_main_y, 'Model-v0.1'+str(i)+str(j), i,j) for i,j in choice_set]
    

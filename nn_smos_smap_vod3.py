import sklearn as skl
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
import time
from datetime import timedelta

print('NEW version of the VOD data, no SM')


path_to_file = '/burg/glab/users/os2328/data/VOD_project/'
path_to_save_nn = '/burg/glab/users/os2328/data/VOD_project/'
path_to_save_output = '/burg/glab/users/os2328/data/VOD_project/'
file_data = path_to_file + 'smos_3dm_seas_c_no_std.pkl'
smos_m = pd.read_pickle(file_data)
smos_m = smos_m.sort_values(by='time')

smos_m = smos_m.set_index(['time', 'lat', 'lon'])

comb_s = smos_m.to_xarray()
comb_s = comb_s.resample(time='9D').mean(dim='time')

smos = comb_s.to_dataframe()
smos = smos.reset_index()
smos = smos.dropna()


file_data = path_to_file + 'smos_vod_3dm_seas_c_no_std.pkl'
overlap_m = pd.read_pickle(file_data)
overlap_m['time'] = pd.to_datetime(overlap_m['time'])
overlap_m = overlap_m.sort_values(by='time')


overlap_m = overlap_m.set_index(['time', 'lat', 'lon'])

comb = overlap_m.to_xarray() 
comb = comb.resample(time='9D').mean(dim='time')

overlap = comb.to_dataframe()
overlap = overlap.reset_index()

overlap = overlap.dropna()
overlap['sin_lon'] = np.sin(2*np.pi*overlap['lon']/360)

###########
#print('just reading existing V5')
#file_data = path_to_file + 'overlap_full_with_NNoutput_v5.pkl'
#output = pd.read_pickle(file_data)

#R = output['out_resid'].corr(output["dev_vod"])
#R2 = r2_score(output['out_resid'], output["dev_vod"])
#rmse = mean_squared_error(output['dev_vod'], output["out_resid"], squared=False)
#print('Correlation R between the NN resid SM signal and the target resid SM signal for the whole period 2015-2020')
#print(R)
#print('R^2 between the NN SM resid and the target resid  SM signal for the whole period 2015-2020')
#print(R2)
#print('RMSE between the NN resid  SM  and the target resid SM signal for the whole period 2015-2020')
#print(rmse)




#################
#print('reloading model')
#model = keras.models.load_model(path_to_save_nn +'NNsmapsmos_ver6.h5')


#bs = 2048

#X_ovelap_full = overlap[[ 'lat', 'lon',  'dev_H', 'dev_V']]
#X_ovelap_full = X_ovelap_full.values.astype(float)
#Y_ovelap_full = overlap['dev_vod'].values.astype(float)

#features_smos = smos[[ 'lat', 'lon',  'dev_H', 'dev_V']]
#features_smos = features_smos.dropna() # don't do drop na before subsetting only 1 angle, otherwise will wrongly drop data due to many na in other angles.

#X_smos = features_smos.values.astype(float)

#scale =preprocessing.StandardScaler()
# fit scaler to all possible smos data
#scalerX = scale.fit(X_smos)

#X_ovelap_full = scalerX.transform(X_ovelap_full)
#X_smos = scalerX.transform(X_smos)



#print('NN performance metrics')

#y_f = model.predict(X_ovelap_full, batch_size=bs, verbose=0)
#y_f = np.asarray(y_f).reshape(-1)
#overlap['out_resid'] = y_f
def perf_m(true_val,pred, print_mes):

    R = np.corrcoef(pred, true_val)

    R2_op = r2_score( true_val, pred)

    rmse = mean_squared_error(true_val, pred, squared=False)
    print('Correlation R between the NN resid SM signal and the target resid SM signal for the ' + print_mes)
    print(R)
    print('R^2 between the NN SM resid and the target resid  SM signal for the ' + print_mes)

    
    print(R2_op)

    print('RMSE between the NN resid  SM  and the target resid SM signal for the ' + print_mes)
    print(rmse)
    return print('  ')


train = overlap.sample(frac=0.8)
test = overlap.drop(train.index)

# features are coordinates and TB residuals
train_dataset = train[[ 'lat', 'sin_lon',  'dev_H', 'dev_V']]
test_dataset = test[[ 'lat', 'sin_lon',  'dev_H', 'dev_V']]

X = train_dataset.values.astype(float)
# target is SM residuals
Y = train['dev_vod'].values.astype(float)

Xt = test_dataset.values.astype(float)
Yt = test['dev_vod'].values.astype(float)

X_ovelap_full = overlap[[ 'lat', 'sin_lon',  'dev_H', 'dev_V']]
X_ovelap_full = X_ovelap_full.values.astype(float)
Y_ovelap_full = overlap['dev_vod'].values.astype(float)


smos['sin_lon'] = np.sin(2*np.pi*smos['lon']/360)


features_smos = smos[[ 'lat', 'sin_lon',  'dev_H', 'dev_V']]
features_smos = features_smos.dropna() # don't do drop na before subsetting only 1 angle, otherwise will wrongly drop data due to many na in other angles.

X_smos = features_smos.values.astype(float)

scale =preprocessing.RobustScaler()
# fit scaler to all possible smos data
scalerX = scale.fit(X_smos)

X = scalerX.transform(X)
Xt = scalerX.transform(Xt)
X_ovelap_full = scalerX.transform(X_ovelap_full)
X_smos = scalerX.transform(X_smos)

# NN parameters
bs = 2048
num_of_units = 1050
adm = optimizers.Adam(lr=0.0009)
epoch=200
#print('mae - 512, 0.004, 60 ')
inputs = tf.keras.layers.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(units=512,  activation='relu')(inputs)
x = tf.keras.layers.Dense(units=600,  activation='relu')(x)
x = tf.keras.layers.Dense(units=700,  activation='relu')(x)
x = tf.keras.layers.Dense(units=800, activation='relu')(x)
x = tf.keras.layers.Dense(units=900,  activation='relu')(x)
x = tf.keras.layers.Dense(units=1000,  activation='relu')(x)
x = tf.keras.layers.Dense(units=1100,  activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# was 'mean_squared_error'
#try:
#    model.compile(loss='mean_absolute_error', optimizer=adm, metrics=['mae'])
#except:
#    print('just loss didnt work')
#loss_try = tf.keras.losses.LogCosh()
#model.compile(loss='mean_squared_error', optimizer=adm, metrics=['mse'])
model.compile(loss=tf.keras.losses.LogCosh(), optimizer=adm, metrics=['logcosh'])


history = model.fit(X, Y, epochs=epoch, batch_size=bs, validation_split=0.2, verbose=2)

model.save(path_to_save_nn +'NNsmapsmos_ver9.h5')

#print(model.summary())

print('as 6, small to large, v9')

y = model.predict(X, batch_size=bs, verbose=0)
y = np.asarray(y).reshape(-1)

perf_m(Y, y, ' TRAINING')
print('target mean')
print(np.mean(Y))
print('pred mean')
print(np.mean(y))

y_t = model.predict(Xt, batch_size=bs, verbose=0)
y_t = np.asarray(y_t).reshape(-1)

perf_m(Yt, y_t, 'test')
print('target mean')
print(np.mean(Yt))
print('pred mean')
print(np.mean(y_t))


y_f = model.predict(X_ovelap_full, batch_size=bs, verbose=0)
y_f = np.asarray(y_f).reshape(-1)

perf_m(Y_ovelap_full, y_f, 'test')
print('target mean')
print(np.mean(Y_ovelap_full))
print('pred mean')
print(np.mean(y_f))


#model.save(path_to_save_nn +'NNsmapsmos_ver8.h5')

quit()



overlap['out_resid'] = y_f
overlap['nn_out'] = overlap['vod_seas_med'] + overlap['out_resid']
#y_ovelap_full = overlap['nn_out'].values.astype(float)




overlap_ver = overlap[['lat', 'lon', 'time', 'nn_out']].copy()

overlap_ver.to_pickle(path_to_save_output + 'NNsmossmap_overlap_output_ver7.pkl', protocol = 4)
overlap.to_pickle(path_to_save_output + 'overlap_full_with_NNoutput_v7.pkl', protocol = 4)

y_smos = model.predict(X_smos, batch_size=bs, verbose=0)
y_smos = np.asarray(y_smos).reshape(-1)
smos['output_resid'] = y_smos
smos['nn_out'] = smos['vod_seas_med']+smos['output_resid']

smos.to_pickle(path_to_save_output + 'NNsmossmap_smos_output_ver7.pkl', protocol = 4)



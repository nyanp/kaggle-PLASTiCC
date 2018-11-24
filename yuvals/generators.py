#should be placed in function folder

import os 
import sys
import time
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd 
import keras
import pdb
from keras.utils import to_categorical
from multiprocessing import Process, Queue, current_process, freeze_support


def load_process(df_meta,timeseries_file,in_queue,queue_length,chunksize):
    extras=[]
    for chunk in pd.read_csv(timeseries_file, chunksize=chunksize):
        first_id=chunk.head(1)['object_id'].values[0]
        last_id=chunk.tail(1)['object_id'].values[0]
        select=chunk['object_id'].isin([first_id,last_id])
        extras.append(chunk[select].copy())
        mid_chunk=chunk[~select].sort_values(['object_id','mjd']).copy()
        chunk_meta=df_meta[df_meta.object_id.isin(mid_chunk.object_id.tolist())].sort_values('object_id').copy()
        if chunk_meta.shape[0]>0:
            in_queue.put([mid_chunk,chunk_meta])
    mid_chunk=pd.concat(extras)
    chunk_meta=df_meta[df_meta.object_id.isin(mid_chunk.object_id.unique())].sort_values('object_id').copy()
    in_queue.put((mid_chunk,chunk_meta))
    time.sleep(5)
    for i in range(queue_length):
        in_queue.put('STOP')
    print('bye bye')

def prepare_process(meta_cols,col_weights,real_targets,in_queue,out_queue,length=128):
    for dfs, dfm in iter(in_queue.get, 'STOP'):
        tar_dict=dict(zip(real_targets,range(len(real_targets))))
        meta=dfm[meta_cols].fillna(0).values*col_weights
        sw=dfm.hostgal_photoz.values
        timeseries0,timeseries,void=inter_mjd_det(dfm,dfs,length,verbose=False)
        out_queue.put((dfm.object_id,timeseries,timeseries0,void,meta,sw))
    out_queue.put('STOPED')
    print('bye')

def calculate_inputs(df_times,df_meta,meta_cols,col_weights,real_targets,length=256,aug=None,return_y=True):
    tar_dict=dict(zip(real_targets,range(len(real_targets))))
    tsf=df_times[df_times.object_id.isin(df_meta.object_id.tolist())]
    if aug is None:
        ts=tsf.copy()
    else:
        ts=tsf.sample(frac=aug['sample']).sort_values(['object_id','mjd']).copy()
    
    dfm=df_meta.sort_values('object_id').copy()
    meta=dfm[meta_cols].fillna(0).values*col_weights
    if return_y:
        ty=to_categorical(df_meta.target.map(tar_dict))
    sw=dfm.hostgal_photoz.values
     #sample(frac=0.95).sort_values(['object_id','mjd'])
    timeseries0,timeseries,void=inter_mjd_det(dfm,ts,length,verbose=False)
#    timeseries=inter_mjd_det(df_meta, ts,256,detected=True, verbose=False)
#    void=get_void_timeseries(df_meta, ts, 256,verbose=False)/100
    if not aug is None:
        sh=np.random.randint(0,aug['shift'],timeseries.shape[0])
        sk=np.random.normal(0,aug['skew'],(timeseries.shape[0],timeseries.shape[2]))
        timeseries=permute(timeseries,sh)
        timeseries=skew(timeseries,sk)
        timeseries0=permute(timeseries0,sh)
        timeseries0=skew(timeseries0,sk)
        void=permute(void,sh)
    if return_y:
        return timeseries,timeseries0,void,meta,sw,ty
    else:
        return timeseries,timeseries0,void,meta,sw
        


def object_resample(df_obj,num_samples=256,detected=False,weight=0):
    new_time_index=df_obj.mjd.min()+(df_obj.mjd.max()-df_obj.mjd.min())/(num_samples-1)*np.arange(num_samples)
    result=np.zeros((num_samples,6))
    for i in range(6):
        select=df_obj.passband==i  
        dd=df_obj[select]
        result[:,i]=np.interp(new_time_index,dd.mjd.values,(dd.flux*(((dd.detected==1) | (not detected))+weight)/(1+weight)).values)
    return new_time_index,result

def object_resample_err(df_obj,num_samples=256):
    new_time_index=df_obj.mjd.min()+(df_obj.mjd.max()-df_obj.mjd.min())/(num_samples-1)*np.arange(num_samples)
    result=np.zeros((num_samples,6))
    for i in range(6):
        select=df_obj.passband==i  
        dd=df_obj[select]
        result[:,i]=np.interp(new_time_index,dd.mjd.values,dd.flux_err)
    return new_time_index,result

def one_hot_encode(value,targets,return_type='float32'):
    return (value==targets).astype(return_type)

def create_XY(df_timeseries,df_meta,Targets=None,num_samples=256,detected=False,weight=0,return_type=None):
        X_size=df_meta.shape[0]
        timeseries= np.zeros((X_size,num_samples,6))
        meta = np.zeros((X_size,6))
        if return_type!='test':
            y = np.zeros((X_size,Targets.shape[0]))
        for j , row in tqdm_notebook(enumerate(df_meta.itertuples()),total=df_meta.shape[0]):
            df_obj=df_timeseries[df_timeseries.object_id==row.object_id]
            if return_type=='error':
                _,timeseries[j,...]=object_resample_err(df_obj,num_samples=num_samples)
            else: 
                _,timeseries[j,...]=object_resample(df_obj,num_samples=num_samples,detected=detected,weight=weight)
            
            max_val=np.log(df_obj.flux.abs().max())
            meta[j,...]=np.array([row.ddf,row.hostgal_specz,row.hostgal_photoz,row.distmod/50.0,row.mwebv,max_val])
            if (return_type!='test') & (return_type!='error'):
                y[j,:]=one_hot_encode(row.target,Targets)
        meta[np.isnan(meta)]=0
        if return_type=='test':
            return timeseries,meta
        elif return_type=='timeseries':
            return timeseries
        elif return_type=='error':
            return timeseries
        else:
            return timeseries,meta,y

def permute(timeseries_arr,shift):
    l=timeseries_arr.shape[0]
    res=np.zeros_like(timeseries_arr)
    for i in range(l):
        res[i,...]=np.roll(timeseries_arr[i,...],shift[i],axis=0)
    return res    

def skew(timeseries_arr,vals):
    l=timeseries_arr.shape[0]
    c=timeseries_arr.shape[2]
    res=np.zeros_like(timeseries_arr)
    for i in range(l):
        for j in range(c):
            res[i,:,j]=timeseries_arr[i,:,j]*(1+vals[i,j])
    return res

def skew_db(timeseries_arr,vals):
    l=timeseries_arr.shape[0]
    c=timeseries_arr.shape[2]
    res=np.zeros_like(timeseries_arr)
    for i in range(l):
        for j in range(c):
            res[i,:,j]=timeseries_arr[i,:,j]*np.power(10,vals[i,j]/20)
    return res

def add_noise(timeseries_arr,noise):
    res=timeseries_arr+noise*(timeseries_arr!=0)
    return res

# tr is training timeseries or test timeseries, 
# tr_m is training metadata or test metadata, 
# length = 256
def inter_mjd(tr_m, tr, length,verbose=False):
    new_df = pd.DataFrame(np.repeat(tr_m['object_id'], 6), columns = ['object_id']).reset_index(drop = True).reset_index()
    new_df['passband'] = np.repeat(np.arange(6)[np.newaxis, ...], tr_m.shape[0], axis = 0).ravel()
    tr_cp = pd.merge(tr, new_df, how = 'outer', on = ['object_id', 'passband']).rename(columns = {'index' : 'ob_p'})

    gp_mjd = tr_cp.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()

    merged = pd.merge(tr_cp, gp_mjd, how = 'left', on = 'object_id')

    merged['mm_scaled_mjd'] = (length - 1) * (merged['mjd'] - merged['mjd_min'])/(merged['mjd_max'] - merged['mjd_min'])

    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    merged=merged.sort_values(['object_id','mjd'])
    unstack = merged[['ob_p', 'mm_scaled_mjd', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()

    mjd_uns = unstack['mm_scaled_mjd'].values[..., np.newaxis]
    flux_uns = unstack['flux'].values[..., np.newaxis]
    mjd_flux = np.concatenate((mjd_uns, flux_uns), axis = 2)
    nan_masks = ~np.isnan(mjd_flux)[:, :, 0]

    x = np.arange(length)
    X = np.zeros((mjd_flux.shape[0], x.shape[0]))
    if verbose:
        t=tqdm_notebook(range(mjd_flux.shape[0]))
    else:
        t=range(mjd_flux.shape[0])
    for i in t:
        intp = np.interp(x, mjd_flux[i][:, 0][nan_masks[i]], mjd_flux[i][:, 1][nan_masks[i]])
        X[i] = intp
    X_reshaped = X.reshape(tr_m.shape[0], 6, length).astype(np.float32)
    X_per_object = X_reshaped.reshape(X_reshaped.shape[0], length * 6)
    res = (X_reshaped)/(1e-2 + X_per_object.std(axis = 1)[..., np.newaxis, np.newaxis])
    return np.transpose(np.nan_to_num(res),(0,2,1))

def inter_mjd_det(tr_m, tr, length,verbose=False):
    MJD_DIFF_MAX = 1094.0653999999995

    tr_cp = tr.copy()
    objs=tr.object_id.unique()
    objs_p=np.concatenate([10*objs+d for d in range(6)])
    tr_cp['ob_p']=tr.object_id*10+tr.passband
    rem=set(objs_p).difference(set(tr_cp['ob_p'].values))
    if len(rem)>0:
        mmjd=tr_cp.mjd.mean()
        data=np.zeros((len(rem),7))
        rml=np.array(list(rem))
        data[:,0] = (rml/10).astype('int')
        data[:,1] = np.ones(len(rem))*mmjd 
        data[:,2]= (rml-data[:,0]*10).astype('int')
        data[:,6]=rml
        df_rem=pd.DataFrame(data=data, columns=['object_id','mjd','passband','flux','flux_err','detected','ob_p'])
        tr_cp=pd.concat([tr_cp,df_rem],ignore_index=True).sort_values(['object_id','mjd']).reset_index(drop=True)
            
    tr_cp['flux_multi_det'] = tr_cp['flux'] * tr_cp['detected']
    tr_cp['flux_multi'] = tr_cp['flux']
    gp_mjd = tr_cp.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()
    
    merged = pd.merge(tr_cp, gp_mjd, how = 'left', on = 'object_id')

    merged['mm_scaled_mjd'] = (length - 1) * (merged['mjd'] - merged['mjd_min'])/(merged['mjd_max']-merged['mjd_min'])

    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    merged=merged.sort_values(['object_id','mjd'])
    unstack = merged[['ob_p', 'mm_scaled_mjd', 'flux_multi_det','flux_multi', 'cc']].set_index(['ob_p', 'cc']).unstack()

    mjd_uns = unstack['mm_scaled_mjd'].values[..., np.newaxis]
    flux_uns = unstack['flux_multi'].values[..., np.newaxis]
    flux_det_uns = unstack['flux_multi_det'].values[..., np.newaxis]
    mjd_flux = np.concatenate((mjd_uns, flux_uns, flux_det_uns), axis = 2)
    nan_masks = ~np.isnan(mjd_flux)[:, :, 0]

    x = np.arange(length)
    X = np.zeros((mjd_flux.shape[0], x.shape[0]))
    X_det = np.zeros((mjd_flux.shape[0], x.shape[0]))
    if verbose:
        t=tqdm_notebook(range(mjd_flux.shape[0]))
    else:
        t=range(mjd_flux.shape[0])
    for i in t:
        if nan_masks[i].any():
            X[i] = np.interp(x, mjd_flux[i][:, 0][nan_masks[i]], mjd_flux[i][:, 1][nan_masks[i]])
            X_det[i] = np.interp(x, mjd_flux[i][:, 0][nan_masks[i]], mjd_flux[i][:, 2][nan_masks[i]])
        else:
            X[i] = np.zeros_like(x)
            X_det[i] = np.zeros_like(x)

    unstack_v = merged[['ob_p', 'mm_scaled_mjd', 'cc']].set_index(['ob_p', 'cc']).unstack()
    X_void = np.zeros((unstack_v.shape[0], length))

    if verbose:
        t=tqdm_notebook(range(length))
    else:
        t=range(length)
    for i in t:
        X_void[:, i] = np.abs((unstack_v - i)).min(axis = 1).fillna(500)
         
    X_reshaped = X.reshape(tr_m.shape[0], 6, length).astype(np.float32)
    X_per_object = 1e-2 +X_reshaped.reshape(X_reshaped.shape[0], length * 6).std(axis = 1)[..., np.newaxis, np.newaxis].copy()
    res = (X_reshaped)/X_per_object
    X_reshaped = X_det.reshape(tr_m.shape[0], 6, length).astype(np.float32)
    res_det=(X_reshaped)/X_per_object
    X_reshaped = X_void.reshape(tr_m.shape[0], 6, length).astype(np.float32)
    coef = ((gp_mjd['mjd_max'] - gp_mjd['mjd_min'])/(length - 1)).values
    res_void = X_reshaped * coef[..., np.newaxis, np.newaxis]/100
    
    return np.transpose(np.nan_to_num(res),(0,2,1)),np.transpose(np.nan_to_num(res_det),\
                                                                 (0,2,1)),np.transpose(np.nan_to_num(res_void),(0,2,1))

def get_void_timeseries(tr_m, tr, length,verbose=False):
    new_df = pd.DataFrame(np.repeat(tr_m['object_id'], 6), columns = ['object_id']).reset_index(drop = True).reset_index()
    new_df['passband'] = np.repeat(np.arange(6)[np.newaxis, ...], tr_m.shape[0], axis = 0).ravel()
    tr_cp = pd.merge(tr, new_df, how = 'left', on = ['object_id', 'passband']).rename(columns = {'index' : 'ob_p'})
    gp_mjd = tr_cp.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()
    merged = pd.merge(tr_cp, gp_mjd, how = 'left', on = 'object_id')
    merged['mm_scaled_mjd'] = (length - 1) * (merged['mjd'] - merged['mjd_min'])/(merged['mjd_max'] - merged['mjd_min'])
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    unstack = merged[['ob_p', 'mm_scaled_mjd', 'cc']].set_index(['ob_p', 'cc']).unstack()
    
    res = np.zeros((unstack.shape[0], length))
    if verbose:
        t=tqdm_notebook(range(length))
    else:
        t=range(length)
    for i in t:
        res[:, i] = np.abs((unstack - i)).min(axis = 1)
        
    res_reshaped = res.reshape(tr_m.shape[0], 6, length).astype(np.float32)
    coef = ((gp_mjd['mjd_max'] - gp_mjd['mjd_min'])/(length - 1)).values
    res_multi = res_reshaped * coef[..., np.newaxis, np.newaxis]
    return np.transpose(res_multi,(0,2,1))

class data_generator(keras.utils.Sequence):

    def __init__(self,df_timeseries,df_meta,batch_size=32,num_samples=256,detected=False,weight=0,return_type=None,shuffle=True):
        self.df_timeseries=df_timeseries
        self.df_meta=df_meta
        self.batch_size=batch_size
        self.num_samples=num_samples
        self.detected=detected
        self.weight=weight
        self.return_type=return_type
        self.shuffle=shuffle
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(self.df_meta.shape[0]/self.batch_size))

    def on_epoch_end(self):
        self.timeseries= np.zeros((self.batch_size,self.num_samples,6))
        self.meta = np.zeros((self.batch_size,6))
        self.y = np.zeros((self.batch_size,Targets.shape[0]))
        self.df_sh=self.df_meta
        if self.shuffle:
            self.df_sh=self.df_sh.sample(frac=1)



    def __getitem__(self,index):
        if ((self.df_sh.shape[0]<(index+1)*self.batch_size) and (self.return_type!='test')):
            self.on_epoch_end()
        df_temp=self.df_sh.iloc[index*self.batch_size:].head(self.batch_size).copy()
        return self._data_generator(df_temp)

    def _data_generator(self,df):
        #pdb.set_trace()
        batch_size=df.shape[0]
        self.timeseries= np.zeros((batch_size,self.num_samples,6))
        self.meta = np.zeros((batch_size,6))
        self.y = np.zeros((batch_size,Targets.shape[0]))
        for j , row in enumerate(df.itertuples()):
            df_obj=self.df_timeseries[self.df_timeseries.object_id==row.object_id]
            _,self.timeseries[j,...]=object_resample(df_obj,num_samples=self.num_samples,detected=self.detected,weight=self.weight)
            max_val=np.log(df_obj.flux.abs().max())
            self.meta[j,...]=np.array([row.ddf,row.hostgal_specz,row.hostgal_photoz,row.distmod/50.0,row.mwebv,max_val])
            if self.return_type!='test':
                self.y[j,:]=one_hot_encode(row.target,Targets)
        self.meta[np.isnan(self.meta)]=0
        if self.return_type=='test':
            return [self.timeseries,self.meta]
        else:
            return [self.timeseries,self.meta],self.y




import math
import time
import os
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from pathlib import Path
from fbprophet import Prophet
import gc
import concurrent
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import datetime

# scoring
from functools import reduce,partial
from scipy.signal import find_peaks


Path.ls = lambda x: [o.name for o in x.iterdir()]

class Predictor:
    def __init__(self, images, n_cls=10, im_size=64, predict_len=180, im_mask=None):
        self.dates = []
        self.files = []
        self.n_cls = n_cls
        self.im_size = im_size
        self.im_mask = im_mask
        self.predict_len = predict_len
        
        self.ts_cls = [] # clusters
        self.ts = [] # time series of each pixel
        self.mean_ts = [] # time series mean of each cluster
        self.prediction = pd.DataFrame({})

        for date, file in images.items():
            if type(date) == 'str':
                self.dates.append(date)
            else:
                self.dates.append(datetime.datetime.fromtimestamp(date / 1000))
            self.files.append(file)
            
        self.df = pd.DataFrame({
                'ds': pd.to_datetime(self.dates),
            }).sort_values(by='ds')
        self.periods = (self.df.ds.max() - self.df.ds.min()).days + 1
        self.full_ds = pd.date_range(str(self.df.ds.min()), periods=self.periods, freq='D')
        
    def _split(self, df, pct):
        '''
            split dataframe into train and validation sets
        '''
        len_train = int(len(df) * pct)
        train_df = df.iloc[0:len_train]
        val_df = df.iloc[len_train:-1]
        return train_df, val_df
    
    def _tslize(self):
        '''
            convert images to time series
        '''
        ims = []
        for file in self.files:
            im_tile = Image.open(file)
            if self.im_mask is None:
                ims.append(im_tile)
            else:
                data_mask = (np.array(self.im_mask)[:, :, 3] / 255).reshape((im_tile.size[1],im_tile.size[0],1))
                data_im = np.array(im_tile)
                ims.append(Image.fromarray(data_im*data_mask.astype('uint8')))
            # ims.append(im_tile)
        data = np.array([np.array(img.resize((64, 64))) for img in ims])
        # Use mean of RGB as pixel value
        ts = np.array(data).mean(3)
        ts = ts.reshape((ts.shape[0], -1)).transpose()
        gc.collect()
        return ts

    def _clusterify(self, ts):
        '''
            clusterify time series of pixels
        '''
        kmeans = KMeans(self.n_cls, random_state=0).fit(ts)
        return kmeans.predict(ts)
    
    def _predict(self, vals, i, df, periods, full_ds, draw=False, val=False):
        if vals.mean() < 1: return None
        df['y'] = [np.nan if x > 150 else x for x in vals]
        new_df = pd.DataFrame({'ds': full_ds}).merge(df, how='left', on='ds')
        new_df['y'] = new_df.set_index('ds').y.interpolate(method='time').tolist()
        if val:
            t_df, v_df = split(new_df, 0.9) 
        else:
            t_df = new_df

        new_v_df = pd.DataFrame({
            'ds': pd.date_range(str(t_df.ds.max() + pd.Timedelta(1, 'D')), periods=self.predict_len, freq='D')
        }) 
        m = Prophet()
        m.fit(t_df)
        forecast = m.predict(new_v_df)
        if draw:
            plt.figure()
            ax = plt.subplot(1,1,1)
            ax.plot(df.ds, df.y)
            ax.plot(forecast.ds, forecast.yhat, '-r')

        scores, ups, downs = tensor2score(
            t_df.ds.tolist() + new_v_df.ds.tolist(),
            [np.array(x) for x in t_df.y.tolist() + forecast.yhat.tolist()])
        return scores, ups, downs
    
    def draw_clusters(self):
        if self.ts_cls == []: return
        im = []
        for i in range(len(self.ts_cls)):
            im.append((self.ts_cls[i]*20, 0, 0))
        newim = Image.new('RGB', (self.im_size,self.im_size))
        newim.putdata(im)

        plt.figure(figsize=(6, 6))
        plt.imshow(newim)
    
    def run(self, n_workers = 1, output_dir='./output'):
        print("converting image to time series")
        self.ts = self._tslize()
        
        print("clusterify time series")
        self.ts_cls = self._clusterify(self.ts)
        self.mean_ts = [self.ts[self.ts_cls==i].mean(0) for i in range(self.n_cls)]
        
        print("start predicting")
        predict_runner = partial(self._predict, df=self.df, periods=self.periods,full_ds=self.full_ds)
        start_time = time.time()
        pred = parallel(predict_runner, self.mean_ts, n_workers)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        print("scoring")
        res_df = pd.DataFrame({'date':[]})
        for i, y in enumerate(pred):
            if y == None:
                res_df['cls_'+str(i)]=[]
                continue
            temp = y[0].reset_index()[['date', 'tgt_score']].rename({'tgt_score': 'cls_'+str(i)}, axis=1)
            res_df = res_df.merge(temp, how='right', on=['date'])
            
        res_df = self.df[['ds']].merge(res_df, how='left', left_on='ds',right_on='date').drop(columns='date')
        self.prediction = res_df
        return res_df
        
    def gen_heatmaps(self):
        op_files = []
        heatmaps = []
        xs = ys = np.linspace(0, self.im_size - 1, self.im_size)
        xv, yv = np.meshgrid(xs, ys, indexing='ij')
        for _, row in self.prediction.iterrows():
            val = [
                row[f'cls_{self.ts_cls[i]}'] if row[f'cls_{self.ts_cls[i]}'] == row[f'cls_{self.ts_cls[i]}'] else -1 
                for i in range(self.im_size * self.im_size)
            ]
            # im_data = [[255, 0, 0, int(255*v) if v == v else 0] for v in val]
            # img_output = Image.fromarray(np.array(im_data).reshape((self.im_size, self.im_size, 4)).astype('uint8'))
            # img_output.save(f'{output_dir}/{str(row.ds)[:10]}.png')
            # op_files.append(f'{output_dir}/{str(row.ds)[:10]}.png')
            time_stamp = row.ds.value // 10 ** 6
            heatmaps.append({
                'timestamp': time_stamp,
                'heatmap': np.stack([xv, yv, np.array(val).reshape(self.im_size, self.im_size)], 2)
            })
        return heatmaps
    
def num_cpus()->int:
    "Get number of cpus"
    try:                   return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count()
def parallel(func, arr, max_workers, leave=False):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    if max_workers<2: results = [func(o,i) for i,o in enumerate(arr)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            results = []
            for f in concurrent.futures.as_completed(futures): 
                results.append(f.result())
    if any([o is not None for o in results]): return results 

def image_score(img_tensor, val_thred=230, ratio_thred=0.1):
    # if cloud ignore, return None
    w_ratio = (img_tensor > val_thred).sum() / (img_tensor != 0).sum()
    return img_tensor.mean() if w_ratio < ratio_thred else -1
def find_ts_peaks(ser, nn_range='120 days'):
    peaks, _ = find_peaks(ser, height=ser.quantile(0.75))
    prev = None
    res = []
    for dt in ser.iloc[peaks].index.sort_values():
        if prev is None:
            res.append(dt)
        else:
            if dt - prev > pd.Timedelta(nn_range):
                res.append(dt)
        prev = dt
    return res
def tensor2score(ts, datas):
    df = (
        pd.DataFrame(
            {
                'date': pd.to_datetime(ts),
                "IMAGE_SCORE": [image_score(data) for data in datas]
            }
        )
        .set_index('date')
        .sort_index()
    )
    df.loc[df.IMAGE_SCORE == -1, 'IMAGE_SCORE'] = np.NaN
    df = df.assign(
        IMAGE_SCORE_INT=df.IMAGE_SCORE.interpolate(method='time')
    )
    df = df.assign(
        IMAGE_SCORE_SMTH=df.IMAGE_SCORE_INT.ewm(span=10).mean(),
    )
    ups = find_ts_peaks(df.IMAGE_SCORE_SMTH, nn_range='120 days')
    downs = find_ts_peaks(df.IMAGE_SCORE_SMTH*(-1), nn_range='120 days')
    df = df.assign(tgt_phase=np.NaN)
    df.loc[ups, 'tgt_phase'] = 'growning'
    df.loc[downs, 'tgt_phase'] = 'croping'
    df = df.assign(
        tgt_phase=df.tgt_phase.fillna(method='ffill')
    )

    df = df.assign(tgt_score=np.NaN)
    df.loc[ups, 'tgt_score'] = 0.
    df.loc[downs, 'tgt_score'] = 1.
    df = df.assign(
        tgt_score=df.tgt_score.interpolate(method='time')
    )
    if (ups + downs):
        df.loc[max(ups+downs):, 'tgt_score'] = np.NaN
        df.loc[max(ups+downs):, 'tgt_phase'] = np.NaN
    return df, ups, downs

def draw_score(df, ups, downs, ax):
    if ax == None:     
        _, ax = plt.subplots(figsize=(12, 10))
    ax.plot(df.IMAGE_SCORE_SMTH, label='Image raw')
    ax.plot(df.loc[downs].index, df.loc[downs]['IMAGE_SCORE_SMTH'], "x", label='Downs')
    ax.plot(df.loc[ups].index, df.loc[ups]['IMAGE_SCORE_SMTH'], "x", label='Peaks')
    ax.plot(df.tgt_score*100, label='Mature score')
    ax.legend();
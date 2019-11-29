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
from functools import reduce, partial, lru_cache
from scipy.signal import find_peaks


Path.ls = lambda x: [o.name for o in x.iterdir()]

class Predictor:
    def __init__(self, images, n_cls=10, im_size=32, predict_len=180, im_mask=None):
        self.dates = []
        self.files = []
        self.n_cls = n_cls
        self.im_origin_size = 512
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
        self.im_origin_size = np.mean([ims[0].size[0], ims[0].size[1]])
        data = np.array([np.array(img.resize((32,32))) for img in ims])
        # Use mean of RGB as pixel value
        ts = np.array(data).mean(3)
        ts = ts.reshape((ts.shape[0], -1)).transpose()
        gc.collect()
        return ts

    def _clusterify(self, ts):
        '''
            clusterify time series of pixels
        '''
        print(self.im_origin_size)
        self.n_cls = int(self.im_origin_size / 20)
        self.n_cls = np.max([self.n_cls, 20])
        self.n_cls = np.min([self.n_cls, 400])
        self.n_cls = 32
        kmeans = KMeans(self.n_cls, random_state=0).fit(ts)
        return kmeans.predict(ts)
    
    def _predict(self, vals, i, df, periods, full_ds, draw=False, val=False):
        if vals.mean() < 10: 
            return None
        df['y'] = [np.nan if x > 150 else x for x in vals]

        scores, ups, downs = tensor2score(
            df.ds, [np.array(x) for x in df.y.tolist()])

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

        min_cls = np.array(self.mean_ts).mean(1).argmin()
        self.mean_ts[min_cls] = np.zeros(len(self.mean_ts[0]))
        
        print("start predicting")
        predict_runner = partial(self._predict, df=self.df, periods=self.periods,full_ds=self.full_ds)
        start_time = time.time()
        pred = parallel(predict_runner, self.mean_ts, n_workers)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        score_start_time = time.time()
        res_df = pd.DataFrame({'date':[]})
        for i, y in enumerate(pred):
            if y == None:
                res_df['cls_'+str(i)]=[]
                continue
            temp = y[0].reset_index()[['date', 'tgt_score']].rename({'tgt_score': 'cls_'+str(i)}, axis=1)
            res_df = res_df.merge(temp, how='right', on=['date'])
        print("----------%s seconds " % (time.time() - score_start_time)) 
        res_df = self.df[['ds']].merge(res_df, how='left', left_on='ds',right_on='date').drop(columns='date')
        self.prediction = res_df
        return res_df
    def gen_heatmaps(self):
        start_time = time.time()
        print('generating heat map')
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
                'heatmap': (np.stack([xv, yv, np.array(val).reshape(self.im_size, self.im_size)], 2)*100).astype('int32')/100
            })
        #line_res = {}
        #for item0 in heatmaps:
         #   ts = item0['timestamp']
          #  for item1 in item0['heatmap'].reshape(item0['heatmap'].shape[0]*item0['heatmap'].shape[1], 3).tolist():
                # print(item1)
           #     x, y, val = item1
            #    y = self.im_size - y - 1
             #   loc_str = f'{int(x)}_{int(y)}'
              #  if loc_str in line_res:
               #     pass
                #else:
                 #   line_res[loc_str] = {}
                #if "x_value" not in line_res[loc_str]: 
                   # line_res[loc_str]["x_value"] = []
                #if "y_value" not in line_res[loc_str]: 
                 #   line_res[loc_str]["y_value"] = []
                #line_res[loc_str]["x_value"].append(ts)
                #line_res[loc_str]["y_value"].append(val)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        return heatmaps,{}
    
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
    diffs = []
    peaks = sorted(ups + downs)
    for i, t in enumerate(peaks):
        if i > 0:
            diffs.append((t - peaks[i - 1]).days)
    ewm = pd.Series(diffs).ewm(com=0.9).mean()
    last_t = (df.index[-1] - np.max(ups+downs)).days

    df.iloc[-1, -1] = last_t / ewm.iloc[-1]
    df = df.assign(
        tgt_score=df.tgt_score.interpolate(method='time')
    )
    #if (ups + downs):
    #    df.loc[max(ups+downs):, 'tgt_score'] = np.NaN
    #    df.loc[max(ups+downs):, 'tgt_phase'] = np.NaN
    return df, ups, downs

def draw_score(df, ups, downs, ax):
    if ax == None:     
        _, ax = plt.subplots(figsize=(12, 10))
    ax.plot(df.IMAGE_SCORE_SMTH, label='Image raw')
    ax.plot(df.loc[downs].index, df.loc[downs]['IMAGE_SCORE_SMTH'], "x", label='Downs')
    ax.plot(df.loc[ups].index, df.loc[ups]['IMAGE_SCORE_SMTH'], "x", label='Peaks')
    ax.plot(df.tgt_score*100, label='Mature score')
    ax.legend();

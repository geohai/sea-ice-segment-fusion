import argparse
import configparser
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import seaborn as sns
import torch
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, jaccard_score)
from torch import nn
from torchvision.transforms import Normalize


def evaluate(config):

    dir_out = os.path.normpath(config['io']['dir_out'])
    model_path = os.path.normpath(config['io']['model_path'])

    test_rasters = [os.path.normpath(f) for f in config['io']['test_rasters'].split('\n')]
    test_label_rasters = [os.path.normpath(f) for f in config['io']['test_label_rasters'].split('\n')]

    if 'test_label_rasters_sec' in config['io'].keys():
        test_label_rasters_sec = [os.path.normpath(f) for f in config['io']['test_label_rasters_sec'].split('\n')]
    else:
        test_label_rasters_sec = ['None' for _ in test_label_rasters]

    mean = [float(val) for val in config['datamodule']['mean'].split(',')]
    std = [float(val) for val in config['datamodule']['std'].split(',')]
    ignore_index = int(config['datamodule']['ignore_index'])

    norms = {}
    norms['input'] = Normalize(mean, std)

    model = torch.jit.load(model_path)
    model.eval()

    # save configuration file:
    with open(os.path.join(dir_out, 'evaluate.cfg'), 'w') as out_file:
        config.write(out_file)

    # run on test rasters:
    softmax = nn.Softmax(0)
    for idx, test_raster in enumerate(test_rasters):

        print(f'Using raster {test_raster}...', end=' ')
        start_time = time.perf_counter()
        raster = rioxarray.open_rasterio(test_raster, masked=True)
        x = torch.from_numpy(raster.values).unsqueeze(dim=0)

        # get input mask 
        mask = np.isnan(raster.values).any(axis=0)

        with torch.no_grad():
            x = [norms['input'](x)]
            x[0] = torch.nan_to_num(x[0])
            res = model(x)
            # compute probabilities (instead of scores):
            res = softmax(torch.squeeze(res,0))
        
        end_time = time.perf_counter()
        print(f'{(end_time-start_time)/60:.2f} minutes for model prediction...', end=' ')
        start_time = time.perf_counter()

        # cast results to numpy
        res = res.detach().numpy()

        # mark nan vals
        for band in res:
            band[mask] = np.nan

        # use raster information to populate output:
        xr_res = xr.DataArray(res, 
                              [('band', np.arange(1, res.shape[0]+1)),
                              ('y', raster.y.values),
                              ('x', raster.x.values)])
        
        xr_res['spatial_ref']=raster.spatial_ref                              
        xr_res.attrs=raster.attrs
        
        # write to file
        out_fname = os.path.join(dir_out, f'pred-{Path(test_raster).stem}.tif')
        if os.path.isfile(out_fname):
            os.remove(out_fname)
        xr_res.rio.to_raster(out_fname, dtype="float32")
        
        # write the class
        y_pred_class = res.argmax(0)
        # 241 is the no data value for uint8
        nodata = 241
        y_pred_class[mask] = nodata
        y_pred_class = np.expand_dims(y_pred_class, 0)

        xr_res = xr.DataArray(y_pred_class, 
                              [('band', [1]),
                              ('y', raster.y.values),
                              ('x', raster.x.values)])
        
        xr_res['spatial_ref']=raster.spatial_ref                              
        xr_res.attrs=raster.attrs

        xr_res.rio.write_nodata(nodata, inplace=True)
        
        out_fname = os.path.join(dir_out, f'class-{Path(test_raster).stem}.tif')
        if os.path.isfile(out_fname):
            os.remove(out_fname)
        xr_res.rio.to_raster(out_fname, dtype="uint8")

        # compute metrics if the labels are available
        if 0 <= idx < len(test_label_rasters):
            raster_y = rioxarray.open_rasterio(test_label_rasters[idx], masked=True)

            y_true = np.squeeze(raster_y.values, 0)
            y_true[y_true==ignore_index]=np.nan
            mask_y = np.isnan(y_true)

            y_pred_class = np.squeeze(y_pred_class, 0)

            # write correctly labeled pixels
            correct = np.array(y_true==y_pred_class).astype('uint8')
            correct[mask_y] = nodata

            xr_res = xr.DataArray(np.expand_dims(correct, 0), 
                                [('band', [1]),
                                ('y', raster.y.values),
                                ('x', raster.x.values)])
            
            xr_res['spatial_ref']=raster.spatial_ref                              
            xr_res.attrs=raster.attrs

            xr_res.rio.write_nodata(nodata, inplace=True)

            out_fname = os.path.join(dir_out, f'correct-prim-{Path(test_raster).stem}-vs-{Path(test_label_rasters[idx]).stem}.tif')
            if os.path.isfile(out_fname):
                os.remove(out_fname)
            xr_res.rio.to_raster(out_fname, dtype="uint8")

            if os.path.isfile(test_label_rasters_sec[idx]):
                raster_y_sec = rioxarray.open_rasterio(test_label_rasters_sec[idx], masked=True)
                y_true_sec = np.squeeze(raster_y_sec.values, 0)
                y_true[y_pred_class==y_true_sec] = y_true_sec[y_pred_class==y_true_sec]

                y_true[y_true==ignore_index]=np.nan

                correct = np.array(y_true==y_pred_class).astype('uint8')
                correct[mask] = nodata

                xr_res = xr.DataArray(np.expand_dims(correct, 0), 
                                    [('band', [1]),
                                    ('y', raster.y.values),
                                    ('x', raster.x.values)])
                
                xr_res['spatial_ref']=raster.spatial_ref                              
                xr_res.attrs=raster.attrs

                xr_res.rio.write_nodata(nodata, inplace=True)

                out_fname = os.path.join(dir_out, f'correct-prim-or-sec-{Path(test_raster).stem}.tif')
                if os.path.isfile(out_fname):
                    os.remove(out_fname)
                xr_res.rio.to_raster(out_fname, dtype="uint8")


            with open(os.path.join(dir_out, f'metrics.txt'), 'a', encoding='utf-8') as outfile:

                if os.path.isfile(test_label_rasters_sec[idx]):
                    outfile.write(f'{Path(test_raster).stem} vs {Path(test_label_rasters[idx]).stem} or {Path(test_label_rasters_sec[idx]).stem} performance \n')
                else:
                    outfile.write(f'{Path(test_raster).stem} vs {Path(test_label_rasters[idx]).stem} performance \n')
                outfile.write(classification_report(y_true[~np.logical_or(mask, mask_y)].ravel(), 
                                                    y_pred_class[~np.logical_or(mask, mask_y)].ravel()))
                outfile.write('\n')
                outfile.write(f'Jaccard Index: \n')
                for avg in ['micro', 'macro', 'weighted']:
                    iou = jaccard_score(y_true[~np.logical_or(mask, mask_y)].ravel(), 
                                        y_pred_class[~np.logical_or(mask, mask_y)].ravel(), average=avg)
                    outfile.write(f'{avg}: {iou:.2f} \n')

                cm = confusion_matrix(y_true[~np.logical_or(mask, mask_y)].ravel(), 
                                      y_pred_class[~np.logical_or(mask, mask_y)].ravel(), 
                                      normalize='true')
                outfile.write('\n')
                outfile.write(f'Confusion Matrix: \n')
                for row in cm:
                    for col in row:
                        outfile.write(f'      {col:.2f}')
                    outfile.write('\n')
                outfile.write('\n\n')

                # save pdf 
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(y_true[~np.logical_or(mask, mask_y)].ravel(), 
                                                        y_pred_class[~np.logical_or(mask, mask_y)].ravel(), 
                                                        normalize='true', 
                                                        values_format = '.2f',
                                                        ax=ax)
                
                if os.path.isfile(test_label_rasters_sec[idx]):
                    fig.savefig(os.path.join(dir_out, f'confusion_matrix-{Path(test_raster).stem}_vs_{Path(test_label_rasters[idx]).stem}_or_{Path(test_label_rasters_sec[idx]).stem}.pdf'))
                else:
                    fig.savefig(os.path.join(dir_out, f'confusion_matrix-{Path(test_raster).stem}_vs_{Path(test_label_rasters[idx]).stem}.pdf'))
            
            y_true = None
            correct = None

        # this uses a lot of memory, delete some stuff:
        raster = None
        x = None
        res = None
        mask = None

        end_time = time.perf_counter()
        print(f'{(end_time-start_time)/60:.2f} minutes for writing files and metrics')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='eval_config.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        evaluate(config)
    
    else:
        print('Please provide a valid configuration file.')

import os
import numpy as np
import pandas as pd
import warnings 
import matplotlib.pyplot as plt
from models import latency
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

def r_squared(data, pred):
    sst = np.sum((data - data.mean())**2)
    ssr = np.sum((data - pred)**2)
    return 1-(ssr/sst)

def individual_fits(data, part, xs_g, name, column, prefix, plot=True, loo=None):
    xs = np.sort(data.contrast.unique())
    if loo is not None:
        subset = data[data.contrast != loo]
    else:
        subset = data
    
    model = latency(name, latencies=subset[column], 
                contrast=subset['contrast'])
    model.fit()
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        model.plot(part, prefix, min_contrast=3.5, max_contrast=93.5, ax=ax, fig=fig)

    mask = np.isin(xs_g, xs)
    preds_con, preds_lat  = model.predictions(xs_g, model.pars)
    if name == 'fechner':
        _, preds_acc = model.predictions_accuracy(preds_con, model.pars)
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            model.plot_acc(preds_acc, preds_con, subset, part, prefix, ax=ax, fig=fig)
    else: 
        preds_acc = np.repeat(np.nan, len(preds_con))
    preds_lat *= 1000 
    preds_con *= 100
    return  model.pars, preds_con, preds_lat, preds_con, preds_acc, model.mse

def latency_model_fit(data, name, column, path='estimation_files/', plot=True, prefix='', loocv=False):
    print(f'{name} _____________________________________________________')
    data = data.copy()
    data.contrast /= 100
    data[column] /= 1000
    if not os.path.exists('diagnostic_plots/individual_model_fits/%s'%name):
        os.makedirs('diagnostic_plots/individual_model_fits/%s'%name)
    n_pars = len(data.participant.unique())
    indiv_pars = np.zeros((n_pars,3))*np.nan
    finite_check = data.groupby('contrast')[column].mean()
    xs_g = np.sort(finite_check[np.isfinite(finite_check)].index.values)
    preds = np.zeros((n_pars, 2, len(xs_g),))*np.nan
    preds_acc = np.zeros((n_pars, 2, len(xs_g),))*np.nan
    parts = data.participant.unique()
    gof = np.zeros(n_pars)
    ## Fitting
    output = Parallel(n_jobs=26)(delayed(individual_fits)(data[(data.participant == part)], part, xs_g, \
        name, column, prefix) for i,part in enumerate(data.participant.unique()))
    for i,part in enumerate(parts):
        indiv_pars[i, :len(output[i][0])], preds[i,0,:], preds[i,1,:],preds_acc[i,0,:], preds_acc[i,1,:], gof[i] = output[i]
    fit_results = pd.DataFrame({'participant':parts,
                                'gof':gof,
                                'name':np.repeat(name, len(gof))})
    fit_results.to_csv(path+prefix+'_'+'fit_results_%s.csv'%name)
    indiv_pars = pd.DataFrame({'participant':parts,
               'par1':indiv_pars[:,0],
               'par2':indiv_pars[:,1],
               'par3':indiv_pars[:,2],
               'name':np.repeat(name, len(gof))})
    indiv_pars.to_csv(path+prefix+'_'+'pars_%s.csv'%name)
    preds = pd.DataFrame({'participant': np.repeat(parts, len(xs_g)),
              'contrast':preds[:,0,:].flatten(),                 
              'mrt':preds[:,1,:].flatten(), 
              'name':np.repeat(name, len(preds[:,0,:].flatten()))})
    preds.to_csv(path+prefix+'_'+'preds_%s.csv'%name)
    if name == 'fechner':
        preds_acc = pd.DataFrame({'participant': np.repeat(parts, len(xs_g)),
          'contrast':preds_acc[:,0,:].flatten(),                 
          'mpc':preds_acc[:,1,:].flatten(), 
          'name':np.repeat(name, len(preds_acc[:,0,:].flatten()))})
        preds_acc.to_csv(path+prefix+'_'+'preds_acc_%s.csv'%name)
    
    #LOOCV
    if loocv:
        loocv = np.zeros((len(xs_g)))*np.nan
        for c,contrast in enumerate(xs_g):
            preds_loocv = np.zeros((n_pars, 2, len(xs_g),))*np.nan
            output = Parallel(n_jobs=-1)(delayed(individual_fits)(data[(data.participant == part)], part, xs_g, \
            name, column, prefix, False, contrast) for i,part in enumerate(data.participant.unique()))
            for i, part in enumerate(parts):
                _, preds_loocv[i,0,:], preds_loocv[i,1,:],_, _, _ = output[i]
            preds_loocv = pd.DataFrame({'participant': np.repeat(parts, len(xs_g)),
                  'contrast':preds_loocv[:,0,:].flatten(),                 
                  'mrt':preds_loocv[:,1,:].flatten()})
            preds_loocv = preds_loocv.sort_values(by='participant')
            obs_mean = data[data.contrast == contrast][column].mean()*1000
            pred_mean = preds_loocv[np.round(preds_loocv.contrast,1) == np.round(contrast*100, 1)].mrt.mean()

            loocv[c] = (obs_mean - pred_mean)**2
    loocv = np.sqrt(np.mean(loocv))
    return loocv, preds

def run_cv(data, part, xs_g, name, column, prefix, i):
    output = Parallel(n_jobs=-1)(delayed(individual_fits)(data[(data.participant == part)], part, xs_g, \
    name, column, prefix) for i,part in enumerate(data.participant.unique()))
    
    
    
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

class latency(object):
    
    def __init__(self, model, latencies, contrast):
        """ 
        fitting Pieron/fechner/linear on latencies
        
        Parameters
        -----------
        model: str
            name of the model can be linear, Pieron or fechner
        latencies: pd.Series
            latencies to fit the model on
        contrast: pd.Series
            Contrast strength corresponding to the latencies
        """
        self.name = model
        model_func_dict = {'linear':self.lm_func, 'fechner':self.wf_diffusion, 'pieron':self.pieron_func}
        self.data = pd.DataFrame({'latencies': latencies,'contrast': contrast})
        self.lat_func = model_func_dict[model]
        self.mean_rt = self.data.latencies.mean()

    def lm_func(self, x, slope, t0):
        """Naive intercept model"""
        return slope*x + t0
        
    def pieron_func(self, x, slope, alpha):
        """ Equation of Pieron for mean latency per contrast (x) """
        return slope*x**-alpha

    def wf_diffusion(self, x,  k, Aprime, t0):
        """fechner-contrast diffusion"""
        x = np.log((x+.025)/(x-.025))#fechner contrast based on 5% difference
        return (Aprime/((k*x))) * np.tanh(Aprime * k * x)+t0

    def prop_corr(self, x, pars):
        """ Proportion correct based on wf_diffusion parameters"""
        x = np.log((x+.025)/(x-.025))#fechner contrast based on 5% difference
        k, Aprime, t0 = pars
        return 1/(1+np.exp(-2*Aprime*k*x))

    def fit(self):
        """ Fits model to provided data """
        bounds_dict = {'linear':([ -np.inf, -np.inf],[np.inf, np.inf]),
                        'pieron':([0, 0], [np.inf, np.inf]) ,# slope,  alpha
                       'fechner': ([0, 0, 0], [18.51, 7.47, 3.69])  # k, Aprime, t0
        }
        x0_dict = {'linear':[1, 1], # slope, t0
                    'pieron':[1, 1],# slope,  alpha
                    'fechner': [1, 1, 1]  # k, Aprime, t0
        }
        popt, pcov, info,_,_ = scipy.optimize.curve_fit(self.lat_func, self.data['contrast'],
                                       self.data['latencies'].values, p0=x0_dict[self.name],
                                       bounds=bounds_dict[self.name], full_output=True)
        self.pars = popt
        self.mse = np.mean((self.lat_func(self.data['contrast'], *popt) - self.data['latencies'].values)**2)


    def plot(self, part,  prefix_plot, min_contrast, max_contrast, ax=None, fig=None, path='diagnostic_plots/individual_model_fits/'):
        """ Save individual contrast-mean latency fit for the estimated parameters."""
        contrast = np.arange(min_contrast, max_contrast)
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        
        ax.plot(contrast, self.lat_func(contrast/100, *self.pars))
        contrasts = self.data['contrast'].unique()
        for c in contrasts:
            mean_latencies = self.data.loc[self.data['contrast']==c, 'latencies'].mean()
            sem = self.data.loc[self.data['contrast']==c, 'latencies'].sem()
            ax.plot(c*100, mean_latencies, '.', color='k')
            ax.errorbar(c*100, mean_latencies, xerr=0, yerr=sem, color='k')
        ax.set_ylabel('mean latencies')
        title = part
        fig.suptitle(title)
        fig.savefig(path+'%s/%s_%s'%(self.name, prefix_plot, title), dpi=300)
        plt.close(fig)

    def plot_acc(self, acc, contrast, data, part, prefix_plot,ax=None, fig=None, path='diagnostic_plots/individual_model_fits/'):
        """ Save individual contrast-mean latency fit for the estimated parameters."""
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        
        ax.plot(contrast, acc)
        contrasts = self.data['contrast'].unique()
        for c in contrasts:
            mean_latencies = data.loc[data['contrast']==c, 'correct'].mean()
            sem = data.loc[data['contrast']==c, 'correct'].sem()
            ax.plot(c, mean_latencies, '.', color='k')
            ax.errorbar(c, mean_latencies, xerr=0, yerr=sem, color='k')
        ax.set_ylabel('mean prop corr')
        title = part + '_' + '_pc'
        fig.suptitle(title)
        fig.savefig(path+'%s/%s_%s_pc'%(self.name,prefix_plot,title), dpi=300)
        plt.close(fig)

    def predictions(self, contrast, pars):
        return contrast, self.lat_func(contrast, *pars)

    def predictions_accuracy(self, contrast, pars):
        return contrast, self.prop_corr(contrast, pars)


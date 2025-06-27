#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pylab import *
from functools import partialmethod
from tqdm import tqdm

import pandas as pd
import arviz as az

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
         '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
         (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
         (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
         (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
         (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
         (1.0, 0.4980392156862745, 0.0),
         (1.0, 1.0, 0.2),
         (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
         (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
         (0.6, 0.6, 0.6)]

rcParams.update({
          'axes.axisbelow':True,
          'axes.grid': True,
          'axes.labelsize': 20.0,
          'axes.prop_cycle': cycler(color=colors), 
          'axes.titlesize': 24.0,
          'figure.figsize': [10.0, 8.0],
          'font.family': ['sans-serif'],
          'font.size': 20.0,
          'grid.color': '0.75',
          'grid.linestyle': '-',
          'legend.fontsize': 20.0,
          'legend.frameon': False,
          'legend.numpoints': 1,
          'legend.scatterpoints': 1,
          'lines.linewidth': 3.0,
          'lines.markersize': 5.0,
          'lines.solid_capstyle':'round',
          'text.color': '.15',
          'xtick.color': '.15',
          'xtick.direction': 'out',
          'xtick.labelsize': 20.0,
          'ytick.color': '.15',
          'ytick.direction': 'out',
          'ytick.labelsize': 20.0,})


# In[1]:


def df_display(df, head=10, tail=10):
    if len(df) <= head + tail:
        display(df)
    else:
        top = df.head(head)
        bottom = df.tail(tail)

        # Create ellipsis row with correct column count and index
        ellipsis_row = pd.DataFrame(
            [["..."] * df.shape[1]],
            columns=df.columns,
            index=["..."]
        )

        # Concatenate while preserving index
        combined = pd.concat([top, ellipsis_row, bottom])
        display(combined)


# In[ ]:


class Storage(object):
    def __init__(self,save_every=1):
        self.save_every=save_every
        self.count=0
        self.data=[]

    def __add__(self,other):
        s=Storage()
        s+=other
        return s

    def __iadd__(self,other):
        if self.count % self.save_every ==0:
            self.append(*other)
        self.count+=1
        return self

    def append(self,*args):
        if not self.data:
            for arg in args:
                self.data.append([arg])

        else:
            for d,a in zip(self.data,args):
                d.append(a)

    def arrays(self):
        ret=tuple([array(_) for _ in self.data])
        if len(ret)==1:
            return ret[0]
        else:
            return ret

    def __array__(self):
        from numpy import vstack
        return vstack(self.arrays())


# In[ ]:


import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.handlers import trace, seed

class MCMCModel:

    def __init__(self, model_function):
        from jax import random
        from numpyro.infer import NUTS
        self.model_function = model_function
        self.rng_key = random.PRNGKey(0)
        self.kernel_dme = NUTS(self.model_function)

    def run(self, num_warmup=500, num_samples=3000,progress_bar=True,**kwargs):
        from numpyro.infer import MCMC,log_likelihood
        self.mcmc_dme = MCMC(self.kernel_dme, 
                             num_warmup=num_warmup, 
                             num_samples=num_samples, 
                             progress_bar=progress_bar)

        self.mcmc_dme.run(self.rng_key,**kwargs)


        # Provide dummy or actual arguments
        rng_key = random.PRNGKey(0)
        self.model_trace = trace(seed(self.model_function, rng_key)).get_trace(**kwargs)
        # Extract only latent stochastic variables (sampled, not observed)
        self.variable_names = [
            name for name, site in self.model_trace.items()
            if site["type"] == "sample" and not site.get("is_observed")
        ]
        self.data_names = [
            name for name, site in self.model_trace.items()
            if site["type"] == "sample" and site.get("is_observed")
        ]

        self.samples = self.mcmc_dme.get_samples()
        self.log_likelihood = log_likelihood(
            self.model_function, 
            self.samples,**kwargs)

    @property
    def bic(self):
        true_latent_sample = {name: self.samples[name][0] for name in self.variable_names}
        k = sum([v.size for v in true_latent_sample.values()])
        n = sum([len(self.model_trace[name]['value']) for name in self.data_names])

        log_lik_values = self.log_likelihood["y"].reshape(self.mcmc_dme.num_samples,-1)

        log_lik_per_sample = jnp.sum(log_lik_values, axis=1)  # shape: (num_samples,)
        log_lik_max = jnp.max(log_lik_per_sample)   

        value = -2 * log_lik_max + k * jnp.log(n)

        return value

    @property
    def aic(self):
        true_latent_sample = {name: self.samples[name][0] for name in self.variable_names}
        k = sum([v.size for v in true_latent_sample.values()])
        n = sum([len(self.model_trace[name]['value']) for name in self.data_names])

        log_lik_values = self.log_likelihood["y"].reshape(self.mcmc_dme.num_samples,-1)

        # Total log-likelihood per sample
        log_lik_total = np.sum(log_lik_values, axis=1)  # shape: (S,)
        log_lik_max = np.max(log_lik_total) 

        AIC = -2 * log_lik_max + 2 * k

        return AIC


    @property
    def waic(self):
        from jax.scipy.special import logsumexp
        true_latent_sample = {name: self.samples[name][0] for name in self.variable_names}
        k = sum([v.size for v in true_latent_sample.values()])
        n = sum([len(self.model_trace[name]['value']) for name in self.data_names])

        log_lik_values = self.log_likelihood["y"].reshape(self.mcmc_dme.num_samples,-1)
        S=log_lik_values.shape[0] # number of samples
        llpd = jnp.sum(logsumexp(log_lik_values,axis=0))-jnp.log(S)

        p_waic=jnp.sum(jnp.var(log_lik_values,axis=0))

        value = -2 * (llpd-p_waic)

        return value

    @property
    def dic(self):
        log_lik_values = np.array(self.log_likelihood["y"].reshape(self.mcmc_dme.num_samples,-1))
        log_lik_total = np.sum(log_lik_values, axis=1)

        # Deviance per sample
        D_theta = -2 * log_lik_total
        D_bar = np.mean(D_theta)  # Expected deviance
        theta_hat_index = np.argmax(log_lik_total)  # Posterior sample with max log-lik (or use mean param estimate)
        D_hat = D_theta[theta_hat_index]

        p_D = D_bar - D_hat
        DIC = D_hat + 2 * p_D   

        return DIC

    def predict(self,N):
        self.posterior_predictive_samples=dist.Multinomial(
            total_count=N, 
            probs=self.samples['theta']).sample(self.rng_key)
        return self.posterior_predictive_samples



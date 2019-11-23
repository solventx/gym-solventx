# -*- coding: utf-8 -*-
"""
Created Today

@author: ?
"""

# IMPORT STANDARD METHODS
import os
import pandas as pd

try:
    import matplotlib.pyplot as plt
except:
    pass

import seaborn as sns

   

#def populate(obj, lyoncsv=''): # for storing separate results for aqueous and organic streams. Think of this as the preprocessing function
#    
#    Nsmod               = [i+1 for i in obj.Ns]
#    NsM                 = [sum(Nsmod[:i+1]) for i in range(len(Nsmod))]
#    
#    for module in obj.mod_space:
#        
#        name, num           = module.split('-')  
#        cols                = [ii+num for ii in obj.column]
#        ye                  = [obj.y[ij] for ij in cols]
#        Nye                 = [len(ij for ij in ye)]        
#        y                   = [ij for jk in ye for ij in jk]
#        Ny                  = [sum(Nye[:ij+1]) for ij in range(len(Nye))]
#
#        mx                  = obj.mix     
#        ns                  = mx.n_species
#        
#        ree                 = obj.ree
#        norg                = obj.mix.phase(obj.org).n_species
#        
#        base = 0
#  Needs more work to convert     
##        ystage, ystageall = [] , []
##        
##        steps = 0
##        for k in range(len(Ny)): # for each column k in the flowsheet
##            
##            count = Ny[k]
##            
##            while count >= base+ns: 
##                
##                steps += 1             
##    
##                ystageall.append([i for i in y[count-ns:count-norg]]+[j for j in y[count-ns-norg:count-ns]])
##                
##                if steps != NsM[k]:  # avoid double counting stages        
##                    ystage.append([i for i in y[count-ns:count-norg]]+[j for j in y[count-ns-norg:count-ns]])
##    
##                count -= ns                    
##                
##            base = Ny[k]
##    
##                
##        ystdf = pd.DataFrame(ystage, columns=obj.mix.species_names)  
##        ystdfall = pd.DataFrame(ystageall, columns=obj.mix.species_names)  
##        
##        obj.ystages = ystage
##    
##        obj.ystdf = ystdf
##    
##        path = os.getcwd()
##        try:
##            os.mkdir(path+'/output')
##        except FileExistsError:
##            pass
##        
##        try:
##            os.mkdir(path+'/output/models')
##        except FileExistsError:
##            pass
##        ystdf.to_csv(path + '/output/models/modelstreams.csv')
##        ystdfall.to_csv(path + '/output/models/modelstreamsall.csv')
##        
##        plotcols(obj, ree)
##                  
###    return obj

    
    
    
    
def populate(obj, lyoncsv=''): # for storing separate results for aqueous and organic streams. Think of this as the preprocessing function
    
    Nsmod               = [i+1 for i in obj.Ns]
    NsM                 = [sum(Nsmod[:i+1]) for i in range(len(Nsmod))]
    
    y                   = obj.y
    Ny                  = obj.Ny
    mx                  = obj.mix     
    ns                  = mx.n_species
    
    ree                 = obj.ree
    norg                = obj.mix.phase(obj.org).n_species
    
    base = 0

    ystage, ystageall = [] , []
    
    steps = 0
    for k in range(len(Ny)): # for each column k in the flowsheet
        
        count = Ny[k]
        
        while count >= base+ns: 
            
            steps += 1             

            ystageall.append([i for i in y[count-ns:count-norg]]+[j for j in y[count-ns-norg:count-ns]])
            
            if steps != NsM[k]:  # avoid double counting stages        
                ystage.append([i for i in y[count-ns:count-norg]]+[j for j in y[count-ns-norg:count-ns]])

            count -= ns                    
            
        base = Ny[k]

            
    ystdf = pd.DataFrame(ystage, columns=obj.mix.species_names)  
    ystdfall = pd.DataFrame(ystageall, columns=obj.mix.species_names)  
    
    obj.ystages = ystage

    obj.ystdf = ystdf

    path = os.getcwd()
    try:
        os.mkdir(path+'/output')
    except FileExistsError:
        pass
    
    try:
        os.mkdir(path+'/output/models')
    except FileExistsError:
        pass
    ystdf.to_csv(path + '/output/models/modelstreams.csv')
    ystdfall.to_csv(path + '/output/models/modelstreamsall.csv')
    
    plotcols(obj, ree)
              
#    return obj


   

def plotcols(obj, ree): # This is the plotting function

    N = sum(obj.Ns)    
    y = [i for i in range(N)]
    
    modaq = obj.ystdf[[re+'+++' for re in ree]]    # create aqueous phase composition vector
    modorg = obj.ystdf[[re+'(H(A)2)3(org)' for re in ree]]

    sns.set_context("poster")
    sns.set_style("white", {"axes.edgecolor":"gray"}) # sns.despine() darkgrid

    plt.figure(figsize=(14,5))#, facecolor='y', edgecolor='k')#  
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.3,hspace = 0.3)
    plt.gcf().subplots_adjust(bottom=0.2)
    
    plt.subplot(1, 2, 1) # nrows, ncols, index
    plt.plot(y, modaq,linestyle='--', marker='*',lw=1,markersize=14)

    plt.ylabel('Molar Flows' )
    plt.xlabel('Stages')
    plt.title('Aqueous phase equilibrium')
    plt.legend(['model '+re for re in ree], loc='lower center', bbox_to_anchor=(0.5,-0.70),fancybox=True)


    plt.subplot(1, 2, 2) # nrows, ncols, index
    plt.plot(y, modorg,linestyle='--', marker='*', lw=1, markersize=14)

    plt.ylabel('Molar Flows' )
    plt.xlabel('Stages')

    plt.title('Organic phase equilibrium')
    plt.legend(['model '+re for re in ree], loc='lower center', bbox_to_anchor=(0.5,-0.70),fancybox=True)

    path = os.getcwd()
    try:
        os.mkdir(path+'/output')
    except FileExistsError:
        pass

    try:
        os.mkdir(path+'/output/figures')
    except FileExistsError:
        pass
    figname = path + '/output/figures/profile.png'
    plt.savefig(figname)
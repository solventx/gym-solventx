"""Black-box function.

This module contains the definition of the black box function that is
optimized by RBFOpt, when using the default command line
interface. The user can implement a similar class to define their own
function.

We provide here an example for a function of dimension 3 that returns
the power of the sum of the three variables, with pre-determined
exponent.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2016.
Research partially supported by SUTD-MIT International Design Center.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os, sys

from scipy.optimize import root, fsolve

import numpy as np
import rbfopt

import cantera as ct
import pandas as pd
import operator


def get_simulator_path():
    gym_solventx_directory = ''
    for directory in sys.path:
        if 'gym-solventx' in directory:
            ind = directory.index('gym-solventx')
            gym_solventx_directory = directory[:ind+12] + '/gym_solventx/'
    return gym_solventx_directory

class solvent_extraction(rbfopt.RbfoptBlackBox):

    """
     Attributes
    ----------

    dimension : int
        Dimension of the problem.
        
    var_lower : 1D numpy.ndarray[float]
        Lower bounds of the decision variables.

    var_upper : 1D numpy.ndarray[float]
        Upper bounds of the decision variables.

    var_type : 1D numpy.ndarray[char]
        An array of length equal to dimension, specifying the type of
        each variable. Possible types are 'R' for real (continuous)
        variables, and 'I' for integer (discrete) variables.

    integer_vars : 1D numpy.ndarray[int]
        A list of indices of the variables that must assume integer
        values.
        
        """
    
    #dynamically obtain path to environment dependencies
    dirlocation = get_simulator_path()
    
    dfree = pd.read_csv(dirlocation+'envs/methods/input/cases/cases.csv') # ree feed compositions (kg/hr)
    mw   = pd.read_csv(dirlocation+'envs/methods/input/material/mw.csv')
    
    reecols = dfree.columns #[1:]

    module_id               = 2
    strip_group             = 4


    immutable_var_names     = ['mol_frac']
    mutable_var_names       = ['H+ Extraction', 'H+ Scrub', 'H+ Strip', 'OA Extraction', 'OA Scrub', 'OA Strip', 'Recycle', 'Extraction', 'Scrub', 'Strip']
    feed_var_names          = ['(HA)2(org)']

#    var_names               = ['(HA)2(org)',	'H+ Extraction', 'H+ Scrub',	'H+ Strip',	'OA Extraction',	'OA Scrub',	'OA Strip', 'Recycle','Extraction','Scrub', 'Strip']#,	'Nd Scrub',	'Pr Scrub',	'Ce Scrub',	'La Scrub']#,	'Nd',	'Pr',	'Ce',	'La','Factor']
    solvs                   = ['H2O(L)', '(HA)2(org)', 'dodecane']
    coltypes                = ['Extraction','Scrub','Strip']
    REEs                    = {1:['Nd','Pr','Ce','La'],2:['Nd','Pr'],3:['Ce','La'], 4:['Nd'],5:['Ce']}
    phasenames              = ['HCl_electrolyte','PC88A_liquid']
    
    ml2l                    = 0.001 # mililiter to liter

    kg_p_g          = 0.001 # kg per gram
    s_2_h           = 1.0/3600 # seconds to hours
    hr_p_d          = 24
    d_p_yr          = 365
    hr_p_day          = 24
    

    
    
    def __init__(self, data=dfree, mw=mw, moduleID=module_id, stripGroup=strip_group, 
                 prep_capex=0, prep_opex=0, prep_revenue=0, prep_npv=0, column_types = coltypes, imvn=immutable_var_names,
                 mvn=mutable_var_names, fvn=feed_var_names, solvents=solvs, ree=REEs, phasenames=phasenames, kg_p_g=kg_p_g): # xbs = bar_stoch,
                 
        """Constructor.
        """
        
        if moduleID == 1:
            target_conc = 45 #135 #22.05 # g/L  #9.95 #
            xml = self.dirlocation + 'envs/methods/xml/PC88A_HCL_NdPrCeLa.xml'
        elif moduleID == 2:
            target_conc = 20# 60 #9.95 # g/L  #9.95 #
            xml = self.dirlocation + 'envs/methods/xml/PC88A_HCL_NdPr.xml'
        else:
            target_conc = 25 #75 #12.1 # g/L  #9.95 #
            xml = self.dirlocation +'envs/methods/xml/PC88A_HCL_CeLa.xml'
                
        
        # Set required data

        self.df             = data              
 
        self.xml           =  xml
        self.phase_names    = phasenames # from xml input file
        self.phase          = ct.import_phases(xml,self.phase_names) 

        # Derived and/or reusable system parameters        
        self.column         = column_types #list(self.mod_input['Section']) # Column name
        self.solv           = solvents # solvent list -.i.e., electrolyte, extractant and organic diluent
        
        # ree by modules
        self.ree            = ree[moduleID] # (rare earth) metal list
        self.ree_strip      = ree[stripGroup] # Strip target
        self.is_ree         = [1 if re in self.ree_strip else 0 for re in self.ree ]
#        self.MID            = moduleID
                   
        # Cantera indices
        self.mix            = ct.Mixture(self.phase)
        self.aq             = self.mix.phase_index(self.phase_names[0])
        self.org            = self.mix.phase_index(self.phase_names[1])
        self.ns             = self.mix.n_species
        self.naq            = self.mix.phase(self.aq).n_species
        self.norg           = self.mix.phase(self.org).n_species
        
        self.HA_Index       = self.mix.species_index(self.org,'(HA)2(org)') # index of extractant in canera species list
        self.Hp_Index       = self.mix.species_index(self.aq,'H+') # index of H+ in cantera species list
        self.Cl_Index       = self.mix.species_index(self.aq,'Cl-') # index of Cl in cantera species list
        
        # outer optimization parameter names        
        self.immutable_var_names        = imvn 
        self.mmutable_var_names         = mvn 
        self.feed_var_names             = fvn
        
        self.dimension                  = len(fvn+mvn)
        
        self.canteranames   = self.mix.species_names
        self.fixed_species  = ['H2O(L)','OH-', 'Cl-', 'dodecane']
        self.canteravars    = [ij for ij in self.canteranames if ij not in self.fixed_species] # 'Cl-',

        self.nsy                     = len(self.canteravars)
        self.naqy                    = len([ij for ij in self.mix.species_names[:self.naq] if ij not in self.fixed_species])
        self.norgy                   = len([ij for ij in self.mix.species_names[self.naq:] if ij not in self.fixed_species])
        
        self.mwre,\
        self.mwslv          = self.get_mw() # g/mol
        self.rhoslv         = [1000, 960, 750] # [g/L]

        # Feed volume
        
        self.ree_mass       = [self.df[ij+'Cl3'][0]*(mw[ij][0]/mw[ij][0]) for ij in self.ree] # kg/hr of ree chloride
        
        mree                = sum(self.ree_mass) # kg/hr
        self.vol        = mree / kg_p_g / target_conc   # [kg/hr]/[kg/g] /[g/L] = [L/hr] 
        self.df['H2O Volume[L/hr]'] = self.vol
        
        self.purity_spec    = 1 # 
        self.recov_spec     = 1 # 
        self.penalty        = [ij for ij in data[['recovery penalty 1','recovery penalty 2', 'purity penalty']] ]
        
        self.revenue        = [0,0,0]
        self.Ns             = [0,0,0]

        self.nsp            = pd.DataFrame() # feed streams (aq and org) for each column       
        self.nsp0           = pd.DataFrame() # feed streams (aq and org) for each column       

        self.y              = {} # all compositions
        self.Ns             = {}
        
        self.variables              = []#[0.45,1e-4,1e-4,1.2,3,1.2,1.6,0.8,4,4,3]

        
        self.var_lower      = np.array([0.15 ,      0.000001,	    0.000001,	   0.5,          1.0,	       1.0,       1.0,	     0.0,	        1,	    1,	     1])
        self.var_upper      = np.array([0.65 ,      0.0001,	         0.0001,       3.0,        6.0,           6.0,	    6.0,	    0.9,	    20,	    20,	     20])             
        self.var_type       = np.array(['R']*(self.dimension-3)+['I']*3)


        
    # -- end function
  



    def get_mw(self, conv=ml2l):
        """ Initialize parameters for cantera simulation. init() calls this function"""
        
        mx                  = ct.Mixture(self.phase)
        aq                  = mx.phase_index(self.phase_names[0])
        org                 = mx.phase_index(self.phase_names[1])

        mwre                = np.zeros(len(self.ree)) # molecular weight of rees
        mwslv               = np.zeros(len(self.solv)) # mw & densities for 'solvents'         
        
        for re in self.ree:
            mwre[self.ree.index(re)]         = mx.phase(aq).molecular_weights[mx.phase(aq).species_index(re+'+++')]
            
        for so in self.solv:   
            if so == 'H2O(L)':
                mwslv[self.solv.index(so)]   = mx.phase(aq).molecular_weights[mx.phase(aq).species_index(so)]
            else:
                mwslv[self.solv.index(so)]   = mx.phase(org).molecular_weights[mx.phase(org).species_index(so)]

        return mwre, mwslv



    def create_var_space(self, n_products=2, n_components=2, input_feeds=1,): # Creates containers for process variables


        var_space = {
            'immutable': {},
            'mutable':   {}, #var, index in obj.variables
        }
        
        mod_space = {}
        x_space = {}
        
        immutable_var_names = ['mol_frac']
        mutable_var_names   = ['H+ Extraction', 'H+ Scrub', 'H+ Strip', 'OA Extraction', 'OA Scrub', 'OA Strip', 'Recycle', 'Extraction', 'Scrub', 'Strip']
        feed_var_names = ['(HA)2(org)']
        lenx = len(mutable_var_names + feed_var_names)

        index = 0
        for feed_num in range(input_feeds):
            var_space['mutable'][f'{feed_var_names[0]}-{feed_num}'] = index
            index += 1 
            
        count = 0
        for module_num in range(n_products-1):
            
            mod_space[f'module-{module_num}'] = module_num
            x_space[f'module-{module_num}'] = self.variables[count:count+lenx]
            count += lenx
            
            for i, var in enumerate(mutable_var_names):
                var_space['mutable'][f'{var}-{module_num}'] = index
                index += 1

        for comp_num in range(n_components):
            var_space['immutable'][f'{immutable_var_names[0]}-{comp_num}'] = index
            index += 1
        self.var_space = var_space
        mutable = self.var_space['mutable']
        immutable = self.var_space['immutable']
        self.combined_var_space = combine_dict(mutable, immutable)
        
        self.mod_space = mod_space
        self.x_space = x_space
        self.num_feeds = len(mod_space) * len(self.column)




    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """ Option 1 - fast evaluation """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


    def create_nsp(self, module, lim, init_recyc=0.0, kg_p_g=kg_p_g ): #, h0, target, xml, cantera_data, experiments):
        
        """ Create mole flows for column feed streams in a given module. Arranges
            them in the dataframe, nsp, in the form that Cantera expects """
               
        nre                 = np.zeros(len(self.ree)) # REE specie moles
        salts               = np.zeros(len(self.ree)) # neutral complexe moles
        name, num           = module.split('-')        

                       
#        Flows
        orgvol              = self.orgvol[self.mod_space[module]]
        aqvols              = [orgvol/(self.variables[self.combined_var_space['OA Extraction-'+num]]),orgvol/(self.variables[self.combined_var_space['OA Scrub-'+num]]), orgvol/(self.variables[self.combined_var_space['OA Strip-'+num]]) ]
        

#        Compositions   
        vol_HA              = orgvol *  (self.variables[self.combined_var_space['(HA)2(org)-0']])
        vol_dodec           = orgvol - vol_HA
        n_HA                = vol_HA * self.rhoslv[self.solv.index('(HA)2(org)')]/self.mwslv[self.solv.index('(HA)2(org)')] # [L/hr]*[g/L]/[g/mol] = [mol/hr]
        n_dodec             = vol_dodec * self.rhoslv[self.solv.index('dodecane')]/self.mwslv[self.solv.index('dodecane')] # [L/hr]*[g/L]/[g/mol] = [mol/hr]            

        for k in range(len(self.column)): # k represents different columns - extraction, scrub or strip

            # check for module ID, then determine upstream source
            parent_col = get_parent(self.column[k]+'-'+num) 
            
            n_H2O           = aqvols[k] * self.rhoslv[self.solv.index('H2O(L)')]/self.mwslv[self.solv.index('H2O(L)')]  # [L/hr]*[g/L]/[g/mol] = [mol/hr]
            n_Hp            = aqvols[k] * (self.variables[self.combined_var_space['H+ '+self.column[k]+'-'+num]])   # [L/hr]*[g/L]/[g/mol] = [mol/hr]

          
            if k==0:
                for re in self.ree:                     
                    nre[self.ree.index(re)] = self.ree_mass[self.ree.index(re)]/kg_p_g / self.mwre[self.ree.index(re)] # [kg/hr]/[kg/g]/[g/mol] = [mol/hr]

            elif k==1:
                uncle = get_parent(self.column[k-1]+'-'+num) # if a parent module exists for ext column  
                if uncle:                       
                    myree =  np.array(self.y[uncle][-self.nsy+self.canteravars.index('H+')+1:-self.norgy])

                    target_indices = highest_remaining(myree, lim)
#                    print('target_indices', target_indices)                     
                    target_ree = myree[target_indices[:round( (len(myree)-lim)/2 )]] # replace 2 with something automatic later
#                    print('target_ree', target_ree)
                    my_is_ree = [1 if ij in target_ree else 0 for ij in myree]
                    
                    for re in self.ree: 
                        nre[self.ree.index(re)] = my_is_ree[self.ree.index(re)] *self.variables[self.combined_var_space['Recycle-'+num]] *init_recyc* myree[self.ree.index(re)] # = [mol/hr]                   
                else:
                    
                    for re in self.ree:                     
                        nre[self.ree.index(re)] = self.is_ree[self.ree.index(re)] *self.variables[self.combined_var_space['Recycle-'+num]] *init_recyc* self.ree_mass[self.ree.index(re)]/kg_p_g / self.mwre[self.ree.index(re)] # [kg/hr]/[kg/g]/[g/mol] = [mol/hr]
                                                                                                                        # 0.05 makes it a small value
            else:
                for re in self.ree:
                    nre[self.ree.index(re)] = 0.0                
                            
            n_Cl            = 3*(sum(nre)) + n_Hp # Cl- mole balance, all REEs come in as chlorides from leaching


            # check for module ID, then determine upstream source
            if parent_col: 
                afa, okwa           = parent_col.split('-')        
                
                # mass balance for chloride ion
                if afa == 'Strip': # if strip column, subtract the recycled portion to estimate equivalent recycle ratio 

                    strips = np.array(self.y[parent_col][-self.nsy+self.canteravars.index('H+')+1:-self.norgy])
                    targ_ind = highest_remaining(strips, lim)
                    strip_ree = strips[targ_ind[0]]
                    scrub_ree = self.nsp['Scrub-'+okwa][self.canteranames.index('Cl-')+1+targ_ind[0]] # corresponding index in scrub-in

                    recycle = scrub_ree/strip_ree # equivalent recycle ratio
                    self.recycle.update({module:recycle})
                    
                    nre = [ij*(1-recycle) for ij in self.y[parent_col][-self.nsy+self.canteravars.index('H+')+1:-self.norgy]] # no of moles of rare earth
                    n_H2O = self.nsp['Strip-'+okwa][self.canteranames.index('H2O(L)')]*(1-recycle)
                else:
                    nre = self.y[parent_col][-self.nsy+self.canteravars.index('H+')+1:-self.norgy]
                    n_H2O =self.nsp[parent_col][self.canteranames.index('H2O(L)')]
                  
                n_Cl = n_Hp + 3*sum(nre)

            n_specs         = [n_H2O,n_Hp,0,n_Cl]+[ii for ii in nre]+[n_HA,n_dodec] +[ij for ij in salts]

 
            # store in pandas dataframe
            self.nsp[self.column[k]+'-'+num]       = n_specs 
            self.nsp0[self.column[k]+'-'+num]      = n_specs 



                
                

    def evaluate(self, x, lim=2, kg_p_g=kg_p_g ): #
        
        """ This is the simpler implementation of the process column design
            it avoids the need to converge recycle streams. For now, it is not
            adequate for multi-module processes (I don't yet have a reliable way of defining
            the equivalent recycle amount)"""

        self.variables                  = [i for i in x] 
        #        MID = ' '+str(self.MID)
        
        self.status         = {} # all status
        self.msg            = {} # all msgs
        self.fun            = {} # all fun vals
        self.recycle        = {}
        self.stage_status   = {}

        # Store all numbers of stages
        for key in self.mod_space:
            name, num           = key.split('-')
            for item in self.column:
                self.Ns.update( {item+'-'+num: int(self.variables[self.combined_var_space[item+'-'+num]]) } )
        
        # Assuming org volumetric flow is the same at every extraction stage - This is for initialization
        self.orgvol              = [self.vol * (self.variables[self.combined_var_space['OA Extraction-0']]) 
                                        for ij in range(len(self.mod_space))]
        
########################construct nsp - dataframe of feed species molar amounts ######################

        
        for module in self.mod_space:

            name, num           = module.split('-')

            self.create_nsp(module, lim, init_recyc=1)


    ########################## Evaluate extraction ########################
    
            resye               = self.eval_column(module,'Extraction')        
    
            self.y.update({'Extraction-'+num:[ij for ij in resye.x]})
            self.status.update({'Extraction-'+num:resye.success})
            self.msg.update({'Extraction-'+num:resye.message})
            self.fun.update({'Extraction-'+num:resye.fun})
            self.stage_status.update({'Extraction-'+num: self.errorcheck(resye.x, num, 'Extraction', self.Ns['Extraction-'+num]) }) #returns list that shows final status of each 'stage' solution as True or False 

    
    ########################## Evaluate scrub  ##############################
            
            self.nsp['Scrub-'+num][self.naq:]  =   [ resye.x[self.canteravars.index('(HA)2(org)')] ] + [self.nsp['Scrub-'+num][self.canteranames.index('dodecane')] ]+\
                                [jk for jk in resye.x[self.canteravars.index('(HA)2(org)')+1:self.nsy]]  # org exit from feed stage 1
    
                
            resysc               = self.eval_column(module, 'Scrub')        

            self.y.update({'Scrub-'+num:[ij for ij in resysc.x]})
            self.status.update({'Scrub-'+num:resysc.success})
            self.msg.update({'Scrub-'+num:resysc.message})
            self.fun.update({'Scrub-'+num:resysc.fun})
            self.stage_status.update({'Scrub-'+num: self.errorcheck(resysc.x, num, 'Scrub', self.Ns['Scrub-'+num]) }) #returns list that shows final status of each 'stage' solution as True or False 
            
    ########################## Evaluate strip ##############################
    
            self.nsp['Strip-'+num][self.naq:]  =   [ resysc.x[self.canteravars.index('(HA)2(org)')] ] + [self.nsp['Strip-'+num][self.canteranames.index('dodecane')] ]+\
                                    [jk for jk in resysc.x[self.canteravars.index('(HA)2(org)')+1:self.nsy]]  # org exit from feed 'Strip'age 1
    
    
            resyst               = self.eval_column(module,'Strip')        

            self.y.update({'Strip-'+num:[ij for ij in resyst.x]})
            self.status.update({'Strip-'+num:resyst.success})
            self.msg.update({'Strip-'+num:resyst.message})
            self.fun.update({'Strip-'+num:resyst.fun})
            self.stage_status.update({'Strip-'+num: self.errorcheck(resyst.x, num, 'Strip', self.Ns['Strip-'+num]) }) #returns list that shows final status of each 'stage' solution as True or False 

       
#            self.reward()#(resy.x)        
#
#            
#        self.constraints('Strip-0')
        
#        self.objective('0')
#        
#        
#        return self.funval        

############################################################################### 
 
    
    
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """ Option 2 - detailed evaluation """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
       
    

    def create_nsp_loop(self, module, lim, init_recyc=0.0, kg_p_g=kg_p_g ): #, h0, target, xml, cantera_data, experiments):
        
        """ Create mole flows for column feed streams in a given module. Arranges
            them in the dataframe, nsp, in the form that Cantera expects """
               
        nre                 = np.zeros(len(self.ree)) # REE specie moles
        salts               = np.zeros(len(self.ree)) # neutral complexe moles
        name, num           = module.split('-')        

                       
#        Flows
        orgvol              = self.orgvol[self.mod_space[module]]
        aqvols              = [orgvol/(self.variables[self.combined_var_space['OA Extraction-'+num]]),orgvol/(self.variables[self.combined_var_space['OA Scrub-'+num]]), orgvol/(self.variables[self.combined_var_space['OA Strip-'+num]]) ]
        

#        Compositions   
        vol_HA              = orgvol *  (self.variables[self.combined_var_space['(HA)2(org)-0']])
        vol_dodec           = orgvol - vol_HA
        n_HA                = vol_HA * self.rhoslv[self.solv.index('(HA)2(org)')]/self.mwslv[self.solv.index('(HA)2(org)')] # [L/hr]*[g/L]/[g/mol] = [mol/hr]
        n_dodec             = vol_dodec * self.rhoslv[self.solv.index('dodecane')]/self.mwslv[self.solv.index('dodecane')] # [L/hr]*[g/L]/[g/mol] = [mol/hr]            

        
        
        for k in range(len(self.column)):

            # check for module ID, then determine upstream source
            parent_col = get_parent(self.column[k]+'-'+num) 
            
            n_H2O           = aqvols[k] * self.rhoslv[self.solv.index('H2O(L)')]/self.mwslv[self.solv.index('H2O(L)')]  # [L/hr]*[g/L]/[g/mol] = [mol/hr]
            n_Hp            = aqvols[k] * (self.variables[self.combined_var_space['H+ '+self.column[k]+'-'+num]])   # [L/hr]*[g/L]/[g/mol] = [mol/hr]

          
            if k==0:
                for re in self.ree: 
                   
                    nre[self.ree.index(re)] = self.ree_mass[self.ree.index(re)]/kg_p_g / self.mwre[self.ree.index(re)] # [kg/hr]/[kg/g]/[g/mol] = [mol/hr]


            elif k==1:
                uncle = get_parent(self.column[k-1]+'-'+num) # if a parent module exists for ext column  
                if uncle:                       
                    myree =  np.array(self.y[uncle][-self.nsy+self.canteravars.index('H+')+1:-self.norgy]) #     maxids = np.array(myree).argsort()[-2:][::-1]

                    # Replace section with something more automatated
                    target_indices = highest_remaining(myree, lim) 
                    target_ree = myree[target_indices[:round( (len(myree)-lim)/2 )]] # replace 2 with something automatic later
                    my_is_ree = [1 if ij in target_ree else 0 for ij in myree]
                    self.tindex.update({module:target_indices})
                    for re in self.ree: 
                        nre[self.ree.index(re)] = my_is_ree[self.ree.index(re)] *self.variables[self.combined_var_space['Recycle-'+num]] *init_recyc* myree[self.ree.index(re)] # = [mol/hr]                   
                else:
                    
                    for re in self.ree:                     
                        nre[self.ree.index(re)] = self.is_ree[self.ree.index(re)] *self.variables[self.combined_var_space['Recycle-'+num]] *init_recyc* self.ree_mass[self.ree.index(re)]/kg_p_g / self.mwre[self.ree.index(re)] # [kg/hr]/[kg/g]/[g/mol] = [mol/hr]
                                                                                                                        # 0.05 makes it a small value
            else:
                for re in self.ree:
                    nre[self.ree.index(re)] = 0.0                
                            
            n_Cl            = 3*(sum(nre)) + n_Hp # Cl- mole balance, all REEs come in as chlorides from leaching


           
            if parent_col: 
                afa, okwa           = parent_col.split('-')        
                
                if afa == 'Strip':
                    nre = [ij*(1-self.variables[self.combined_var_space['Recycle-'+num]]) for ij in self.y[parent_col][-self.nsy+self.canteravars.index('H+')+1:-self.norgy]]
                    n_H2O = self.nsp['Strip-'+okwa][self.canteranames.index('H2O(L)')]*(1-self.variables[self.combined_var_space['Recycle-'+num]])
                    self.recycle.update({module:self.variables[self.combined_var_space['Recycle-'+num]]})

                else:
                    nre = self.y[parent_col][-self.nsy+self.canteravars.index('H+')+1:-self.norgy]
                    n_H2O = self.nsp[parent_col][self.canteranames.index('H2O(L)')]
                   
                n_Cl = n_Hp + 3*sum(nre)

            n_specs         = [n_H2O,n_Hp,0,n_Cl]+[ii for ii in nre]+[n_HA,n_dodec] +[ij for ij in salts]

            # store in pandas dataframe
            self.nsp[self.column[k]+'-'+num]       = n_specs 
            self.nsp0[self.column[k]+'-'+num]      = n_specs 


                
                


    def evaluate_loop(self, x, lim=0, kg_p_g=kg_p_g ): # Option 3:
        """ This is a more representative implementation of the process column design
            it explicitly converges recycle streams - but this makes it very expensive."""

        self.variables                  = [i for i in x] 

        self.status         = {} # all status
        self.msg            = {} # all msgs
        self.fun            = {} # all fun vals
        self.tindex         = {} # target indices for nre recycle convergence for downstream modules
        self.recycle        = {}
        self.stage_status   = {}
        


        # Store all stage numbers

        for key in self.mod_space:
            name, num           = key.split('-')
            for item in self.column:
                self.Ns.update( {item+'-'+num: int(self.variables[self.combined_var_space[item+'-'+num]]) } )

            
        # Assuming org is the same at every extraction stage - This is for initialization
        self.orgvol              = [self.vol * (self.variables[self.combined_var_space['OA Extraction-0']]) 
                                        for ij in range(len(self.mod_space))]
        
########################construct nsp#  dataframe of feed species molar amounts ######################

        for module in self.mod_space:

            name, num           = module.split('-')

            self.create_nsp_loop(module, lim, init_recyc=0.05)
            

########################## Evaluate extraction ############################

            resye               = self.eval_column(module, 'Extraction')        
    
            self.y.update({'Extraction-'+num:[ij for ij in resye.x]})
            self.status.update({'Extraction-'+num:resye.success})
            self.msg.update({'Extraction-'+num:resye.message})
            self.fun.update({'Extraction-'+num:resye.fun})
            self.stage_status.update({'Extraction-'+num: self.errorcheck(resye.x, num, 'Extraction', self.Ns['Extraction-'+num]) }) #returns list that shows final status of each 'stage' solution as True or False 
    
    ########################## Evaluate scrub & strip #####################

            nre                 = np.array([ij for ij in self.nsp['Scrub-'+num][self.canteranames.index('Cl-')+1:self.naq-lim]]) # only rare earths                    
#            converge            = root(tearcolumns, nre, args=(self, resye.x, lim, num, module, 'Scrub','Strip'), method='excitingmixing', options=None)
            converge            = fsolve(tearcolumns, nre, args=(self, resye.x, lim, num, module, 'Scrub','Strip'))
            
            

    #############################update y vector of stream compositions####      
            count = 1
            for item in self.resy:

                self.y.update({self.column[count]+'-'+num:[ij for ij in item.x]})
                self.status.update({self.column[count]+'-'+num:item.success})
                self.msg.update({self.column[count]+'-'+num:item.message})
                self.fun.update({self.column[count]+'-'+num:item.fun})

                self.stage_status.update({self.column[count]+'-'+num: self.errorcheck(item.x, num, self.column[count], self.Ns[self.column[count]+'-'+num]) }) #returns list that shows final status of each 'stage' solution as True or False 

                count += 1
                
            self.converge = converge                
                  
#            self.reward()#(resy.x)  
#
#            
#        self.constraints('Strip-0')
        
#        self.objective('0')
#        
#        
#        return self.funval        
        
        
 

    def eval_column(self, module, col):
        
        """ This function evaluates the column to compute stream compositions
            for all stages """

        name, num           = module.split('-')        
        Ns                  = int(self.variables[self.combined_var_space[col+'-'+num]]) # Ns (number of stages per column)


        ycol                = self.inity(col,num, Ns) # initialize y (stream vector)      
#        resy               = root(eColOne, ycol, args=(self, num, col, Ns), method='df-sane', options=None) # method='hybr', options=None) #options={'disp':True, 'maxfev':15}    
        resy               = root(eColOne, ycol, args=(self, num, col, Ns), method='df-sane', options=None) # method='hybr', options=None) #options={'disp':True, 'maxfev':15}    

        return resy


     
            
        
    def constraints(self, col):

        purity              = 0
        C                   = []
        
        for re in self.ree_strip:
            C.append(self.recov_spec - self.recovery[col][self.ree.index(re)])
            purity += self.purity[col][self.ree.index(re)]

        if purity >= 0.98:
            C.append((self.purity_spec - purity)**2 ) # purity
        else:
            C.append(self.purity_spec - purity ) # purity
                    
        self.cineq = C





    def inity(self, col, num, Ns): # Initialize y

        y_i = []

        for m in range(Ns): 

            y_i.append([self.nsp[col+'-'+num][self.canteranames.index(jk)] for jk in self.canteravars]) # pick values from nsp that corespond to the variable entries

        y = np.array([ii for ij in y_i for ii in ij]) 
                      
        return y
            




    def errorcheck(self, y, num, column, Ns): # call this function 
        
        status             = ecolumncheck (y, self, num, column, Ns)

        return status             



           
            

    def  reward(self):
        
        col_out = {} 
        max_pur, purity  = {}, {}
        max_recov, recovery = {}, {}
        argmax = {}  
        reemax = {}
        feed_in   = {} #self.nsp['Extraction-0'][self.canteranames.index('Cl-')+1:self.naq]        
        scrub_in  = {} # for closed, do not use scrub in
        

        parents = [get_parent(item) for item in self.y.keys() if get_parent(item) != None]            
        
        for key, value in self.y.items():

            if key not in parents:                
                feed_in.update({key: [ij for ij in self.nsp['Extraction-0'][self.canteranames.index('Cl-')+1:self.naq]]})
                stream_0 = value[-self.nsy+self.canteravars.index('H+')+1:-self.norgy]

                name, num           = key.split('-')        
                if name == 'Strip':
                    stream = [ij-jk for ij,jk in zip(value[-self.nsy+self.canteravars.index('H+')+1:-self.norgy], self.nsp['Scrub-'+num][self.canteranames.index('Cl-')+1:self.naq])]
                else:
                    stream = stream_0.copy() #value[-self.nsy+self.canteravars.index('H+')+1:-self.norgy]
                    
                col_out.update({key:stream})
                purity.update({key: [0 if sum(stream_0) == 0 else i/sum(stream_0) for i in stream_0 ]})
                max_pur.update({key:max(purity[key])})
                argmax.update({key:np.argmax(purity[key])})
                reemax.update({key:self.ree[argmax[key]]})

                name, num           = key.split('-')        
                if name == 'Scrub':
                    scrub_in.update({key:[ij for ij in self.nsp['Scrub-'+num][self.canteranames.index('Cl-')+1:self.naq] ]})
                else:
                    scrub_in.update({key:[0 for ij in self.nsp['Scrub-'+num][self.canteranames.index('Cl-')+1:self.naq]]})


        scrubs = get_sums(scrub_in,'Scrub-0')             
        feeds  = get_sums(feed_in,'Scrub-0')
        feeds = [ij/len(feed_in) for ij in feeds]
        
                
        for key, value in col_out.items():
            recovery.update({key:[ij/jk for ij,jk in zip(col_out[key],feed_in[key])]})
            
            max_recov.update({key: recovery[key][argmax[key]]})
         
        self.recovery = recovery
        self.max_recov = max_recov
        self.ree_max   = reemax
        
        self.purity  = purity
        self.max_pur = max_pur
        self.raffinates = col_out
        
        self.feed_in    = feed_in
        self.parent_cols = parents
        
        self.total_feed = feeds
        self.total_scrubs = scrubs




    def get_dimension(self):
        """Return the dimension of the problem.

        Returns
        -------
        int
            The dimension of the problem.
        """
        return self.dimension
    # -- end function
    
    def get_var_lower(self):        
        """Return the array of lower bounds on the variables.

        Returns
        -------
        List[float]
            Lower bounds of the decision variables.
        """
        return self.var_lower
    # -- end function
        
    def get_var_upper(self):
        """Return the array of upper bounds on the variables.

        Returns
        -------
        List[float]
            Upper bounds of the decision variables.
        """
        return self.var_upper
    # -- end function

    def get_var_type(self):
        """Return the type of each variable.
        
        Returns
        -------
        1D numpy.ndarray[char]
            An array of length equal to dimension, specifying the type
            of each variable. Possible types are 'R' for real
            (continuous) variables, and 'I' for integer (discrete)
            variables.
        """
        return self.var_type


    
# -- end class

#####################################################################################################       
#------------------------------HELPER FUNCTIONS------------------------------


#reverse key value pairs 
def reverse_dict(D):
	return {v: k for k, v in D.items()}



#combine a and b and return the new dictionary
def combine_dict(a, b):
	c = {}
	c.update(a)
	c.update(b)
	return c


def get_parent(var):
  '''
    takes variable name, determines module number, then returns the column_names
    and module number that feeds into that module_num

    Extraction-1 -> Extraction-0
	Extraction-4 -> Strip-1
    Strip-4 -> None
    Extraction-0 -> None
  '''

  name, num = var.split('-')
  num = int(num)
  if  name != 'Extraction' or num < 1:
    return None
  parentNum = (num-1)//2
#  print('child', var, '   parent', (f'Extraction-{parentNum}',f'Strip-{parentNum}')[num % 2 == 0])
  return (f'Extraction-{parentNum}',f'Strip-{parentNum}')[num % 2 == 0]




def get_next(item_num, dir):
    '''
      returns next node number in tree in dir specified
          0      level: 0
        / \    
        1   2           1
      / \ / \
      3  4 5  6         2
  
      get_next(2, 'left') => 5
      get_next(1, 'right') => 4 
    '''
    item_num   = int(item_num)
    level, pos = get_level(item_num)
    next_level = pow(2, level+1) #items in next level
    add        = next_level-(pow(2, level)-pos)
    left       = item_num + add
    
    return (left, left+1)[dir=='right']


        
def get_level(curNum):
    '''
      helper function to get level and pos of a number in a binary tree
          0      level: 0
        / \    
        1   2           1
      / \ / \
      3  4 5  6         2
  
    get_level(2) => 1, 1
    get_level(4) => 2, 1 
    '''
    curNum = int(curNum)
    level = 0
    while pow(2, level+1) <= curNum+1:
      level+=1
    pos = curNum - pow(2, level) + 1 #0 indexed
    return level, pos




def tearcolumns(nre, obj, y, lim, num, module, sc, st): #ree
   
#    print('len y in tear', len(y))
    
    resy                = []
    recycle             = obj.variables[obj.combined_var_space['Recycle-'+num]]
    
    # update scrub feed with recycled stream
    obj.nsp[sc+'-'+num] = [ij for ij in obj.nsp0[sc+'-'+num]] # update with primary scrub feed - no address sharing
    

    obj.nsp[sc+'-'+num][obj.canteranames.index('Cl-')+1:obj.naq-lim] = [max(0,ij) for ij in nre] # update ree !!!!!!!!!!REMOVED +=!!!!!!!!!!
    obj.nsp[sc+'-'+num][obj.canteranames.index('Cl-')] =  obj.nsp[sc+'-'+num][obj.canteranames.index('H+')] + 3*sum(obj.nsp[sc+'-'+num][obj.canteranames.index('Cl-')+1:obj.naq] )# update cl
    
    
    obj.nsp[sc+'-'+num][obj.naq:]  =   [ y[obj.canteravars.index('(HA)2(org)')] ] + [obj.nsp[sc+'-'+num][obj.canteranames.index('dodecane')] ]+\
                            [jk for jk in y[obj.canteravars.index('(HA)2(org)')+1:obj.nsy]]  # org exit from feed stage 1

        
    resy.append( obj.eval_column(module, sc) )
    y1                  = resy[0].x

    ####################################################################

    obj.nsp[st+'-'+num][obj.naq:]  =   [ y1[obj.canteravars.index('(HA)2(org)')] ] + [obj.nsp[st+'-'+num][obj.canteranames.index('dodecane')] ]+\
                            [jk for jk in y1[obj.canteravars.index('(HA)2(org)')+1:obj.nsy]]  # org exit from feed stage 1

    resy.append( obj.eval_column(module, st) )
    
    obj.resy = resy
    
    # compute strip exit ree molar amount
    rhs = resy[1].x[-obj.nsy+obj.canteravars.index('H+')+1:-obj.norgy-lim] # 
    
#    print (sum(recycle * rhs - nre))

    return (recycle * rhs) - nre



    
    


def eColOne (yI, obj, num, column, Ns, ree=[]) : # This should be equivalent to actual Lyon et al 4 recycling

    y                       = np.array([i for i in yI])
    naq                     = obj.naq
    nsy                     = obj.nsy
    norgy                   = obj.norgy
    stream                  = np.zeros(obj.ns)
    
###############################################################################
#    print(Ns, len(y))
    
    count = 0 # adjust reference to column inlet stream  

    if Ns == 1: # single stage column
        stream[:naq]        = obj.nsp[column+'-'+num][:naq] # aqueous feed 
        stream[naq:]        = obj.nsp[column+'-'+num][naq:]
        obj.mix.species_moles    = stream.copy() 
        obj.mix.equilibrate('TP',log_level=0) # default  # maxsteps=100, maxiter=20
        y[count:count+nsy] = [obj.mix.species_moles[obj.canteranames.index(ij)] for ij in obj.canteravars]    #  # update aq and org variable vector 

        
    else:
    
        for i in range(1,Ns+1): # for each stage in column extraction   
            
            if i == 1: # feed aqueous stage 1  
                stream[:naq]        = obj.nsp[column+'-'+num][:naq] # aqueous feed 
                stream[naq:]        = [y[count+nsy+obj.canteravars.index('(HA)2(org)')]] +\
                                        [obj.nsp[column+'-'+num][obj.canteranames.index('dodecane')] ]+\
                                        [jk for jk in y[count+(nsy)+obj.canteravars.index('(HA)2(org)')+1:count+(2*nsy)] ]
                                        
            elif i == Ns : #feed organic stage N  
                stream[naq:]        = obj.nsp[column+'-'+num][naq:]
                
                cl_minus            = y[count-nsy+obj.canteravars.index('H+')] + 3*sum([jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]])            
                stream[:naq]        = [obj.nsp[column+'-'+num][obj.canteranames.index('H2O(L)')]] +\
                                        [y[count-nsy+obj.canteravars.index('H+')]]+ [obj.nsp[column+'-'+num][obj.canteranames.index('OH-')] ]+\
                                        [cl_minus]+\
                                        [jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]] # counts from after Cl up until the last ree
                                    # The use of count-nsy follows the logic that at the beginning of the second stage (N=2), count has been updated by nsy, but
                                    # this indexing for y is just starting (at zero); so to correct for this lag, we have to subtract nsy from count each time                                
                           
            else:
                cl_minus            = y[count-nsy+obj.canteravars.index('H+')] + 3*sum([jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]])            
                stream[:naq]        = [obj.nsp[column+'-'+num][obj.canteranames.index('H2O(L)')]] +\
                                        [y[count-nsy+obj.canteravars.index('H+')]]+ [obj.nsp[column+'-'+num][obj.canteranames.index('OH-')] ]+\
                                        [cl_minus]+\
                                        [jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]] # equivalent to count-nsy+naqy
    
                
                stream[naq:]        = [y[count+nsy+obj.canteravars.index('(HA)2(org)')]]+\
                                        [obj.nsp[column+'-'+num][obj.canteranames.index('dodecane')] ]+\
                                        [jk for jk in y[count+(nsy)+obj.canteravars.index('(HA)2(org)')+1:count+(2*nsy)] ]


            obj.mix.species_moles    = [max(0,ii) for ii in stream] #stream.copy() 
    
            try:
                obj.mix.equilibrate('TP',log_level=0) # default 
                y[count:count+nsy] = [obj.mix.species_moles[obj.canteranames.index(ij)] for ij in obj.canteravars]    #  # update aq and org variable vector
                
            except:
                print ('some people got Gibbs problems')
                pass
    
                        
    #        obj.mix.equilibrate('TP',log_level=0, maxsteps=100, maxiter=20) # default                      
            #y[count:count+nsy] = [obj.mix.species_moles[obj.canteranames.index(ij)] for ij in obj.canteravars]    #  # update aq and org variable vector
    #        except:
            
            count += nsy

    return yI-y  



    
    
    

def ecolumncheck (yI, obj, num, column, Ns, ree=[]) : # This should be equivalent to actual Lyon et al 4 recycling

    y                       = np.array([i for i in yI])
    naq                     = obj.naq
    nsy                     = obj.nsy
    norgy                   = obj.norgy
    stream                  = np.zeros(obj.ns)
    
###############################################################################
#    print(Ns, len(y))
    
    count = 0 # adjust reference to column inlet stream 
    stagestatus = [True for i in range(Ns)]


    if Ns == 1: # single stage column
        stream[:naq]        = obj.nsp[column+'-'+num][:naq] # aqueous feed 
        stream[naq:]        = obj.nsp[column+'-'+num][naq:]
        obj.mix.species_moles    = stream.copy() 
        try:
            obj.mix.equilibrate('TP',log_level=0) # default  # maxsteps=100, maxiter=20
        except:
            stagestatus[0] = False
            pass
        y[count:count+nsy] = [obj.mix.species_moles[obj.canteranames.index(ij)] for ij in obj.canteravars]    #  # update aq and org variable vector 

        
    else:
    
        for i in range(1,Ns+1): # for each stage in column extraction   
            
            if i == 1: # feed aqueous stage 1  
                stream[:naq]        = obj.nsp[column+'-'+num][:naq] # aqueous feed 
                stream[naq:]        = [y[count+nsy+obj.canteravars.index('(HA)2(org)')]] +\
                                        [obj.nsp[column+'-'+num][obj.canteranames.index('dodecane')] ]+\
                                        [jk for jk in y[count+(nsy)+obj.canteravars.index('(HA)2(org)')+1:count+(2*nsy)] ]
                                        
            elif i == Ns : #feed organic stage N  
                stream[naq:]        = obj.nsp[column+'-'+num][naq:]
                
                cl_minus            = y[count-nsy+obj.canteravars.index('H+')] + 3*sum([jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]])            
                stream[:naq]        = [obj.nsp[column+'-'+num][obj.canteranames.index('H2O(L)')]] +\
                                        [y[count-nsy+obj.canteravars.index('H+')]]+ [obj.nsp[column+'-'+num][obj.canteranames.index('OH-')] ]+\
                                        [cl_minus]+\
                                        [jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]] # counts from after Cl up until the last ree
                                    # The use of count-nsy follows the logic that at the beginning of the second stage (N=2), count has been updated by nsy, but
                                    # this indexing for y is just starting (at zero); so to correct for this lag, we have to subtract nsy from count each time                                
                           
            else:
                cl_minus            = y[count-nsy+obj.canteravars.index('H+')] + 3*sum([jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]])            
                stream[:naq]        = [obj.nsp[column+'-'+num][obj.canteranames.index('H2O(L)')]] +\
                                        [y[count-nsy+obj.canteravars.index('H+')]]+ [obj.nsp[column+'-'+num][obj.canteranames.index('OH-')] ]+\
                                        [cl_minus]+\
                                        [jk for jk in y[count-nsy+obj.canteravars.index('H+')+1:count-norgy]] # equivalent to count-nsy+naqy
    
                
                stream[naq:]        = [y[count+nsy+obj.canteravars.index('(HA)2(org)')]]+\
                                        [obj.nsp[column+'-'+num][obj.canteranames.index('dodecane')] ]+\
                                        [jk for jk in y[count+(nsy)+obj.canteravars.index('(HA)2(org)')+1:count+(2*nsy)] ]


            obj.mix.species_moles    = stream.copy() 
    
            try:
                obj.mix.equilibrate('TP',log_level=0) # default  
            except:
                stagestatus[i-1] = False
                pass    
                        
    #        obj.mix.equilibrate('TP',log_level=0, maxsteps=100, maxiter=20) # default                      
            y[count:count+nsy] = [obj.mix.species_moles[obj.canteranames.index(ij)] for ij in obj.canteravars]    #  # update aq and org variable vector
    #        except:
            
            count += nsy

    return stagestatus # yI-y  




def flatten(iter):
  if type(iter) == dict:
    return np.array(list(iter.values())).reshape(1, -1)[0]
  elif type(iter) == list:
    return np.array(iter).reshape(1, -1)[0]


def get_sums(d,key):
  i = len(d) 	 #matrix rows
  j = len(d[key])  #matrix cols
  s = flatten(d) #1d list of dictionary `d`
  return list(np.sum(s.reshape(i, j), axis=0)) #sum cols of s





def highest_remaining(values, remove_n_lowest):
  '''
    values: numpy array containing values of components
  '''
  if type(values).__module__ == np.__name__: #values = numpy array
    vals = values.tolist()
  else:
    vals = values
  inds = [i for i in range(len(vals))]
  if remove_n_lowest > len(vals):
    indices = []
  elif remove_n_lowest >= 0:
    pairs = [(i, val) for i, val in zip(inds, vals)]

    remaining = sorted(pairs, key=operator.itemgetter(1), reverse=True)[:-remove_n_lowest]
    indices = [i for i, val in sorted(remaining, key=operator.itemgetter(0))]
  else:
    indices = inds
  return indices

""" root(tear()) timings (seconds):
    
lm: 39
broyden1: 11
broyden2: no convergence
anderson: 11
linearmixing: no convergence
diagbroyden:10
excitingmixing:4
krylov:8
hybr: 13
df-sane: 12

"""
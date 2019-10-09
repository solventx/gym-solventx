# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:05:32 2019

@author: ciloeje
"""

import os, sys

import numpy   as np
import pandas  as pd
import cantera as ct

from scipy.optimize import root

def get_simulator_path():
    gym_solventx_directory = ''
    for directory in sys.path:
        if 'gym-solventx' in directory:
            ind = directory.index('gym-solventx')
            gym_solventx_directory = directory[:ind+12] + '/gym_solventx/'
    return gym_solventx_directory
    
class solvent_extraction:
    
    #dynamically obtain path to environment dependencies
    dirlocation = get_simulator_path()
    
    xml                     = dirlocation + 'envs/methods/xml/PC88A_HCL_NdPr.xml'
    datacsv                 = dirlocation + 'envs/methods/input/data.csv'
    bndcsv                  = dirlocation + 'envs/methods/input/bounds.csv'
    module_id               = 2
    strip_group             = 4

    
    var_names               = ['(HA)2(org)',	'H+ Extraction', 'H+ Scrub',	'H+ Strip',	'OA Extraction',	'OA Scrub',	'OA Strip', 'Recycle','Extraction','Scrub', 'Strip']
    dimension              = len(var_names)

    solvs                   = ['H2O(L)', '(HA)2(org)', 'dodecane']
    rees                    = {1:['Nd','Pr','Ce','La'],2:['Nd','Pr'],3:['Ce','La'], 4:['Nd'],5:['Ce']}
    phasenames              = ['HCl_electrolyte','PC88A_liquid']

    

    def __init__(self, scale, feedvol, dimension=dimension, data=datacsv,
                 xml=xml, bounds=bndcsv, module_id=module_id, 
                 strip_group=strip_group,varnames=var_names, solvents=solvs, ree=rees, phasenames=phasenames):
                 
        """Constructor.
        """

#         Set required data
        self.dimension      = dimension  
        self.df             = pd.read_csv(data)
        self.scale          = scale

        # feed data must be in g/l for components and ml for volumetric flow
        self.bounds       = pd.read_csv(bounds)

        self.lower      = self.bounds[varnames].iloc[0].values
        self.upper      = self.bounds[varnames].iloc[1].values

 
        self.xml            = xml  
        self.phase_names    = phasenames
        self.phase          = ct.import_phases(xml,self.phase_names) 

        # Derived and/or reusable system parameters        
        self.column_names   = ['Extraction','Scrub','Strip'] # Column name
        self.solv           = solvents # solvent list -.i.e., electrolyte, extractant and organic diluent
        
        # ree by modules
        self.ree            = ree[module_id] # (rare earth) metal list
        self.ree_strip      = ree[strip_group] # Strip target
        self.is_ree         = [1 if re in self.ree_strip else 0 for re in self.ree ]
                   
        # Cantera indices
        self.mix                 = ct.Mixture(self.phase)
        self.n_species           = pd.DataFrame() # feed streams (aq and org) for each column   

        
        # outer optimization parameter names        
        self.varnames       = varnames #+ [i for i in ree]
        self.variables      = self.df[varnames].values[0]
        
        # Feed concentrations
        self.feed           = self.df[self.ree].values[0] # Nd, Pr concentrations, g/L
        self.feed_total     = sum([ik for ik in self.feed])
        self.fracbnds       = [0.55, 0.85] # 
        self.fbounds        = [ik*self.feed_total for ik in self.fracbnds] # g/l
        
        self.get_mw() # molecular weights for rare earth elements and solvents

        self.feedvol        = feedvol * scale
        self.rhoslv         = [1000, 960, 750]
        
        self.purity_spec    = .999
        self.recov_spec     = .99

        self.org                 = self.mix.phase_index(self.phase_names[1])
        self.aq                  = self.mix.phase_index(self.phase_names[0])
        
        self.HA_Index       = self.mix.species_index(self.org,'(HA)2(org)') # index of extractant in canera species list
        self.Hp_Index       = self.mix.species_index(self.aq,'H+') # index of H+ in cantera species list
        self.Cl_Index       = self.mix.species_index(self.aq,'Cl-') # index of Cl in cantera species list

        

    def get_mw(self):
        """ Initialize parameters for cantera simulation"""
        
        mx                  = self.mix
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

        self.mwslv             = mwslv            
        self.mwre              = mwre # ree molecular weight
        


    def update_flows(self, x):
        
     
        varnames            = self.varnames #+ [re for re in self.ree]        
        feedvol             = self.feedvol # l/unit time
        rhoslv              = self.rhoslv               
        mwre                = self.mwre
        mwslv               = self.mwslv
        rhoslv              = self.rhoslv       
        nre                 = np.zeros(len(self.ree)) # REE specie moles
        salts               = np.zeros(len(self.ree)) # neutral complexe moles

#        Flows
        orgvol              = feedvol * x[varnames.index('OA Extraction')]
        aqvols              = [feedvol,orgvol/x[varnames.index('OA Scrub')], orgvol/x[varnames.index('OA Strip')] ]

#        Compositions   
        vol_HA              = orgvol *  x[varnames.index('(HA)2(org)')]
        vol_dodec           = orgvol - vol_HA
        n_HA                = vol_HA * rhoslv[self.solv.index('(HA)2(org)')]/mwslv[self.solv.index('(HA)2(org)')]
        n_dodec             = vol_dodec * rhoslv[self.solv.index('dodecane')]/mwslv[self.solv.index('dodecane')]            


        for k in range(len(self.column_names)):

            n_H2O           = aqvols[k] * rhoslv[self.solv.index('H2O(L)')]/mwslv[self.solv.index('H2O(L)')] #[l/s]*[g/l]/[g/mol]
            n_Hp            = aqvols[k] * (x[varnames.index('H+ '+self.column_names[k])])  #[l/s]*[g/l]/[g/mol] = mol
            
            if k==0:
                for re in self.ree:                     
#                    nre[self.ree.index(re)] = aqvols[k] *self.df[re]/ mwre[self.ree.index(re)] # [l/s]*[g/l]/[g/mol] = mol
                    nre[self.ree.index(re)] = aqvols[k] *self.feed[self.ree.index(re)]/ mwre[self.ree.index(re)] # [l/s]*[g/l]/[g/mol] = mol

            elif k==1:
                for re in self.ree:                     
#                    nre[self.ree.index(re)] = self.is_ree[self.ree.index(re)] *(x[varnames.index('Recycle')])* aqvols[k] *self.df[re]/ mwre[self.ree.index(re)] # [l/s]*[g/l]/[g/mol] = mol
                    nre[self.ree.index(re)] = self.is_ree[self.ree.index(re)] *(x[varnames.index('Recycle')])* aqvols[k] *self.feed[self.ree.index(re)]/ mwre[self.ree.index(re)] # [l/s]*[g/l]/[g/mol] = mol
                
            else:
                for re in self.ree:
                    nre[self.ree.index(re)] = 0.0                
                            
            n_Cl            = 3*(sum(nre)) + n_Hp # Cl- mole balance, all REEs come in as chlorides from leaching

            n_specs         = [n_H2O,n_Hp,0,n_Cl]+[ii for ii in nre]+[n_HA,n_dodec] +[ij for ij in salts] 

            self.n_species[self.column_names[k]]       = n_specs       


            
            

    def create_column(self,x) :

        y,Ny, Ns            = [],[],[]
        varnames            = self.varnames
        x=self.variables
 
        for k in range(len(self.column_names)):

            n_specs = [i for i in self.n_species[self.column_names[k]] ]
            y.append(n_specs*(int(round(x[varnames.index(self.column_names[k])]))+1) )# initialize
            Ny.append(len(n_specs*(int(round(x[varnames.index(self.column_names[k])]))+1)) )           
            Ns.append(int(round(x[varnames.index(self.column_names[k])])) ) 
        
        self.y              = np.array([ii for jj in y for ii in jj])        
        self.Ny             = [sum(Ny[:i+1]) for i in range(len(Ny))]
        self.Ns             = Ns



    def update_column(self, x):
        ns      = self.mix.n_species
        y = self.y.copy()
        Nyi = self.Ny.copy()


        Ns = [int(i) for i in x[self.varnames.index('Extraction'):] ]
        ybucket = []
        Nybucket = []

                
        icount = 0
        for col in range(len(Ns)):
            yi = [i for i in y[icount:Nyi[col]] ]

            
            if Ns[col] > self.Ns[col]: # add stages
                n = Ns[col] - self.Ns[col]
                stage = [i for i in yi[-ns:] ]
                ybucket.append(yi + n*stage)
                Nybucket.append(len(yi +n*stage))
                icount = Nyi[col]  

                
            elif Ns[col] < self.Ns[col]:
                n = self.Ns[col] - Ns[col]
                lim = int(n * ns)
                ybucket.append(yi[:-lim])
                Nybucket.append(len(yi[:-lim]))
                icount = Nyi[col]  
                
            else:
                ybucket.append(yi)
                Nybucket.append(len(yi)) #
                icount = Nyi[col]  
            
        self.y = np.array([i for j in ybucket for i in j])
        self.Ny = [sum(Nybucket[:i+1]) for i in range(len(Nybucket))] # 
        self.Ns = Ns.copy()

        
        

    def update_system(self, x):
        
        self.update_flows(x)
        self.update_column(x)
        
    
    def update_var_by_index(self, indices, values):
        
        """ var_names = ['(HA)2(org)',	'H+ Extraction', 'H+ Scrub',
                         Strip',	'OA Extraction',	'OA Scrub',	'OA Strip',
                         'Recycle','Extraction','Scrub', 'Strip'] """
        
        for item, jtem in zip(indices, values):
            self.variables[item] = jtem
            self.update_system(self.variables)


    def update_var_by_label(self, labels, values):
     
        for item, jtem in zip(labels, values):
            self.variables[self.varnames.index(item)] = jtem
            self.update_system(self.variables)


    def recovery(self, x): 
          
        mx                  = self.mix
        aq                  = mx.phase_index(self.phase_names[0])
        ns                  = mx.n_species 

        Ny                  = self.Ny
        ree                 = self.ree
        nsp                 = self.n_species
        yo                  = self.y
            
        strip_out,\
        strip_recov         = np.zeros(len(ree)), np.zeros(len(ree))        
        feed_in             = np.zeros(len(ree))      
        scrub_out, \
        scrub_recov         = np.zeros(len(ree)), np.zeros(len(ree))     
        ext_out, ext_recov  = np.zeros(len(ree)), np.zeros(len(ree)) 
        
        

        for re in ree:
            # control volume tight around all three columns - accounts for recycle
            strip_out[ree.index(re)]    = yo[Ny[-1] - ns + mx.species_index(aq,re+'+++')] * self.mwre[ree.index(re)]
            feed_in[ree.index(re)]      = (nsp['Extraction'][mx.species_index(aq,re+'+++')]+ nsp['Scrub'][mx.species_index(aq,re+'+++')] ) *self.mwre[ree.index(re)]
            strip_recov[ree.index(re)]  = (strip_out[ree.index(re)]) / feed_in[ree.index(re)] #(data[re] /self.mwre[ree.index(re)])
                                          
            scrub_out[ree.index(re)]    = (yo[Ny[1] - ns + mx.species_index(aq,re+'+++')] )* self.mwre[ree.index(re)]
            scrub_recov[ree.index(re)]  = scrub_out[ree.index(re)]/feed_in[ree.index(re)]#(data[re] /self.mwre[ree.index(re)]) 
                            
            ext_out[ree.index(re)]      = (yo[Ny[0] - ns + mx.species_index(aq,re+'+++')] ) * self.mwre[ree.index(re)]
            ext_recov[ree.index(re)]    = ext_out[ree.index(re)]/feed_in[ree.index(re)]#(data[re] /self.mwre[ree.index(re)])

        self.strip_recov    = strip_recov
        self.scrub_recov    = scrub_recov
        self.ext_recov      = ext_recov
                                            
        if sum(strip_out)==0:
            self.strip_pur  = [0 for re in strip_out]
        else:
            self.strip_pur  = [re/sum(strip_out ) for re in strip_out]
        
        if sum(scrub_out)==0:
             self.scrub_pur = [0 for re in scrub_out] 
        else: 
            self.scrub_pur  = [re/sum(scrub_out ) for re in scrub_out] 
            
        if sum(ext_out)==0:
            self.ext_pur    = [0 for re in ext_out] 
        else:
            self.ext_pur    = [re/sum(ext_out ) for re in ext_out] 
        


    def errorcheck(self, x): # call this function 
        

        status             = ecolumncheck (self.y, self.n_species, self.Ns, self.Ny,
                                                           self.column_names, self.xml, self.phase_names,
                                                           self.Hp_Index, self.Cl_Index, self.HA_Index, x[self.varnames.index('Recycle')])
                            
        return status      


        
    def evaluate(self, x):
        
        resy                = root(ecolumn, self.y.copy(), args=(self.n_species, self.Ns, self.Ny,
                                                           self.column_names, self.xml, self.phase_names,
                                                           self.Hp_Index, self.Cl_Index, self.HA_Index, x[self.varnames.index('Recycle')]), method='hybr', options=None) #options={'disp':True, 'maxfev':15} 

        self.y              = resy.x.copy()
        self.status         = resy.success # bool, True or False, checks if root finding function was successful
        self.stage_status   = self.errorcheck(x) # returns list that shows final status of each 'stage' solution as True or False     
        self.recovery(x)      
        
#equilibrates each stage of a column and calculates error from a guess to the actual. yI is guess
def ecolumn (yI, nsp, Ns, Ny, header,xml, phase_names, Hp_Index, Cl_Index, HA_Index, recycle) : # note the change. xis ore multiplicative fractions now
    
    y                       = yI.copy()

    ct_handle               = ct.import_phases(xml,phase_names)   
    mx                      = ct.Mixture(ct_handle)
    aq                      = mx.phase_index(phase_names[0])
    org                     = mx.phase_index(phase_names[1])
    naq                     = mx.phase(aq).n_species # number of aqueous species
    norg                    = mx.phase(org).n_species
    ns                      = mx.n_species 
 
    stream                  = np.zeros(ns) 
    Hp_I                    = Hp_Index
    Cl_I                    = Cl_Index
    HA_I                    = HA_Index
    
    
    y[Ny[0]-norg:Ny[0]]     =   nsp['Extraction'][naq:] # organic feed                
        
#    ################# update aqueous feeds ################################### 
#                

    icount                  = 0
               
    for j in range(len(Ny)): 

        y[icount:icount+naq]    = nsp[header[j]][:naq]  # Feed stays constant
            
        y[icount+Cl_I]          =  y[icount+Hp_I] + 3*sum(y[icount + Cl_I+1: icount + HA_I]) # ensure mass balance for chloride
            
        icount                  = Ny[j] # step to next column 
        
########################################## aqueous feed driven solution ##############################
                    

    count = 0 # adjust reference to column inlet stream  
    
    for mk in range(len(Ns)) : # Ns = column stages in recovery plant        
        N = Ns[mk] # number of stages in column m  
        
        for i in range(N): # for each stage in column m   
            
            stream[:naq]        = y[(i*ns)+count:(i*ns)+count+naq] # aqueous feed in stage i, column m
            stream[naq:]        = y[(i+1)*ns+count+naq:((i+2)*ns)+count] # organic feed in stage i, column m
            mx.species_moles    = stream 

            try:
                mx.equilibrate('TP',log_level=0) # default          
                
                y[(i+1)*ns+count:(i+1)*ns+count+naq] = mx.species_moles[:naq] #mx.species_moles[:naq]  # update aq variable vector corresponding to stage i, column m exit stream
                y[(i)*ns+naq+count:(i+1)*ns + count] = mx.species_moles[naq:] #mx.species_moles[naq:]  # update org variable vector corresponding to stage i, column m exit stream
                

            except:
                pass

#         Update the corresponding org tear stream
                
        if mk < len(Ny)-1:
            
            y[Ny[mk+1]-norg:Ny[mk+1] ]   = y[count+naq:count+ns ] 
 
        count                   = Ny[mk] # count + Nx[m] 


    return yI-y

 
def ecolumncheck (yI, nsp, Ns, Ny, header,xml, phase_names, Hp_Index, Cl_Index, HA_Index, recycle) : # note the change. xis ore multiplicative fractions now
    
    y                       = yI.copy()

    ct_handle               = ct.import_phases(xml,phase_names)   
    mx                      = ct.Mixture(ct_handle)
    aq                      = mx.phase_index(phase_names[0])
    org                     = mx.phase_index(phase_names[1])
    naq                     = mx.phase(aq).n_species
    norg                    = mx.phase(org).n_species
    ns                      = mx.n_species 
 
    stream                  = np.zeros(ns) 
    Hp_I                    = Hp_Index
    Cl_I                    = Cl_Index
    HA_I                    = HA_Index
    
    
    y[Ny[0]-norg:Ny[0]]     =   nsp['Extraction'][naq:] # organic feed                
        
#    ################# update aqueous feeds ###################################        

    icount                  = 0
               
    for j in range(len(Ny)): 

        y[icount:icount+naq]    = nsp[header[j]][:naq]  # Feed stays constant
            
        y[icount+Cl_I]          =  y[icount+Hp_I] + 3*sum(y[icount + Cl_I+1: icount + HA_I]) # ensure mass balance for chloride
            
        icount                  = Ny[j] # step to next column 

        
########################################## aqueous feed driven solution ##############################
                    

    count = 0 # adjust reference to column inlet stream  
    stagestatus = [True for i in range(sum(Ns))]
    cfs = 0 # feed stage index for current column - 0 for extraction, 0+Ns[0] for scrub, etc.
    for mk in range(len(Ns)) : # Ns = column stages in recovery plant        
        N = Ns[mk] # number of stages in column m  
        
        for i in range(N): # for each stage in column m   
            
            stream[:naq]        = y[(i*ns)+count:(i*ns)+count+naq] # aqueous feed in stage i, column m
            stream[naq:]        = y[(i+1)*ns+count+naq:((i+2)*ns)+count] # organic feed in stage i, column m
            mx.species_moles    = stream 

            try:
                mx.equilibrate('TP',log_level=0) # default          
                
                y[(i+1)*ns+count:(i+1)*ns+count+naq] = mx.species_moles[:naq] #mx.species_moles[:naq]  # update aq variable vector corresponding to stage i, column m exit stream
                y[(i)*ns+naq+count:(i+1)*ns + count] = mx.species_moles[naq:] #mx.species_moles[naq:]  # update org variable vector corresponding to stage i, column m exit stream
                

            except:
                stagestatus[cfs+i] = False
                pass

#         Update the corresponding org tear stream
        cfs = Ns[mk] # update error check count 
        
        if mk < len(Ny)-1:
            
            y[Ny[mk+1]-norg:Ny[mk+1] ]   = y[count+naq:count+ns ] 
 
        count                   = Ny[mk] # count + Nx[m] 


    return stagestatus #yI-y
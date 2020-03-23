# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:43:20 2020

@author: splathottam
"""

DEFAULT_module0_x = [0.65,1e-5,1e-5,1.2,5.9,1.2,1.6,5,6,6,4]
DEFAULT_module1_x = [1e-5,1e-5,1.2,6,1,4,0.51,1,8,2]
DEFAULT_module2_x = [1e-5,1e-5,1.2,6,1,1,0.01,4,9,2]


valid_processes = {'a':{'input':['Nd','Pr','Ce','La'],
                        'strip':['Nd','Ce'],
                        'modules':{"0":{"strip_group":["Nd","Pr"],"x":DEFAULT_module0_x},
                                  "1":{"strip_group":["Nd"],"x":DEFAULT_module1_x},
                                  "2":{"strip_group":["Ce"],"x":DEFAULT_module2_x}}
                        },
                   'b':{'input':['Nd','Pr'],
                        'strip':['Nd'],
                        'modules':{"1":{"strip_group":["Nd"],"x":DEFAULT_module1_x}}
                        },
                   'c':{'input':['Ce','La'],
                        'strip':['Ce'],
                        'modules':{"2":{"strip_group":["Ce"],"x":DEFAULT_module2_x}}
                        },
                   'd':{'input':['Nd','Pr','Ce','La'],
                        'strip':['Nd'],
                        'modules':{"0":{"strip_group":["Nd","Pr"],"x":DEFAULT_module0_x},
                                  "1":{"strip_group":["Nd"],"x":DEFAULT_module1_x}}
                        }                   
                    }

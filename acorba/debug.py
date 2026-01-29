# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:56:03 2021

@author: Nelson
"""

def scanner_debug(install_dir='E:/ACORBA/v1.2'):
    import argparse
    import os
    args = argparse.Namespace(input_folder="E:/Pic libraries/test bank/Vincent/test2 new model/test v1.2/Scanner/4tf",
                          rootplot='False',
                          savesegmentation="False",
                          exp_type="Scanner",
                          normalization="True",
                          prediction="None",
                          method="Deep Machine Learning",
                          superaccuracy="False",
                          binary_folder="",
                          custom="",
                          smooth="False",
                          tradmethod="Entropy",
                          save_lenghts="False",
                          scale="1",
                          circlepix="40",
                          broken="100",
                          vector="10"
                          )
    os.chdir(install_dir)
    return args

def micro_debug(install_dir='E:/ACORBA/v1.2'):
    import argparse
    import os
    args = argparse.Namespace(input_folder="E:/Pic libraries/test bank/Vincent/test2 new model/test v1.2/micro sand/test with a bug in one timeframe/",
                          rootplot='False',
                          savesegmentation="False",
                          exp_type="Scanner",
                          normalization="True",
                          prediction="None",
                          method="Deep Machine Learning",
                          superaccuracy="False",
                          binary_folder="",
                          custom="",
                          smooth="False",
                          tradmethod="Entropy",
                          #save_lenghts="True",
                          #scale="1"
                          circlepix="40",
                          broken="100",
                          vector="10"
                          )
    os.chdir(install_dir)
    return args




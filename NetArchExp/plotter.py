# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:13:42 2023

@author: Peter
"""

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import shutil
import os

#tanh 0-60
#sigmoid 61-121
#relu 122-182
#swish 183-243

Doc=pd.read_csv('Documentation.csv')

print(Doc)

standardxlabel = 'Iteration [1-6 Layer] und [1-40 Neuronen]'
barxlabel = '[6 Layer + 1-40 Neuronen] und [10 Layer + 100 Neuronen]'

origin = 'C:/Users/Peter/workspace/bachelor/'
target = 'C:/Users/Peter/workspace/bachelor/graphen/'
graphlegend = ["1Layer","2Layer","3Layer","4Layer","5Layer","6Layer"]
z=1

xlist = [0,61,122,183]
for x in xlist:
    
    
    if x==0:
        name='tanh'
    if x==61:
        name='sigmoid'
    if x==122:
        name='relu'
    if x==183:
        name='swish'


    
    ylist=['adamtime','lbfgstime','totaltime']
    for t in ylist:
        timegraph=Doc.iloc[x:x+10].plot(y=t, title=str(name))
        for w in range (1,6,1):
              Doc.iloc[x+w*10:x+(w+1)*10].plot(y=t,ax=timegraph, title=str(name)+' '+str(t))     
        timegraph.set_ylabel("Zeit in s")
        timegraph.set_xlabel(str(standardxlabel))
        timegraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
        
        plt.savefig(str(target)+str(z)+str(name)+str(t)+'.png')
        z=z+1
    
#-------------------------------------------------------------------------------------------------------------           
    
    ylist=['totaltime']
    for t in ylist: 
        timegraph=Doc.iloc[x+50:x+61].plot.bar(y=t, title=str(name)+' '+str(t))
        timegraph.set_ylabel("Zeit in s")
        timegraph.set_xlabel(str(barxlabel))
        timegraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(t)+'barplot.png')
        z=z+1
        
#-------------------------------------------------------------------------------------------------------------   

    ylist=['steps']
    for s in ylist:
        stepsgraph=Doc.iloc[x:x+10].plot(y=s, title=str(name))
        for w in range (1,6,1):
            Doc.iloc[x+w*10:x+(w+1)*10].plot(y=s,ax=stepsgraph, title=str(name)+' '+str(s))      
        stepsgraph.set_ylabel("Anzahl Schritte")
        stepsgraph.set_xlabel(str(standardxlabel))
        stepsgraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
    
        plt.savefig(str(target)+str(z)+str(name)+str(s)+'.png')
        z=z+1
    
#-------------------------------------------------------------------------------------------------------------    
    
    ylist=['steps']
    for s in ylist: 
        stepsgraph=Doc.iloc[x+50:x+61].plot.bar(y=s, title=str(name)+' '+str(s))
        stepsgraph.set_ylabel("Anzahl Schritte")
        stepsgraph.set_xlabel(str(barxlabel))
        stepsgraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(s)+'barplot.png')
        z=z+1

#-------------------------------------------------------------------------------------------------------------       
    
    ylist=['adamtrainloss1', 'adamtrainloss2', 'adamtrainloss3',
            'adamtestloss1', 'adamtestloss2', 'adamtestloss3',
            'lbfgstrainloss1', 'lbfgstrainloss2', 'lbfgstrainloss3',
            'lbfgstestloss1', 'lbfgstestloss2', 'lbfgstestloss3']
    for l in ylist:
        lossgraph=Doc.iloc[x:x+10].plot(y=l, title=str(name))
        for w in range (1,6,1):
            Doc.iloc[x+w*10:x+(w+1)*10].plot(y=l,ax=lossgraph, title=str(name)+' '+str(l))          
        lossgraph.set_ylabel("Verlust")
        lossgraph.set_yscale('log')
        lossgraph.set_xlabel(str(standardxlabel))
        lossgraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
        
        plt.savefig(str(target)+str(z)+str(name)+str(l)+'.png')
        z=z+1
        
        lossbargraph=Doc.iloc[x+50:x+61].plot.bar(y=l, title=str(name)+' '+str(l))
        lossbargraph.set_ylabel("Verlust")
        lossbargraph.set_yscale('log')
        lossbargraph.set_xlabel(str(barxlabel))
        lossbargraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(l)+'barplot.png')
        z=z+1
    
#-------------------------------------------------------------------------------------------------------------      
    
    ylist=['diffpoint', 'mean']
    for d in ylist:
        diffgraph=Doc.iloc[x:x+10].plot(y=d, title=str(name))
        for w in range (1,6,1):
            Doc.iloc[x+w*10:x+(w+1)*10].plot(y=d,ax=diffgraph, title=str(name)+' '+str(d))
        diffgraph.set_ylabel(str(d)+' '+"Abweichung")
        diffgraph.set_yscale('log')
        diffgraph.set_xlabel(str(standardxlabel))
        diffgraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
        
        plt.savefig(str(target)+str(z)+str(name)+str(d)+'.png')
        z=z+1
        
#-------------------------------------------------------------------------------------------------------------        
        
    ylist=['mean']
    for d in ylist:
        diffgraph=Doc.iloc[x+50:x+61].plot.bar(y=d, title=str(name)+' '+str(d))
        diffgraph.set_ylabel("Mittlere Abweichung")
        diffgraph.set_yscale('log')
        diffgraph.set_xlabel(str(barxlabel))
        diffgraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(d)+'barplot.png')
        z=z+1
        
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:13:42 2023

@author: Peter
"""

import pandas as pd
import matplotlib.pyplot as plt

#tanh 0-60
#sigmoid 61-121
#relu 122-182
#swish 183-243

Doc=pd.read_csv('Documentation.csv')

print(Doc)

standardxlabel = 'Anzahl Neuronen'
barxlabel = 'Anzahl Neuronen'
bartitle = '6Layer/1-40Neurons--10Layer/100Neurons'

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
        timegraph=Doc.iloc[x:x+10].plot(y=t, x='neurons',  title=str(name))
        for w in range (1,6,1):
              Doc.iloc[x+w*10:x+(w+1)*10].plot(y=t, x='neurons', ax=timegraph, title=str(name)+' '+str(t))     
        timegraph.set_ylabel("Zeit in s")
        timegraph.set_xlabel(str(standardxlabel))
        timegraph.grid('on')
        legend = timegraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
        
        plt.savefig(str(target)+str(z)+str(name)+str(t)+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
        z=z+1
    
#-------------------------------------------------------------------------------------------------------------           
    
    ylist=['totaltime']
    for t in ylist: 
        timegraph=Doc.iloc[x+50:x+61].plot.bar(y=t, x='neurons', title=str(name)+' '+str(t)+' | '+str(bartitle))
        timegraph.set_ylabel("Zeit in s")
        timegraph.set_xlabel(str(barxlabel))
        timegraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(t)+'barplot.png', bbox_inches='tight')
        z=z+1
        
#-------------------------------------------------------------------------------------------------------------   

    ylist=['steps']
    for s in ylist:
        stepsgraph=Doc.iloc[x:x+10].plot(y=s, x='neurons', title=str(name))
        for w in range (1,6,1):
            Doc.iloc[x+w*10:x+(w+1)*10].plot(y=s, x='neurons', ax=stepsgraph, title=str(name)+' '+str(s))      
        stepsgraph.set_ylabel("Anzahl Schritte")
        stepsgraph.set_xlabel(str(standardxlabel))
        stepsgraph.grid('on')
        legend = stepsgraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
    
        plt.savefig(str(target)+str(z)+str(name)+str(s)+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
        z=z+1
    
#-------------------------------------------------------------------------------------------------------------    
    
    ylist=['steps']
    for s in ylist: 
        stepsgraph=Doc.iloc[x+50:x+61].plot.bar(y=s, x='neurons', title=str(name)+' '+str(s)+' | '+str(bartitle))
        stepsgraph.set_ylabel("Anzahl Schritte")
        stepsgraph.set_xlabel(str(barxlabel))
        stepsgraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(s)+'barplot.png', bbox_inches='tight')
        z=z+1

#-------------------------------------------------------------------------------------------------------------       
    
    ylist=['adamtrainloss1', 'adamtrainloss2', 'adamtrainloss3',
            'adamtestloss1', 'adamtestloss2', 'adamtestloss3',
            'lbfgstrainloss1', 'lbfgstrainloss2', 'lbfgstrainloss3',
            'lbfgstestloss1', 'lbfgstestloss2', 'lbfgstestloss3']
    for l in ylist:
        lossgraph=Doc.iloc[x:x+10].plot(y=l, x='neurons', title=str(name))
        for w in range (1,6,1):
            Doc.iloc[x+w*10:x+(w+1)*10].plot(y=l, x='neurons', ax=lossgraph, title=str(name)+' '+str(l))          
        lossgraph.set_ylabel("Verlust")
        lossgraph.set_yscale('log')
        lossgraph.set_xlabel(str(standardxlabel))
        lossgraph.grid('on')
        legend = lossgraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
        
        plt.savefig(str(target)+str(z)+str(name)+str(l)+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
        z=z+1
        
        lossbargraph=Doc.iloc[x+50:x+61].plot.bar(y=l, x='neurons', title=str(name)+' '+str(l)+' | '+str(bartitle))
        lossbargraph.set_ylabel("Verlust")
        lossbargraph.set_yscale('log')
        lossbargraph.set_xlabel(str(barxlabel))
        lossbargraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(l)+'barplot.png', bbox_inches='tight')
        z=z+1
    
#-------------------------------------------------------------------------------------------------------------      
    
    ylist=['diffpoint', 'mean']
    for d in ylist:
        diffgraph=Doc.iloc[x:x+10].plot(y=d, x='neurons', title=str(name))
        for w in range (1,6,1):
            Doc.iloc[x+w*10:x+(w+1)*10].plot(y=d, x='neurons', ax=diffgraph, title=str(name)+' '+str(d))
        diffgraph.set_ylabel(str(d)+' '+"Abweichung")
        diffgraph.set_yscale('log')
        diffgraph.set_xlabel(str(standardxlabel))
        diffgraph.grid('on')
        legend = diffgraph.legend(graphlegend, prop={'size':10}, bbox_to_anchor=(1.0,0.5))
        
        plt.savefig(str(target)+str(z)+str(name)+str(d)+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
        z=z+1
        
#-------------------------------------------------------------------------------------------------------------        
        
    ylist=['mean']
    for d in ylist:
        diffgraph=Doc.iloc[x+50:x+61].plot.bar(y=d, x='neurons', title=str(name)+' '+str(d)+' | '+str(bartitle))
        diffgraph.set_ylabel("Mittlere Abweichung")
        diffgraph.set_yscale('log')
        diffgraph.set_xlabel(str(barxlabel))
        diffgraph.get_legend().remove()
        
        plt.savefig(str(target)+str(z)+str(name)+str(d)+'barplot.png', bbox_inches='tight')
        z=z+1
        
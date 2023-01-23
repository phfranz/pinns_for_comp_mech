#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys, os
import subprocess

# activationfunction loop
acticounter = 1
while acticounter < 5:
   
    # layer loop
    layer = 2
    while layer < 4:
        
        # neuron loop   
        neurons = 3
        while neurons < 4: 
            
            if acticounter==1 :
                actifunc='tanh'
            if acticounter==2 :
                actifunc='sigmoid'
            if acticounter==3 :
                actifunc='relu'   
            if acticounter==4 :
                actifunc='swish'   
                
                
            os.system('copy BaseheatEq.py actifuncfile.py' )
            
            with open('BaseheatEq.py') as f:
                newText=f.read().replace('ACTIVATIONFUNCTION',str(actifunc))
                
            with open('actifuncfile.py', "w") as f:
                    f.write(newText)
                    
            with open('actifuncfile.py') as f:
                newText=f.read().replace('NUMBERLAYER', str(layer))
                
            with open('layerfile.py', "w") as f:
                    f.write(newText)
            
            with open('layerfile.py') as f:
               newText=f.read().replace('NUMBERNEURONS', str(neurons))
                
            with open('neuronfile.py', "w") as f:
                f.write(newText)
                        
            heat=subprocess.run(
                [sys.executable, 'neuronfile.py'],capture_output=True, text=True, shell=True
            )
            print(heat.stdout)
            
            print('LoopActivationfunction: '+str(acticounter)+' , '+str(actifunc))
            print('LoopLayer: '+str(layer))
            print('LoopNeuron: '+str(neurons))
            print()
            print('----------------------------------------------------------------------------------------------------------------------------------------------')
            print()
            #end of neuron loop
            neurons += 1
        
        #end of layer loop
        layer += 1
    
    #end of activation loop    
    acticounter += 1
    

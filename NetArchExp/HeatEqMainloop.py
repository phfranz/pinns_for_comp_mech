import sys, os
import subprocess

# activationfunction loop
actifunclist = ['tanh', 'sigmoid', 'relu', 'swish']
for actifunc in actifunclist:
 
    # layer loop
    for layer in range (4,5,1):
   
        # neuron loop 
        for neurons in range (2,3,1):
            
            os.system('copy BaseheatEq.py WorkheatEq.py' )
            
            with open('BaseheatEq.py') as f:
                newText=f.read().replace('ACTIVATIONFUNCTION',str(actifunc))\
                                .replace('NUMBERLAYERS',str(layer))\
                                .replace('NUMBERNEURONS',str(neurons))
                
            with open('WorkheatEq.py', "w") as f:
                    f.write(newText)
       
            heat=subprocess.run(
                [sys.executable,'WorkheatEq.py' ],capture_output=True, text=True, shell=True
            )
            print(heat.stdout)
            
            print('LoopActivationfunction: '+str(actifunc))
            print('LoopLayer: '+str(layer))
            print('LoopNeuron: '+str(neurons))
            print()
            print('----------------------------------------------------------------------------------------------------------------------------------------------')
            print()

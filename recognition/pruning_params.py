import mxnet as mx  
import pdb 
import numpy as np

sym, arg_params, aux_params = mx.model.load_checkpoint('models/y2-arcface-retina/model', 1) 
params_sort = sorted(arg_params.keys())

#i=0
#for param_name in params_sort:
#    print('{}, {}, {}'.format(i,param_name,arg_params[param_name].shape)) 
#    i=i+1

save_index_res_2 = []
save_index_res_3 = []
save_index_res_4 = []
save_index_res_5 = []
save_index_dconv = []

cfg = 0 # 2 blocks and each block has 31 parameters, it ranges as 4 4 3 4 4 3 4 3 2
i=0
cfg_index_nobn = [6,17,32,50,62]
cfg_nobn = 0
cfg_index_care = [35,53,62]
cfg_care = 0

for param_name in params_sort:
    if 'res_2' in param_name:
        if len(arg_params[param_name].shape)==4 and 'conv_proj' not in param_name and i < 58:
            out_c = arg_params[param_name].shape[0]
            prune_n = int(out_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_2.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]            

        elif len(arg_params[param_name].shape)==4 and i == cfg_index_care[cfg_care]:
            #shape[0] prune half
            out_c = arg_params[param_name].shape[0]
            in_c = arg_params[param_name].shape[1]
            prune_n = int(out_c * 0.5)
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())  
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_2.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]  

            #shape[1] prune half
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_care = cfg_care + 1
                      
        elif len(arg_params[param_name].shape)==4 and i == cfg_index_nobn[cfg_nobn]:
            in_c = arg_params[param_name].shape[1]
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_nobn = cfg_nobn + 1                            
        i=i+1            
            
i=0
cfg_index_nobn = [6,17,28,39,50,61,72,83,98,116,134,152,170,188,206,224,248]
cfg_nobn = 0
cfg_index_care = [101,119,137,155,173,191,209,227,248]
cfg_care = 0
for param_name in params_sort:                   
    if 'res_3' in param_name:
        if len(arg_params[param_name].shape)==4 and 'conv_proj' not in param_name and i < 232:
            out_c = arg_params[param_name].shape[0]
            prune_n = int(out_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_3.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]     
            
        elif len(arg_params[param_name].shape)==4 and i == cfg_index_care[cfg_care]:
            out_c = arg_params[param_name].shape[0]
            in_c = arg_params[param_name].shape[1]
            prune_n = int(out_c * 0.5)
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())  
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_3.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]     
                        
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_care = cfg_care + 1
                      
        elif len(arg_params[param_name].shape)==4 and i == cfg_index_nobn[cfg_nobn]:
            in_c = arg_params[param_name].shape[1]
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_nobn = cfg_nobn + 1 
        i=i+1              

i=0
cfg_index_nobn = [6,17,28,39,50,61,72,83,94,105,116,127,138,149,160,171,186,204,222,240,258,276,294,312,330,348,366,384,402,420,438,456,496]
cfg_nobn = 0
cfg_index_care = [189,207,225,243,261,279,297,315,333,351,369,387,405,423,441,459,496]
cfg_care = 0
for param_name in params_sort:                  
    if 'res_4' in param_name:
        if len(arg_params[param_name].shape)==4 and 'conv_proj' not in param_name and i < 464:
            out_c = arg_params[param_name].shape[0]
            prune_n = int(out_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_4.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]     
            
        elif len(arg_params[param_name].shape)==4 and i == cfg_index_care[cfg_care]:
            out_c = arg_params[param_name].shape[0]
            in_c = arg_params[param_name].shape[1]
            prune_n = int(out_c * 0.5)
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())  
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_4.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]     
                        
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_care = cfg_care + 1
                      
        elif len(arg_params[param_name].shape)==4 and i == cfg_index_nobn[cfg_nobn]:
            in_c = arg_params[param_name].shape[1]
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_nobn = cfg_nobn + 1 
        i=i+1

i=0
cfg_index_nobn = [6,17,28,39,50,61,72,83,98,116,134,152,170,188,206,224,248]
cfg_nobn = 0
cfg_index_care = [101,119,137,155,173,191,209,227,248]
cfg_care = 0      
for param_name in params_sort:          
    if 'res_5' in param_name:
        if len(arg_params[param_name].shape)==4 and 'conv_proj' not in param_name and i < 232:
            out_c = arg_params[param_name].shape[0]
            prune_n = int(out_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_5.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]     
            
        elif len(arg_params[param_name].shape)==4 and i == cfg_index_care[cfg_care]:
            out_c = arg_params[param_name].shape[0]
            in_c = arg_params[param_name].shape[1]
            prune_n = int(out_c * 0.5)
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())  
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_res_5.append(mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]     
                        
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_care = cfg_care + 1
                      
        elif len(arg_params[param_name].shape)==4 and i == cfg_index_nobn[cfg_nobn]:
            in_c = arg_params[param_name].shape[1]
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]
            cfg_nobn = cfg_nobn + 1 
        i=i+1  

for param_name in params_sort:                  
    if 'dconv' in param_name:
        if len(arg_params[param_name].shape)==4 and 'conv_proj' not in param_name:
            out_c = arg_params[param_name].shape[0]
            prune_n = int(out_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())
            save_dict = np.sum(params, axis=(1,2,3))
            arg_max = np.argsort(save_dict)
            index_1 = arg_max[::-1][:prune_n]
            mask = np.zeros(out_c)
            mask[index_1.tolist()]=1
            save_index_dconv.append(mask)    
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist(),:,:,:]  
        elif len(arg_params[param_name].shape)==4:    
            in_c = arg_params[param_name].shape[1]
            prune_n_in = int(in_c * 0.5)
            params = abs(arg_params[param_name].copy().asnumpy())   
            save_dict_in = np.sum(params, axis=(0,2,3))
            arg_max_in = np.argsort(save_dict_in)
            index_in = arg_max[::-1][:prune_n_in]
            mask_in = np.zeros(in_c)
            mask_in[index_in.tolist()]=1
            idx = np.squeeze(np.argwhere(np.asarray(mask_in)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[:,idx.tolist(),:,:]

cfg = 0 # 2 blocks and each block has 31 parameters, it ranges as 4 4 3 4 4 3 4 3 2
i=0    # so 
j = 0

for param_name in params_sort:  
    if 'res_2' in param_name and i < 58:
        if len(arg_params[param_name].shape)==1:
            mask = save_index_res_2[cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist()]
            j = j + 1
            if 'conv_proj' in param_name and j == 3:
                j = 0
                cfg = cfg + 1
            elif j == 4:
                j = 0 
                cfg = cfg + 1             
        i = i + 1

cfg = 0 # 8 blocks and each block has 31 parameters, it ranges as 4 4 3 4 4 3 4 3 2
i=0    # so 
j = 0
for param_name in params_sort:  
    if 'res_3' in param_name and i < 232:
        if len(arg_params[param_name].shape)==1:
            mask = save_index_res_3[cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist()]
            j = j + 1
            if 'conv_proj' in param_name and j == 3:
                j = 0
                cfg = cfg + 1
            elif j == 4:
                j = 0 
                cfg = cfg + 1   
        i = i + 1        

cfg = 0 # 16 blocks and each block has 31 parameters, it ranges as 4 4 3 4 4 3 4 3 2
i=0    # so 
j = 0
for param_name in params_sort:  
    if 'res_4' in param_name and i < 464:
        if len(arg_params[param_name].shape)==1:
            mask = save_index_res_4[cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist()]
            j = j + 1
            if 'conv_proj' in param_name and j == 3:
                j = 0
                cfg = cfg + 1
            elif j == 4:
                j = 0 
                cfg = cfg + 1   
        i = i + 1 

cfg = 0 # 8 blocks and each block has 31 parameters, it ranges as 4 4 3 4 4 3 4 3 2
i=0    # so 
j = 0
for param_name in params_sort:  
    if 'res_5' in param_name and i < 232:
        if len(arg_params[param_name].shape)==1:
            mask = save_index_res_5[cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist()]
            j = j + 1
            if 'conv_proj' in param_name and j == 3:
                j = 0
                cfg = cfg + 1
            elif j == 4:
                j = 0 
                cfg = cfg + 1   
        i = i + 1 

cfg = 0 # dconv
i=0    # so 
j = 0
for param_name in params_sort:  
    if 'dconv' in param_name and 'conv_proj' not in param_name:
        if len(arg_params[param_name].shape)==1:
            mask = save_index_dconv[cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask)))
            param = arg_params[param_name].copy()
            arg_params[param_name] = None
            arg_params[param_name] = param[idx.tolist()]
            j = j + 1
            if j == 3:
                j = 0 
                cfg = cfg + 1    
        i = i + 1 

#i=0
#for param_name in params_sort:
#    print('{}, {}, {}'.format(i,param_name,arg_params[param_name].shape)) 
#    i=i+1



            
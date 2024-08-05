from __future__ import print_function

import torch.utils.data

# from training import *


def MFCN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50):
    #out = MFCN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):   # G = Gs,    Z_opt = Zs,    noise_amp = NosieAmp
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2     # 5
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device) 
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)   
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)        
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)  
                # I_prev = m(I_prev)
                # I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                # I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)    # (I_prev,4/3,opt)
                I_prev = m(I_prev)  

            if n < gen_start_scale:
                z_curr = Z_opt   
                
            z_in = noise_amp*(z_curr)+I_prev    
            I_curr,_ = G(z_in.detach(),I_prev)

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
            images_cur.append(I_curr)
        n+=1
    return I_curr.detach()



def MFCN_generate_sr(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50):   
    #out = MFCN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):   # G = Gs,    Z_opt = Zs,    noise_amp = NosieAmp
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2     # 5
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
            z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])   
            z_curr = m(z_curr)   

            if images_prev == []:
                I_prev = m(in_s)   
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)    # (I_prev,4/3,opt)
                I_prev = m(I_prev)   

            z_in = noise_amp*(z_curr)+I_prev     
            I_curr,_ = G(z_in.detach(),I_prev)    

            if n == len(reals)-1:
                dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass

            images_cur.append(I_curr)
        n+=1
    return I_curr.detach()
import numpy as np
import sys
import math
import scipy.linalg as sl
import matplotlib.pyplot as plt            
### TODOs ###
# Test std classical correlators reproduce frequencies.

DECONV_ITER_CUTOFF=10


def readInput(inputfilename="input.inp"):
    global mass,dt,dim,nsteps, gamma,T, pot_type # qtb cutoff, qtb seg length, etc
    infile=open(inputfilename,"r")

    # default values
    T=300
    gamma=0.1
    mass=1.
    dim=1
    pot_type="ho"
    pot_params=[]
    qtb_update=1000
    ensemble="nvt"
    use_qtb=1
    omega_cutoff=2.5 # the cutoff frequency should be  few times the max frequency appearing in the system
    stridelen=1
    use_theta_tilde=1
    integrator="baoab"
    for line in infile:
        if line[0] !="#":
          line = line.strip().split('=')
          if line[0].strip() == 'mass':
            mass = float(line[1].strip())
          elif line[0].strip() == 'ic_sampling':
            ic_sampling = str(line[1].strip())            
          elif line[0].strip() == 'qtb':
            use_qtb = int(line[1].strip())
          elif line[0].strip() == 'integrator':
            integrator = str(line[1].strip())  
          elif line[0].strip() == 'use_theta_tilde':
            use_theta_tilde = int(line[1].strip())
          elif line[0].strip() == 'ensemble':
              ensemble=str(line[1].strip())
          # if line[0].strip() == 'omega':
          #     omega = float(line[1].strip())              
          elif line[0].strip() == 'dt':
              dt = float(line[1].strip())
          elif line[0].strip() == 'dimension':
              dim = int(line[1].strip())
              x = np.zeros(dim)
              #x = np.array([0.1610980686750549, 0.24029959298851195])
              v = np.zeros(dim)

          elif line[0].strip() == 'nsteps':
              nsteps=int(line[1].strip())
              
          elif line[0].strip() == 'stridelen':
              stridelen=int(line[1].strip())
              
          elif line[0].strip() == 'init_conditions':
              ic_toks=[float(tok) for tok in line[1].split()]

              if dim==1:
                  x=np.array([ic_toks[0]])
                  v=np.array([ic_toks[1]])
              elif dim==2:
                  x=np.array([ic_toks[0],ic_toks[2]])
                  v=np.array([ic_toks[1],ic_toks[3]])              
              # init conditions specified as an array [[x1,x2,...],[v1,v2,...]]
              pass
          elif line[0].strip() == 'gamma':
              gamma=float(line[1].strip())
          elif line[0].strip() == 'temperature':
              T=float(line[1].strip())
          elif line[0].strip() == 'potential':
              print("pot found")
              pot_type=str(line[1].split()[0])              
              pot_params=[float(param) for param in line[1].split()[1:]]
              # if pot_type=="cho":
              #     x = np.array([0.1610980686750549, 0.0])
#              print(pot_params)

          elif line[0].strip() == 'qtb_update':
              qtb_update=int(line[1].strip())
          elif line[0].strip() == 'omega_cutoff':
              omega_cutoff=float(line[1].strip())


    try:
        if ic_sampling=="mb":
            if pot_type == "ho_toy":
                omega=np.sqrt(pot_params[0]/mass)
            print("sampling from canonical distro for x and v. omega = {} Only implemented for HO.".format(omega))
            kbT=kb*T
            v = np.random.normal(scale=kbT*mass,size=dim)#array([np.sqrt(2*kbT/mass)]*dim)
            x = np.random.normal(scale=kbT/(mass*omega**2),size=dim)#array([np.sqrt(2*kbT/mass)]*dim)
    except:
        print("Warning:vel_sampling not defined")

          

    print("pot_params={} mass= {}".format(pot_params, mass))

    return stridelen,x,v, T, gamma, mass, dt, nsteps, dim, pot_type, qtb_update, omega_cutoff,pot_params, ensemble, use_qtb,use_theta_tilde,integrator




def Symmetrize(sig):
    nomega=len(sig)
    sig_symm=np.zeros(2*nomega-1)
    sig_symm[nomega-1] = sig[0]
    for omega_idx in range(1,nomega):
        
      sig_symm[nomega+omega_idx-1] = sig[omega_idx]

      sig_symm[nomega-omega_idx-1] = sig[omega_idx]
              
    return sig_symm



def DeconvoluteSpectrum(omega_arr_symm, theta_symm, theta):

    denom=0.

    deconv_iter=0
    theta_tilde_outfile=open("theta_tilde.out","w")
    nomegadc=nomega
    fn = np.zeros(2*nomegadc-1)
    fnp1 = np.zeros(2*nomegadc-1)
    h_isra = np.zeros(2*nomegadc-1)
    theta_conv_kernel=np.zeros((2*nomegadc-1,2*nomegadc-1))
    D=np.zeros((2*nomegadc-1,2*nomegadc-1))
    theta_conv_kernel_norm_fac = np.zeros(2*nomegadc-1)
    vv_deconv = np.zeros(2*nomega-1)

        
    for omega_0_idx in range(2*nomegadc-1):
        omega_0= omega_arr_symm[omega_0_idx]
        omega_0sq=omega_0*omega_0
        fn[omega_0_idx]=theta_symm[omega_0_idx] # f0=theta

        for omega_idx in range(2*nomegadc-1):
            omega= omega_arr_symm[omega_idx]
            omegasq=omega*omega
          
#            if(np.abs(omega_0) > 0.0 or np.abs(omega) > 0.0):
            if(omega_0_idx!=(nomegadc-1) or omega_idx!=(nomegadc-1)):
#                print("omega_0={} omega={}".format(omega_0, omega))
                theta_conv_kernel[omega_0_idx][omega_idx] = 1.*(gamma*omegasq)/(np.pi*((omegasq-omega_0sq)*(omegasq-omega_0sq) + gamma*gamma*omegasq))
            else:
 #               print("found zero. gamma={}".format(gamma))
  #              print(omega_0_idx)
                theta_conv_kernel[omega_0_idx][omega_idx] = 1.0/(np.pi*gamma)
        #        print("theta_conv_kernel[omega_0_idx][omega_idx]= {}".format(theta_conv_kernel[omega_0_idx][omega_idx]))

        
            theta_conv_kernel_norm_fac[omega_0_idx] += theta_conv_kernel[omega_0_idx][omega_idx]
            
        #print(theta_conv_kernel_norm_fac[1499])

      
      # for omega_idx in range(2*nomegadc):
      #   omega= omega_arr_symm[omega_idx]        
    theta_conv_kernel=np.einsum("ij,i->ij",theta_conv_kernel,1./(theta_conv_kernel_norm_fac*domega))

    # for omega_0_idx in range(2*nomegadc-1):
    #     for omega_idx in range(2*nomegadc-1):


    #         theta_conv_kernel[omega_0_idx][omega_idx] = theta_conv_kernel[omega_0_idx][omega_idx]/theta_conv_kernel_norm_fac[omega_0_idx]/domega



    h_isra = domega*np.einsum("ij,i->j",theta_conv_kernel,theta_symm)

    # for omega_idx in range(2*nomegadc-1):
    #     for x_idx  in range(2*nomegadc-1):
            
            
    #             try:
    #                 h_isra[omega_idx] += domega*theta_conv_kernel[x_idx][omega_idx]*theta_symm[x_idx]
    #             except:
    #                 print(f"kern={theta_conv_kernel[x_idx][omega_idx]} theta_symm={theta_symm[x_idx]} x_idx={x_idx}")
    #                 exit()

      # set up D(omega,x) = int dy C(y,omega)*C(y,x)

    D=np.einsum("ij,ik->ik",theta_conv_kernel,theta_conv_kernel)
    D*=domega
    print("Set up double convolution kernel")
    # for omega_idx in range(2*nomegadc-1):
    #     for x_idx  in range(2*nomegadc-1):
    #         for y_idx  in range(2*nomegadc-1):
    #             D[omega_idx][x_idx] += domega*theta_conv_kernel[y_idx][omega_idx]*theta_conv_kernel[y_idx][x_idx]

    corr_factor=np.zeros(2*nomegadc-1)

    while(deconv_iter < DECONV_ITER_CUTOFF):
        rel_sq_diff = 0.0
        rel_sq_diff_denom=0.0
        print("iteration {}\n".format(deconv_iter))
#         for omega_idx in range(2*nomegadc-1): #// loop over omega to set f_n+1(omega)=f_n(omega)*h(omega)/int dx D(omega,x) f_n(x)
#             denom=0.
#             for x_idx  in range(2*nomegadc-1):
#                 denom += domega*fn[x_idx]*D[omega_idx][x_idx]

#             fnp1[omega_idx]=fn[omega_idx]*h_isra[omega_idx]/denom
# #        denom = domega*np.einsum("i,ji->",fn,D)
# #fnp1=np.einsum("i,i->i",fn,h_isra/denom)
#         rel_sq_diff_denom=np.einsum("i,i->",fn,fn)
#         rel_sq_diff=np.einsum("i,i->",fnp1-fn,fnp1-fn)/rel_sq_diff_denom
#         print(f"iter = {deconv_iter} rsqd={rel_sq_diff}") 
#         fn=fnp1
        
        for omega_idx in range(2*nomegadc-1): #// loop over omega to set f_n+1(omega)=f_n(omega)*h(omega)/int dx D(omega,x) f_n(x)
            denom=0.

            omega= omega_arr_symm[omega_idx]
            #calculate denominator for each omega
            for x_idx  in range(2*nomegadc-1):
                denom += domega*fn[x_idx]*D[omega_idx][x_idx]
        
            fnp1[omega_idx]=fn[omega_idx]*h_isra[omega_idx]/denom
            rel_sq_diff_denom += fn[omega_idx]*fn[omega_idx]
      

        for omega_idx in range(2*nomegadc-1): 
          rel_sq_diff+= (fnp1[omega_idx] - fn[omega_idx])*(fnp1[omega_idx] - fn[omega_idx])/rel_sq_diff_denom
        print(f"iter = {deconv_iter} rsqd={rel_sq_diff}") 
        
        
        for omega_idx in range(2*nomegadc-1): 
            omega= omega_arr_symm[omega_idx]

            fn[omega_idx] = fnp1[omega_idx]
        
        deconv_iter+=1

      
      # reconvolve theta_tilde to obtain theta

    theta_rec = np.zeros(2*nomegadc-1)
#    theta_rec = domega*np.einsum("ij,j->i",theta_conv_kernel,fn)

    for omega_idx in range(2*nomegadc-1):
    
        if theta_symm[omega_idx] >0.0:
            corr_factor[omega_idx]=(fn[omega_idx]/theta_symm[omega_idx])-1.
        else:
            corr_factor[omega_idx]=0.0
    # try:
    #     corr_factor=(fn/theta_symm)-1.
    #     print(f"corr_factor={corr_factor}")
    # except:
    #     print(f"theta_symm={theta_symm}")

    # for omega_idx in range(2*nomegadc-1):
    #     omega= omega_arr_symm[omega_idx]

    #     if theta_symm[omega_idx] >0.0:
    #         corr_factor[omega_idx]=(fn[omega_idx]/theta_symm[omega_idx])-1.
    #     else:
    #         corr_factor[omega_idx]=0.0

      # reconvolve theta_tilde to obtain theta      
    theta_rec = np.zeros(2*nomegadc-1)
    
    theta_rec=domega*np.einsum("ij,i->j",theta_conv_kernel,fn)
    # for omega_idx in range(2*nomegadc-1):
    #     omega= omega_arr_symm[omega_idx]

    #     if theta_symm[omega_idx] >0.0:
    #         corr_factor[omega_idx]=(fn[omega_idx]/theta_symm[omega_idx])-1.
    #     else:
    #         corr_factor[omega_idx]=0.0
    #     for omegap_idx in range(2*nomegadc-1):
    #       theta_rec[omega_idx] += domega*theta_conv_kernel[omega_idx][omegap_idx]*fn[omegap_idx]



            
    # for omegap_idx in range(2*nomegadc-1):
    #       theta_rec[omega_idx] += domega*theta_conv_kernel[omega_idx][omegap_idx]*fn[omegap_idx]
    for omega_idx in range(nomegadc-1, 2*nomegadc-1):
        vv_deconv[omega_idx-nomegadc] = fn[omega_idx]
        omega= omega_arr_symm[omega_idx]
        theta_tilde_outfile.write("{}\t{}\t{}\t{}\t{}\n".format(omega, theta_symm[omega_idx], vv_deconv[omega_idx-nomegadc],theta_rec[omega_idx],corr_factor[omega_idx]))


    return vv_deconv
      



def DeconvoluteCorrelator(gamma,omega_arr_symm, sig_symm, signal_name="vv"):
    
    denom=0.
    nomega=nomega_deconv
    deconv_iter=0
    sig_deconv_outfile=open("{}_deconv.out".format(signal_name),"w")
    fn = np.zeros(2*nomega-1)
    fnp1 = np.zeros(2*nomega-1)
    h_isra = np.zeros(2*nomega-1)
    conv_kernel=np.zeros((2*nomega-1,2*nomega-1))
    D=np.zeros((2*nomega-1,2*nomega-1))
    conv_kernel_norm_fac = np.zeros(2*nomega-1)



        
    for omega_0_idx in range(2*nomega-1):
        omega_0= omega_arr_symm[omega_0_idx]
        omega_0sq=omega_0*omega_0
        fn[omega_0_idx]=sig_symm[omega_0_idx] # f0=sig

        for omega_idx in range(2*nomega-1):
            omega= omega_arr_symm[omega_idx]
            omegasq=omega*omega
          
#            if(np.abs(omega_0) > 0.0 or np.abs(omega) > 0.0):
            if(omega_0_idx!=(nomega-1) or omega_idx!=(nomega-1)):
                #print("omega_0={} omega={}".format(omega_0, omega))
                try:
                    conv_kernel[omega_0_idx][omega_idx] = 1.*(gamma*omegasq)/(np.pi*((omegasq-omega_0sq)*(omegasq-omega_0sq) + gamma*gamma*omegasq))
                except:
                    print("omega_0={} omega={}".format(omega_0, omega))
            else:
 #               print("found zero. gamma={}".format(gamma))
  #              print(omega_0_idx)
                conv_kernel[omega_0_idx][omega_idx] = 1.0/(np.pi*gamma)
        #        print("theta_conv_kernel[omega_0_idx][omega_idx]= {}".format(theta_conv_kernel[omega_0_idx][omega_idx]))


            conv_kernel_norm_fac[omega_0_idx] += conv_kernel[omega_0_idx][omega_idx]
        #print(theta_conv_kernel_norm_fac[1499])

      
      # for omega_idx in range(2*nomega):
      #   omega= omega_arr_symm[omega_idx]        
    conv_kernel=np.einsum("ij,i->ij",conv_kernel,1./(conv_kernel_norm_fac*domega))

    # for omega_0_idx in range(2*nomega-1):
    #     for omega_idx in range(2*nomega-1):


    #         conv_kernel[omega_0_idx][omega_idx] = conv_kernel[omega_0_idx][omega_idx]/conv_kernel_norm_fac[omega_0_idx]/domega




    for omega_idx in range(2*nomega-1):
        for x_idx  in range(2*nomega-1):
                h_isra[omega_idx] += domega*conv_kernel[x_idx][omega_idx]*sig_symm[x_idx] 

      # set up D(omega,x) = int dy C(y,omega)*C(y,x)

    D=np.einsum("ij,ik->ik",conv_kernel,conv_kernel)
    D*=domega
    print("Set up double convolution kernel")
    # for omega_idx in range(2*nomega-1):
    #     for x_idx  in range(2*nomega-1):
    #         for y_idx  in range(2*nomega-1):
    #             D[omega_idx][x_idx] += domega*theta_conv_kernel[y_idx][omega_idx]*theta_conv_kernel[y_idx][x_idx]

    corr_factor=np.zeros(2*nomega-1)

    while(deconv_iter < DECONV_ITER_CUTOFF):
        rel_sq_diff = 0.0
        rel_sq_diff_denom=0.0
        print("iteration {}\n".format(deconv_iter))
        for omega_idx in range(2*nomega-1): #// loop over omega to set f_n+1(omega)=f_n(omega)*h(omega)/int dx D(omega,x) f_n(x)
            denom=0.

            omega= omega_arr_symm[omega_idx]
            #calculate denominator for each omega
            for x_idx  in range(2*nomega-1):
                denom += domega*fn[x_idx]*D[omega_idx][x_idx]
        
            fnp1[omega_idx]=fn[omega_idx]*h_isra[omega_idx]/denom
            rel_sq_diff_denom += fn[omega_idx]*fn[omega_idx]
            

        for omega_idx in range(2*nomega-1): 
          rel_sq_diff+= (fnp1[omega_idx] - fn[omega_idx])*(fnp1[omega_idx] - fn[omega_idx])/rel_sq_diff_denom
        
        
        
        for omega_idx in range(2*nomega-1): 
            omega= omega_arr_symm[omega_idx]

            fn[omega_idx] = fnp1[omega_idx]
        
        deconv_iter+=1

      
      # reconvolve sig_deconv to obtain sig      
    sig_rec = np.zeros(2*nomega-1)
    sig_deconv = np.zeros(2*nomega-1)
    for omega_idx in range(2*nomega-1):
        omega= omega_arr_symm[omega_idx]

        corr_factor[omega_idx]=(fn[omega_idx]/sig_symm[omega_idx])-1.
        for omegap_idx in range(2*nomega-1):
          sig_rec[omega_idx] += domega*conv_kernel[omegap_idx][omega_idx]*fn[omegap_idx]

    for omega_idx in range(nomega-1, 2*nomega-1):
        sig_deconv[omega_idx-nomega] = fn[omega_idx]
        omega= omega_arr_symm[omega_idx]
        sig_deconv_outfile.write("{}\t{}\t{}\t{}\t{}\n".format(omega, sig_symm[omega_idx], sig_deconv[omega_idx-nomega],sig_rec[omega_idx],corr_factor[omega_idx]))


    return sig_deconv[nomega-1:] # return only 0 and +ve freqs
    


BOLTZMANN_CONSTANT=3.1668114e-06

#gamma=1.46e-3


inputfilename="input.inp"
# get input params from input.inp
stridelen,x,v, T, gamma, mass, dt, nsteps,dim,pot_type,qtb_update,omega_cutoff,pot_params,ensemble,use_qtb,use_theta_tilde,integrator=readInput(inputfilename) # x and v are np arrays of shape (dim)



ndim=dim
nsteps_raw=int(sys.argv[1])
nsteps=nsteps_raw
print("integrator={}".format(integrator))
deconv=1
nsteps_equilib=0
nomega=nsteps

nomega_max=2*np.pi/dt
mass=1837.15417302


domega=(2*np.pi/dt)/(nsteps)
qtb_update=1000


#nomega_deconv=int(omega_cutoff/domega)
nomega_deconv=nomega
#print("nomega_deconv={}".format(nomega_deconv))

#print("domega (signal) = {}".format(domega))
print("dt=  {}".format(dt))
print("gamma=  {}".format(gamma))
#print("int(omega_cutoff/domega)={}".format(int(omega_cutoff/domega)))
# set up kubo array

kbT=BOLTZMANN_CONSTANT*T
beta=1./kbT
kubo_arr=np.zeros(nsteps)

kubo_outfile = open("kubo_fac.out","w")
print("nsteps=  {}".format(nsteps))

if (nsteps%2 !=0):
    print("need even number of steps")


#np.savetxt(kubo_outfile,kubo_arr)
# exit()    

# print(kubo_arr)
# exit()

# for el in kubo_arr:
#     print("{}\n".format(el))

vel_filename="velocities.dat"

vel_infile=open(vel_filename,"r")


# read in position data


# read in velocity data        

vel_arr=np.zeros((ndim,nsteps))

for i in range(nsteps):
    line=vel_infile.readline()
    tok=line.split()
    
    if i > nsteps_equilib:    

        if ndim==1:
            vel1=float(tok[1])
            vel_arr[0,i]=vel1
            
        elif ndim==2:
            vel1=float(tok[1])
            vel2=float(tok[2])
            vel_arr[0,i]=vel1
            vel_arr[1,i]=vel2        

# testing convolution of a Lorentzian            

# read in force data        
        
    # symmetrize omega array and theta

if(deconv):
    omega_arr_symm=np.zeros(2*nomega_deconv-1)

    # for multiplication of vv FFT by theta, require it to be symmetrized with entries 0,dom,2*dom,...,nom_deconv,-nom_deconv,-(nom_deconv-dom),...,,-dom
    for omega_idx in range(1,nomega_deconv):
        
        omega_arr_symm[nomega_deconv+omega_idx-1] =  omega_idx*domega
        omega_arr_symm[nomega_deconv-omega_idx-1] = -omega_idx*domega

        


    # compute theta and obtain theta_tilde

    


# construct Kubo-transformed correlators. 

vvm_kubo_t0=np.zeros((ndim,ndim))
ppm_kubo_t0=np.zeros((ndim,ndim))
posposm_kubo_t0=np.zeros((ndim,ndim))
forceforcem_kubo_t0=np.zeros((ndim,ndim))

vvm_deconv=np.zeros(nomega_deconv)
posposm_deconv=np.zeros(nomega_deconv)
forceforcem_deconv=np.zeros(nomega_deconv)

vv_std_outfile=open("vv.out","w")


omega0=0.010039615439445369


for d1 in range(ndim):
    for d2 in range(ndim):

        

        
        vel1_fft=np.fft.fft(vel_arr[d1,:])
        vel2_fft=np.fft.fft(vel_arr[d2,:])

        # at this point, we can construct the vv correlator and deconvolute before Kubo transforming.
        vvm_ft_conv=np.real(vel1_fft[:]*np.conj(vel2_fft[:]))
        vvm_ift=np.fft.ifft(vvm_ft_conv)
        print("\n2*KE (Kubo)={} \t theta({})={}\n".format(mass*vvm_ift[0]/nomega, omega0,.5*omega0/np.tanh(.5*omega0/kbT)))
        # for om in range(1,nomega):
        #     omega=om*domega
        #     theta=.5*omega/np.tanh(.5*omega/kbT)
        #     vv_std_outfile.write("{} {} {}\n".format(omega,mass*vvm_ift[0]/nomega,theta/2.))
        # write out vvm. compare with theta/2
        if (deconv):
            
            # freqs for qtbseg are domega_theta. 
            # divide by theta_tilde
            #vvm_ft[0]/=1
            vvm_deconv[0] = vvm_ft_conv[0]
            for w in range(1,int(nomega_deconv/2)):
            # w = 0,1,..OM,OM-1,...,-1
                # truncate vv,xx,ff
                vvm_deconv[w]=vvm_ft_conv[w]
                vvm_deconv[-w]=np.conj(vvm_ft_conv[w])


            vvm_ft_symm=Symmetrize(vvm_ft_conv)# symmetrize omega array and signal
            # np.savetxt("vvm_ft_conv.out", vvm_ft_conv)
            # np.savetxt("vvm_ft_symm.out", vvm_ft_symm)
            # print(f"vvm_ft_conv={vvm_ft_conv}")
            # print(f"vvm_ft_symm={vvm_ft_symm}")
            # exit()
            vvm_ft=DeconvoluteSpectrum(omega_arr_symm,vvm_ft_symm, vvm_ft_conv)
            vvm_ft=vvm_ft[0:nomega]
            #vvm_ft=DeconvoluteCorrelator(gamma,omega_arr_symm,vvm_ft_symm,signal_name="vv")
dt/=41.34137
freq = np.fft.fftfreq(len(vvm_ft_conv), dt)
freq = freq*33400             
plt.plot(freq,np.real(vvm_ft_conv))
plt.plot(freq,np.real(vvm_ft))
plt.show()

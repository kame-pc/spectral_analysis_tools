import numpy as np
import scipy as sp
import scipy.special as spec
from scipy.stats.mstats import gmean
from matplotlib.colors import to_rgb, LinearSegmentedColormap, LogNorm
from functools import partial
from multiprocessing import Pool, cpu_count
from numpy.random import rand, randn
import logging

#----------------Miscellaneous--------------------

logger = logging.getLogger(__name__)

def _configure_logger(verbose: bool):
    """Configure and return module logger."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def gradient_dict(col_vec, alpha_vec=[], name_cm=None):
    """Create a dictionary interpolating all colors in 'col_vec'.

    Input:
        col_vec: list with colors (tuples or strings).
        name_cm: name of the colormap.
    """
    red = []
    green = []
    blue = []
    alpha = []
    x = np.linspace(0., 1., len(col_vec))
    if alpha_vec == []:
        alpha_vec = [1 for col in col_vec]
    else:
        assert len(alpha_vec) == len(col_vec)
    for (i, col) in enumerate(col_vec):
        col = to_rgb(col)
        red.append(tuple([x[i], col[0], col[0]]))
        green.append(tuple([x[i], col[1], col[1]]))
        blue.append(tuple([x[i], col[2], col[2]]))
        alpha.append(tuple([x[i], alpha_vec[i], alpha_vec[i]]))
    cdict = {'red': red, 'green': green, 'blue': blue, 'alpha': alpha}
    return LinearSegmentedColormap(name_cm, cdict)

def confidence_ivals_log(x, dx, Px, err_tol, m):
    """Confidence interval for the dist. (x, Px) with given error tolerance."""
    l, r = 0, len(Px)
    
    if np.any(np.diff(x) < 0):
        x_ord = np.argsort(x)
        x, Px = x[x_ord], Px[x_ord]
    
#     dx = x[1:]-x[:-1]
    Px_cumsum_l = np.cumsum(dx*np.nan_to_num(Px))/sum(np.nan_to_num(Px)*dx)
    Px_cumsum_r = np.cumsum(dx[::-1]*np.nan_to_num(Px)[::-1])[::-1]/sum(np.nan_to_num(Px)*dx)
#     Px_cumsum_l = np.cumsum(dx*np.sqrt(np.nan_to_num(Px[1:])*np.nan_to_num(Px[:-1])))\
#     /sum(np.sqrt(np.nan_to_num(Px[1:])*np.nan_to_num(Px[:-1]))*dx)
#     Px_cumsum_r = np.cumsum(dx[::-1]*np.sqrt(np.nan_to_num(Px[1:])*np.nan_to_num(Px[:-1]))[::-1])[::-1]\
#     /sum(np.sqrt(np.nan_to_num(Px[1:])*np.nan_to_num(Px[:-1]))*dx)
    
    il = np.argmax(Px_cumsum_l > err_tol)
    ir = np.argmin(Px_cumsum_r > err_tol)
    
    left_limit = x[il]
    rght_limit = x[ir]
    if il == 0 and Px_cumsum_l[0] <= err_tol:
        left_limit = x[0]
    if ir <= 0 and Px_cumsum_r[0] <= err_tol:
        rght_limit = x[-1]
    if not left_limit <= m:
#         raise Exception()
        left_limit = x[0]
        rght_limit = x[np.argmin(Px_cumsum_r > err_tol*2)]
    return left_limit, rght_limit  

def confidence_ivals(x, Px, err_tol, m):
    """Confidence interval for the dist. (x, Px) with given error tolerance."""
    l, r = 0, len(Px)
    
    if np.any(np.diff(x) < 0):
        x_ord = np.argsort(x)
        x, Px = x[x_ord], Px[x_ord]
    dx = x[1]-x[0]
    
    Px_cumsum_l = dx*np.cumsum(np.nan_to_num(Px))/sum(np.nan_to_num(Px)*dx)
    Px_cumsum_r = dx*np.cumsum(np.nan_to_num(Px)[::-1])[::-1]/sum(np.nan_to_num(Px)*dx)
    
    il = np.argmax(Px_cumsum_l > err_tol)
    ir = np.argmin(Px_cumsum_r > err_tol)
    
    left_limit = x[il]
    rght_limit = x[ir]
    if il == 0 and Px_cumsum_l[0] <= err_tol:
        left_limit = x[0]
    if ir <= 0 and Px_cumsum_r[0] <= err_tol:
        rght_limit = x[-1]
    if not left_limit <= m :
#         raise Exception()
        left_limit = x[0]
        rght_limit = x[np.argmin(Px_cumsum_r > err_tol*2)]
    return left_limit, rght_limit

def shift_phase_coord(ph, center):
    coord_min = center - np.pi
    return 2*np.pi*np.mod((ph - coord_min)/(2*np.pi), 1) + coord_min

def trim_and_reshape(tr, M_s):
    tr_re = tr[..., :tr.shape[-1] - tr.shape[-1]%M_s]
    return tr_re.reshape((np.prod(tr_re.shape[:-1])*M_s, -1))

#----------------Auto-correlations--------------------

def Df(i_f, Nf):
    """Parameter inside corr. functions, e.g. in P_Lambdaf_D().

    Input:
        i_f : freq. index.
        Nf : number of freqs. (beware Nf = N/2+1 for REAL F.T.).
    """
    if i_f == 0 or i_f == Nf-1:
        return 1./2.
    else:
        return 1.
    
def P_Lambdaf_D(Lambda_f, bLambda_f, M, i_f, Nf):
    """Autocorr.-function dist. for 1 variable, P(Lambda_f | data).

    Input:
        bLambda_f: 1/M \sum_m |alpha_k^{(m)}|**2, sufficient estimator.
            Different for k=0: variance of alpha_0^{(m)}.
        M: number of independent batches.
        Lambda_f: autocorr. function, scalar.
        i_f: freq. index.
        Nf: number of freqs. (beware Nf = N/2+1 for REAL F.T.).
    Source: p. CFCFT.12."""
    N = (Nf-1)*2.
    df = Df(i_f, Nf)
    x = bLambda_f/Lambda_f
    test=x>0
    if type(x)==list: x=np.array(x)
    if type(x)==np.ndarray:
        res_log = -np.inf*np.ones_like(x)
        res_log[test]=(M*df)*np.log(x[test]) + np.log(x[test]) - M*df*(x[test]-1)
    else:
        if ~test: res_log = -np.inf
        else: res_log =  (M*df)*np.log(x) + np.log(x) - M*df*(x-1)
#     res_log =  (M*df)*np.log(x) - np.log(x) - M*df*(x-1)

    return np.exp(res_log)

def PSD(x, dt, Ndist=200, f_lims=None, mode='mle', return_P=False, tol=0.1, verbose=True):
    """Calculate auto_corr. functions with error bars.

    Input:
        x: array of measurements, [block, time].
    Source: [EAF], [PRI]."""
    assert len(x.shape) == 2
    M = len(x) # #blocks
    N = len(x[0]) # #points per block in real space
    if N%2 != 0: N-=1; x=x[:,:-1];
    
    freq = np.fft.rfftfreq(x.shape[1], d=dt)
    f_flt = np.ones(len(freq), dtype= bool) if f_lims is None else (min(f_lims) <= freq) & (freq < max(f_lims))
    
    alpha = np.fft.rfft(x, norm='ortho')
    bLambda = np.mean(np.abs(alpha)**2, axis=0)
    bLambda[0] = np.var(alpha[:, 0])
    
    Nf = len(bLambda)
    Lambda_mle = M/(M+1) * bLambda # estimator
    #Lambda_mle[0] =  M/(M+2) * bLambda[0]
    Lambda_mle[-1] = M/(M+2) * bLambda[-1]
    P_Lambda = np.nan * np.ones((Nf, 2, Ndist)) # [freq, Lambda or P(Lambda), i_Lambda]
    Lambda_MLE = np.nan * np.ones(Nf)
    Lambda_MELE = np.nan * np.ones(Nf)
    Lambda = np.nan * np.ones(Nf)
    Lambda_ci = np.nan * np.ones((Nf,2))
    
    logger = _configure_logger(verbose)
    if verbose:
        logger.info('M = %s', M)
    for i_f in range(Nf):
        if f_flt[i_f]:
            Lambda_uplim = Lambda_mle[i_f] # domain upper limit
            P_Lambdaf_D_ref = P_Lambdaf_D(Lambda_mle[i_f], bLambda[i_f], M, i_f, Nf)
            while P_Lambdaf_D(Lambda_uplim, bLambda[i_f], M, i_f, Nf)/P_Lambdaf_D_ref > 1.e-3:
                Lambda_uplim *= 2.
            Lambda_vec = np.linspace(np.log10(Lambda_mle[i_f])-6, np.log10(Lambda_uplim), Ndist+1)
            Lambda_vec = 10**Lambda_vec
            dL = Lambda_vec[1:]-Lambda_vec[:-1]
            Lambda_vec = np.sqrt(Lambda_vec[1:]*Lambda_vec[:-1])
            
            P_Lambda[i_f, 0] = Lambda_vec*dt
            P_Lambda[i_f, 1] = P_Lambdaf_D(Lambda_vec, bLambda[i_f], M, i_f, Nf)
            
            if mode == 'mele':  Lambda[i_f] = dt*np.mean(Lambda_vec*P_Lambda[i_f, 1]*dL)/np.mean(P_Lambda[i_f, 1]*dL)
            elif mode == 'bar':  Lambda[i_f] = bLambda[i_f]*dt
            else: Lambda[i_f] = Lambda_vec[np.argmax(P_Lambda[i_f, 1, :])]*dt
                
            Lambda_ci[i_f] = confidence_ivals_log(P_Lambda[i_f, 0],dL*dt,P_Lambda[i_f, 1],tol/2,Lambda[i_f])
            if verbose:
                logger.info('Calculating auto corr. %d/%d', int(np.sum(f_flt[:i_f])) + 1, int(np.sum(f_flt)))
    
        
    result = {'freq': freq[f_flt], 'Lambda': Lambda[f_flt], 'Lambda_ci': Lambda_ci[f_flt]}
    
    if return_P:
        result.update({'P_Lambda': P_Lambda[f_flt,1], 'Lambda_vec': P_Lambda[f_flt,0]})
        
    return result



#----------------Auto-correlations clustered--------------------

def PSD_clus(x, dt=1, N_x=100, N_y=100, tol=0.1, f_lims=None,
             return_P=False, mode='mle', verbose=True):
    """Calculate clustered auto-correlation distributions and estimators.

    Input:
        x: array of measurements, [block, time].
    """
    assert len(x.shape) == 2
    M = len(x) # #blocks
    N = len(x[0]) # #points per block
    if N%2 != 0: N -= 1; x = x[:,:-1];
    
    freq = np.fft.rfftfreq(x.shape[1], d=dt)
    f_flt = np.ones(len(freq), dtype= bool) if f_lims is None else (min(f_lims) <= freq) & (freq < max(f_lims))
    f_flt[0] = False # don't include freq=0

    # Sufficient statistics
    alpha = np.fft.rfft(x, norm='ortho') # Fourier transform with 1/sqrt(N) included
    bLambda = np.mean(np.abs(alpha)**2, axis=0) # Usual result of PSD
    bLambda[0] = np.var(alpha[:, 0])
    Nf = len(bLambda)
 
    P_Lambda = np.nan * np.ones((Nf, 2, N_y)) # [freq, Lambda or P(Lambda), i_Lambda]
    
    x_bins = np.linspace(np.min(np.log10(freq[f_flt])), np.max(np.log10(freq[f_flt])), N_x)
    x_dig = np.digitize(np.log10(freq[f_flt]), x_bins,right=True) # 1st bin to have index 0
    Lambda = np.nan * np.ones_like(x_bins)
    F_pcolor = np.zeros([len(x_bins), 2, N_y])
    Q_pcolor = np.zeros(len(x_bins))
    bin_cntx = np.zeros(len(x_bins)) # bin count
    Lambda_ci = np.zeros([len(x_bins),2]) # bin count

    Lambda_uplim = 10*max(bLambda[1:]) # domain upper limit
    Lambda_dolim = 0.5*min(bLambda[1:-1]) # domain lower limit
    Lambda_vec = np.linspace(np.log10(Lambda_dolim), np.log10(Lambda_uplim), N_y+1)
    Lambda_vec = 10**Lambda_vec
    dL = Lambda_vec[1:]-Lambda_vec[:-1]
    Lambda_vec = np.sqrt(Lambda_vec[1:]*Lambda_vec[:-1])
    
    logger = _configure_logger(verbose)
    for i in range(len(freq[f_flt])):
        bi = x_dig[i]
        bin_cntx[bi] += 1
        Q_pcolor[bi] += bLambda[i+1]
    if verbose:
        logger.info('Calculating clustered auto corr.')
    for bi in range(N_x):
        if bin_cntx[bi] != 0:
            QQ=Q_pcolor[bi]/bin_cntx[bi]
      
            F_pcolor[bi,0,:] = Lambda_vec*dt
            F_pcolor[bi,1,:] = P_Lambdaf_D(Lambda_vec,QQ,M*bin_cntx[bi],Nf//2,Nf)
            F_pcolor[bi,1,:] /= max(F_pcolor[bi,1,:])

            if mode == 'bar': Lambda[bi] = QQ
            elif mode == 'mele': Lambda[bi] = np.mean(Lambda_vec*F_pcolor[bi,1]*dL)/np.mean(F_pcolor[bi, 1]*dL)
            else: Lambda[bi] = Lambda_vec[np.argmax(F_pcolor[bi, 1])]

            Lambda_ci[bi]=confidence_ivals_log(Lambda_vec,dL,F_pcolor[bi,1],tol/2,Lambda[bi])
        if verbose:
            logger.info('Clustered auto corr. %d/%d', bi + 1, N_x)
                
    result = {'freq': 10**x_bins[bin_cntx>0], 'Lambda': Lambda[bin_cntx>0]*dt, 'Lambda_ci':Lambda_ci[bin_cntx>0]*dt, 'Lambda_vec':Lambda_vec*dt, 'M_eff':M*bin_cntx[bin_cntx>0]}        
    if return_P: result.update({'P_Lambda': F_pcolor[bin_cntx>0,1,:]})
    return result


#----------------Cross-correlations--------------------

from scipy import optimize, integrate

def P_sf_phif_D(sf, phif, bsf, bphif, M, i_f, Nf):
#     N = 2*(Nf-1)
    df = Df(i_f, Nf)
    MM=(M-(i_f==0))*df
    delta=((i_f!=0 and i_f!=Nf-1) or (i_f==0 and (phif==0 or phif==np.pi))\
           or (i_f==Nf-1 and (phif==0 or phif==np.pi)))*1.
    if type(sf)==list:
        sf=np.array(sf)
        prefactor = np.empty_like(sf)
        prefactor[(sf < 1)] = MM*np.log(1.-sf[(sf < 1)]**2)
        prefactor[~(sf < 1)] = -np.inf*sf[~(sf < 1)]
    elif type(sf)==np.ndarray:
        prefactor = np.empty_like(sf)
        prefactor[(sf < 1)] = MM*np.log(1.-sf[(sf < 1)]**2)
        prefactor[~(sf < 1)] = -np.inf*sf[~(sf < 1)]
    else:
        if sf==1: prefactor = -np.inf
        else: prefactor = MM*np.log(1.-sf**2)

    zf = sf*bsf*np.cos(phif-bphif)    
    factor_log=(0.5-2*MM)*np.log(1-zf)
    hyp2_term=np.log(spec.hyp2f1(0.5,0.5,2*MM+0.5,0.5*(1+zf)))
    tot=prefactor+factor_log+hyp2_term
    return delta*np.exp(prefactor+factor_log+hyp2_term-(MM*np.log(1-bsf**2)\
            +(0.5-2*MM)*np.log(1-bsf**2)+np.log(spec.hyp2f1(0.5,0.5,2*MM+0.5,0.5*(1+bsf**2)))))

def log_i0(x):
    '''Logarithm of Modified Bessel Function of the First Kind'''
    if type(x)==list:
        x=np.array(x)
    if type(x)==np.ndarray:
        ret=np.empty_like(x)
        ret[x>709]=-0.5*np.log(2*np.pi)-0.5*np.log(x[x>709])+x[x>709]
        ret[x<=709]=np.log(spec.i0(x[x<=709]))
        return ret
    else:
        if x>709:
            return -0.5*np.log(2*np.pi)-0.5*np.log(x)+x
        else:
            return np.log(spec.i0(x))
def log_k0(x):
    '''Logarithm of Modified Bessel Function of the Second Kind'''
    if type(x)==list:
        x=np.array(x)
    if type(x)==np.ndarray:
        ret=np.empty_like(x)
        ret[x>742]=0.5*np.log(np.pi/2)-0.5*np.log(x[x>742])-x[x>742]
        ret[x<=742]=np.log(spec.k0(x[x<=742]))
        return ret
    else:
        if x>742:
            return 0.5*np.log(np.pi/2)-0.5*np.log(x)-x
        else:
            return np.log(spec.k0(x))

def log_F(x,L=1,m=2,bsf=0.5):
    '''Logarithm of integrand'''
    return (m-0.5)*np.log(x)-1.5*np.log(x+1)+log_i0(2*m/L*x)+log_k0(2*m/L/bsf*np.sqrt(x*(x+1)))

def F_norm(x,xmax=1,L=1,m=2,bsf=0.5):
    '''Scaled integrand'''
    return np.exp(log_F(x,L=L,m=m,bsf=bsf)-log_F(xmax,L=L,m=m,bsf=bsf))
    
def find_max(L=1,m=1,bsf=0.5,x0=1e-25):
    '''Find maximum of the integrand'''
    def pol(x):
        return (m-5/4)/x-7/4/(x+1)+2*m/L-m/L/bsf*(2*x+1)/np.sqrt(x*(x+1))
    return sp.optimize.root_scalar(pol,bracket=[1e-25,1e11],x0=x0,xtol=1e-25,maxiter=200,method='brentq').root

def logP_LambdaAB(L,m=2,bsf=0.5,tol=1e-2):
    '''Find integration limits given tolerance,
    calculate logarithm of the integral and revert scaling'''
    n_points = 5000 if m>5e4 else 500
    div = 2 if m<1e4 else 1+1/m**(1/3)
    
    xmax = find_max(L=L,m=m,bsf=bsf)
    x_lft = xmax; x_rgt = xmax;
    
    while(F_norm(x_lft,xmax,L,m,bsf)>tol):
        x_lft = x_lft/div
    while(F_norm(x_rgt,xmax,L,m,bsf)>tol):
        x_rgt = x_rgt*div

    xarr=np.arange(np.log(x_lft),np.log(x_rgt),(np.log(x_rgt)-np.log(x_lft))/n_points)
    xarr=np.exp(xarr)
    #     integral, err = sp.integrate.quad(F_norm,x_lft, x_rgt,args=(xmax,m,L,bsf))
    #     integral = sum(F_norm(xarr[1:],xmax,m=m,L=L,bsf=bsf)*(xarr[1:]-xarr[:-1]))
    integral = sum(F_norm(np.sqrt(xarr[1:]*xarr[:-1]),xmax,L=L,m=m,bsf=bsf)*(xarr[1:]-xarr[:-1]))

    return np.log(integral)+log_F(xmax,L,m,bsf)-(2*m+1)*np.log(L)

def P_LambdaAB(L,M,bsf,i_f,Nf):
    '''Calculate probability distribution with respect to the value at Lambda=1'''
    m=Df(i_f,Nf)*(M-1.*(i_f==0))
    return np.exp(logP_LambdaAB(L,m,bsf)-logP_LambdaAB(1,m,bsf))

def log_F_phi(x,L=1,m=2,bsf=0.5,dphif=0):
    '''Logarithm of integrand'''
    return (m-0.5)*np.log(x)-1.5*np.log(x+1)+(2*m*np.cos(dphif)/L*x)+log_k0(2*m/L/bsf*np.sqrt(x*(x+1)))

def log_F_phi_vec(x,L,m=2,bsf=0.5,dphif=0):
    '''Logarithm of integrand'''
    if type(x)==list: x=np.array(x)
    if type(L)==list: L=np.array(L)
    if type(dphif)==list: dphif=np.array(dphif)
    if type(L)==np.ndarray and type(x)==np.ndarray:
        L=np.transpose(np.tile(L,(np.shape(x)[1],1)))
    if type(dphif)==np.ndarray and type(x)==np.ndarray:
        dphif=np.transpose(np.tile(dphif,(np.shape(x)[1],1)))
    return (m-0.5)*np.log(x)-1.5*np.log(x+1)+(2*m*np.cos(dphif)/L*x)+log_k0(2*m/L/bsf*np.sqrt(x*(x+1)))

def F_phi_norm(x,xmax=1,L=1,m=2,bsf=0.5,dphif=0):
    '''Scaled integrand'''
    return np.exp((m-0.5)*np.log(x)-1.5*np.log(x+1)+(2*m*np.cos(dphif)/L*x)+log_k0(2*m/L/bsf*np.sqrt(x*(x+1)))\
                  -((m-0.5)*np.log(xmax)-1.5*np.log(xmax+1)+(2*m*np.cos(dphif)/L*xmax)+log_k0(2*m/L/bsf*np.sqrt(xmax*(xmax+1)))))

def F_phi_norm_vec(x,L,xmax=1,m=2,bsf=0.5,dphif=0):
    '''Scaled integrand'''
    if type(L)==list: L=np.array(L)
    if type(dphif)==list: dphif=np.array(dphif)
    if type(x)==list: x=np.array(x)
    if type(xmax)==list: xmax=np.array(xmax)
    if type(L)==np.ndarray:
        if type(dphif)==np.ndarray: assert L.shape==dphif.shape
        if type(xmax)==np.ndarray: 
            assert np.shape(xmax)==np.shape(L)
            assert len(np.shape(xmax))==1
            log_max = log_F_phi(xmax,L=L,m=m,bsf=bsf,dphif=dphif)
            log_max = np.transpose(np.tile(log_max,(np.shape(x)[1],1)))
        else: 
            log_max = log_F_phi(xmax,L=L,m=m,bsf=bsf,dphif=dphif)
            log_max = np.transpose(np.ones_like(np.outer(x,L)*1.)*log_max)
    return np.exp(log_F_phi_vec(x,L=L,m=m,bsf=bsf,dphif=dphif)-log_max)

def logP_LambdaAB_phi(L,m=2,bsf=0.5,dphif=0,tol=1e-2,n=200):
    '''Find integration limits given tolerance,
    calculate logarithm of the integral and revert scaling'''
    n_points = 10*n if m>5e4 else n
    div = 2 if m<1e4 else 1+1/m**(1/3)
    
    xmax = find_max_phi(L=L,m=m,bsf=bsf,dphif=dphif)
    x_lft = xmax/div; x_rgt = xmax*div;
    c1,c2=0,0
    log_max=((m-0.5)*np.log(xmax)-1.5*np.log(xmax+1)+(2*m*np.cos(dphif)/L*xmax)+log_k0(2*m/L/bsf*np.sqrt(xmax*(xmax+1))))
    while(np.exp((m-0.5)*np.log(x_lft)-1.5*np.log(x_lft+1)\
                 +(2*m*np.cos(dphif)/L*x_lft)+log_k0(2*m/L/bsf*np.sqrt(x_lft*(x_lft+1)))-log_max)>tol):
        x_lft = x_lft/div
    while(np.exp((m-0.5)*np.log(x_rgt)-1.5*np.log(x_rgt+1)\
                 +(2*m*np.cos(dphif)/L*x_rgt)+log_k0(2*m/L/bsf*np.sqrt(x_rgt*(x_rgt+1)))-log_max)>tol):
        x_rgt = x_rgt*div
    xarr=np.linspace(np.log(x_lft),np.log(x_rgt),n_points,endpoint=True)
    xarr=np.exp(xarr)
    integral = sum(F_phi_norm(np.sqrt(xarr[1:]*xarr[:-1]),xmax,L=L,m=m,bsf=bsf,dphif=dphif)*(xarr[1:]-xarr[:-1]))

    return np.log(integral)+log_F_phi(xmax,L,m,bsf,dphif)-(2*m+1)*np.log(L)

def logP_LambdaAB_phi_vec(L,m=2,bsf=0.5,dphif=0,tol=1e-2,n=200):
    '''Find integration limits given tolerance,
    calculate logarithm of the integral and revert scaling'''
    n_points = 10*n if m>5e4 else n
    div = 2 if m<1e4 else 1+1/m**(1/3)
    dphifi=dphif
    if type(L)==list: L=np.array(L)
    if type(dphif)==list: dphif=np.array(dphif);
    if type(L)==np.ndarray:
        xmax=np.ones_like(L)*1.
        xarr = np.zeros((len(L),n_points))
        for i, Li in enumerate(L):
            if type(dphif)==np.ndarray: dphifi=dphif[i]
            xmax[i] = find_max_phi(L=Li,m=m,bsf=bsf,dphif=dphifi)
            x_lft = xmax[i]/div; x_rgt = xmax[i]*div;
            log_max=log_F_phi(xmax[i],L=Li,m=m,bsf=bsf,dphif=dphifi)
            while(np.exp(log_F_phi(x_lft,Li,m,bsf,dphifi)-log_max)>tol):
                x_lft = x_lft/div
            while(np.exp(log_F_phi(x_rgt,Li,m,bsf,dphifi)-log_max)>tol):
                x_rgt = x_rgt*div
            xarr[i]=np.linspace(np.log(x_lft),np.log(x_rgt),n_points,endpoint=True)
            xarr[i]=np.exp(xarr[i])
        integral = np.sum(F_phi_norm_vec(np.sqrt(xarr[:,1:]*xarr[:,:-1]),L,xmax,m=m,bsf=bsf,dphif=dphif)*(xarr[:,1:]-xarr[:,:-1]),axis=1)
        
    else:
        xmax = find_max_phi(L=L,m=m,bsf=bsf,dphif=dphif)
        x_lft = xmax/div; x_rgt = xmax*div;
        while(F_phi_norm(x_lft,xmax,L,m,bsf,dphif)>tol):
            x_lft = x_lft/div
        while(F_phi_norm(x_rgt,xmax,L,m,bsf,dphif)>tol):
            x_rgt = x_rgt*div

        xarr=np.linspace(np.log(x_lft),np.log(x_rgt),n_points,endpoint=True)
        xarr=np.exp(xarr)
        integral = sum(F_phi_norm(np.sqrt(xarr[1:]*xarr[:-1]),xmax,L=L,m=m,bsf=bsf,dphif=dphif)*(xarr[1:]-xarr[:-1]))

    return np.log(integral)+log_F_phi(xmax,L,m,bsf,dphif)-(2*m+1)*np.log(L)


def find_max_phi(L=1,m=1,bsf=0.5,dphif=0,x0=1e-37):
    '''Find maximum of the integrand'''

    def pol(x):
        return (m-3/4)/x-7/4/(x+1)+2*m*np.cos(dphif)/L-m/L/bsf*(2*x+1)/np.sqrt(x*(x+1))
    return sp.optimize.root_scalar(pol,bracket=[1e-40,1e7],x0=1e-37,xtol=1e-40,maxiter=200,method='brentq').root

def CSD(A, B, dt, Ndist=100, Ndist_phi=None, Ndist_sf=None, f_lims=None, tol=0.1, phase_center=0.5*np.pi,
        return_P=False, mode='mle', verbose=True):
    """Calculate cross-correlation distributions and estimators.

    Input:
        A, B: arrays of measurements in Rspace, [block, time].
    Source: [CFCFT], [PRI]."""
    assert len(A.shape) == 2, len(B.shape) == 2
    M = len(A) # #blocks
    N = len(A[0]) # #points per block
    if N%2 != 0: N-=1; A = A[:,:-1]; B = B[:,:-1];
    assert N%2 == 0
    assert M>=1
    freq = np.fft.rfftfreq(A.shape[1], d=dt)
    f_flt = np.ones(len(freq), dtype= bool) if f_lims is None else (min(f_lims) <= freq) & (freq < max(f_lims))
    f_flt[0]=False
    # Sufficient statistics
    Af = np.fft.rfft(A, norm='ortho') # Fourier transform with 1/sqrt(N) included
    Bf = np.fft.rfft(B, norm='ortho')

    bLambda_A = np.mean(np.abs(Af)**2, axis=0) # Usual result of PSD
    bLambda_B = np.mean(np.abs(Bf)**2, axis=0)
    Nf = len(bLambda_A)
    AfBf_cos_w0 = np.real(Af)*np.real(Bf) + np.imag(Af)*np.imag(Bf) # Some definitions
    AfBf_sin_w0 = np.real(Af)*np.imag(Bf) - np.imag(Af)*np.real(Bf)
    Qcos = np.mean(AfBf_cos_w0, axis=0)
    Qsin = np.mean(AfBf_sin_w0, axis=0)
    Qcos[0] = np.mean(np.real(Af[:, 0]-np.mean(Af[:, 0]))*np.real(Bf[:, 0]-np.mean(Bf[:, 0])))
    Qsin[0] = 0.
    bLambda = np.sqrt(Qcos**2+Qsin**2) # Usual estimation of CSD
    bsf = np.sqrt(Qcos**2+Qsin**2)/np.sqrt(bLambda_A*bLambda_B) # Usual estimation of correlation coefficient
    bphif = np.angle(Qcos+Qsin*1j) # Usual estimation of the phase
    
    if Ndist_sf is None: Ndist_sf=Ndist
    if Ndist_phi is None: Ndist_phi=Ndist

    # Make sure the phase vector includes 0 and np.pi for the dirac deltas in the prob. distribution
    phif_vec=np.concatenate((np.linspace(0,np.pi,Ndist_phi//2,endpoint=False)\
            ,np.linspace(np.pi,2*np.pi,Ndist_phi//2,endpoint=False)%(2*np.pi))) # Possible values of phase
    sf_vec = np.linspace(0., 1, Ndist_sf) # Possible values of sf
    dphif = np.mean(np.diff(phif_vec))

    P_sf_phif = np.nan*np.ones((len(freq), len(sf_vec), len(phif_vec))) 

    Lambda = np.nan * np.ones(Nf)
#     Lambda_MELE = np.nan * np.ones(Nf)
    Lambda_ci = np.nan * np.ones([Nf,2])
    sf_ci = np.nan * np.ones([Nf,2])
    phif_ci = np.nan * np.ones([Nf,2])
    P_Lambda = np.nan * np.ones((Nf, 2, Ndist)) # [freq, Lambda or P(Lambda), i_Lambda]

    logger = _configure_logger(verbose)
    if verbose:
        logger.info('M = %s', M)
        logger.info('Calculating cross corr.')
    for i_f in range(len(freq)):
        m=Df(i_f,Nf)*(M-1.*(i_f==0))
        if f_flt[i_f]:
            if verbose:
                logger.info('Calculating cross corr. %d/%d', int(np.sum(f_flt[:i_f])) + 1, int(len(freq[f_flt])))
            Lambda_uplim = 1 # domain upper limit
            logP_ref = logP_LambdaAB(1,m,bsf[i_f])
            while (logP_LambdaAB(Lambda_uplim,m,bsf[i_f])-logP_ref)> np.log(1.e-2):
                Lambda_uplim *= 2.
            Lambda_vec = np.linspace(-3, np.log10(Lambda_uplim), Ndist+1)
            Lambda_vec=10**Lambda_vec
            dL=Lambda_vec[1:]-Lambda_vec[:-1]
            Lambda_vec = np.sqrt(Lambda_vec[1:]*Lambda_vec[:-1])
            P_Lambda[i_f, 0, :] = Lambda_vec*bLambda[i_f]*dt
            P_Lambda[i_f, 1, :] = [(logP_LambdaAB(L,m,bsf[i_f])) for L in Lambda_vec]
            P_Lambda[i_f, 1, :] = np.exp(P_Lambda[i_f, 1, :]-logP_ref)
            P_Lambda[i_f, 1, :] = P_Lambda[i_f, 1, :]/max(P_Lambda[i_f, 1, :])
                
#             Lambda_MELE[i_f] = bLambda[i_f]*np.mean(Lambda_vec[1:]*\
#                             np.sqrt(P_Lambda[i_f, 1, 1:]*P_Lambda[i_f, 1, :-1])*(Lambda_vec[1:]-Lambda_vec[:-1]))\
#             /np.mean(np.sqrt(P_Lambda[i_f, 1, 1:]*P_Lambda[i_f, 1, :-1])*(Lambda_vec[1:]-Lambda_vec[:-1]))
            if mode == 'bar': Lambda[i_f] = bLambda[i_f]
            elif mode == 'mele': Lambda[i_f] = bLambda[i_f]*np.mean(Lambda_vec[1:]*P_Lambda[i_f, 1, 1:]*(Lambda_vec[1:]-Lambda_vec[:-1]))\
            /np.mean(P_Lambda[i_f, 1, 1:]*(Lambda_vec[1:]-Lambda_vec[:-1]))
            else: Lambda[i_f] = bLambda[i_f]*Lambda_vec[np.argmax(P_Lambda[i_f, 1, :])]
            
            Lambda_ci[i_f]=confidence_ivals_log(P_Lambda[i_f, 0, :],dL*bLambda[i_f]*dt,P_Lambda[i_f, 1, :],tol/2,Lambda[i_f])
            for i, phif in enumerate(phif_vec):
                # Calculate probability distributions for sf and phif
                P_sf_phif[i_f, :, i] = P_sf_phif_D(sf_vec, phif, bsf[i_f], bphif[i_f], M, i_f, len(freq))
            # Normalize the integral of the distribution
            P_sf_phif[i_f] = P_sf_phif[i_f]/(sf_vec[1]*dphif*np.nansum(P_sf_phif[i_f],axis=(0,1)))     
    
    P_sf = np.sum(P_sf_phif,axis=2)*dphif
    for i in range(len(P_sf[:,0])):
        P_sf[i]=P_sf[i]/max(P_sf[i])
    if mode == 'bar':  sf = bsf
    elif mode == 'mele':  sf = np.mean(sf_vec * P_sf, axis = 1)/np.mean(P_sf, axis = 1)
    else: sf = sf_vec[np.argmax(P_sf,axis=1)]

    # Probability distribution for phif by integral over sf
    P_phif = np.sum(P_sf_phif,axis=1)*sf_vec[1]
    for i in range(len(P_phif[:,0])):
        P_phif[i]=P_phif[i]/max(P_phif[i])
    phif_MLE = phif_vec[np.argmax(P_phif,axis=1)]
    phif_MLE = shift_phase_coord(phif_MLE,phase_center)
    phif_MELE = np.zeros(len(phif_MLE))
    for i_f in range(len(phif_MLE)):
        if f_flt[i_f]:
            phif_MELE[i_f] = np.mean(shift_phase_coord(phif_vec, phif_MLE[i_f]) * P_phif[i_f])/np.mean(P_phif[i_f])
            phif_MELE[i_f] = shift_phase_coord(phif_MELE[i_f], np.mean(phif_vec))
            phif_ci[i_f,:] = confidence_ivals(shift_phase_coord(phif_vec, phif_MLE[i_f]), P_phif[i_f],\
                                               0.5*tol, phif_MLE[i_f])

            sf_ci[i_f,:] = confidence_ivals(sf_vec, P_sf[i_f], 0.5*tol, sf[i_f])
        # Make maximum equal 1
        P_sf_phif[i_f] = P_sf_phif[i_f]/(sf_vec[1]*dphif*np.nansum(P_sf_phif[i_f],axis=(0,1)))
        
    if mode == 'bar': phif = shift_phase_coord(bphif,phase_center)
    elif mode == 'mele': phif = shift_phase_coord(phif_MELE,phase_center)
    else: phif = shift_phase_coord(phif_MLE,phase_center)
    phif_vec = shift_phase_coord(phif_vec,phase_center)
    
    result={'freq': freq[f_flt], 'sf':sf[f_flt], 'sf_ci':sf_ci[f_flt],\
            'phif':phif[f_flt], 'phif_ci':phif_ci[f_flt], 'phif_vec': phif_vec,\
            'Lambda': Lambda[f_flt]*dt, 'Lambda_ci': Lambda_ci[f_flt]}
    if return_P:
        result.update({'P_Lambda': P_Lambda[f_flt,1], 'Lambda_vec': P_Lambda[f_flt,0], 'P_sf': P_sf[f_flt], 'sf_vec':sf_vec,\
                       'P_phif': P_phif[f_flt]})
    return result

#----------------Cross-correlations clustered--------------------

def CSD_clus(A, B, dt=1, N_x=100, N_y=100, tol=0.1, f_lims=None, phase_center=0.5*np.pi,
             return_P=False, mode='mle', verbose=True):
    """Calculate cross-correlation distributions and estimators.

    Input:
        A, B: arrays of measurements in Rspace, [block, time].
    Source: [CFCFT], [PRI]."""
    assert len(A.shape) == 2, len(B.shape) == 2
    M = len(A) # #blocks
    N = len(A[0]) # #points per block
    if N%2 != 0: N-=1; A = A[:,:-1]; B = B[:,:-1];
    assert N%2 == 0
    assert M>=1
    freq = np.fft.rfftfreq(A.shape[1], d=dt)
    f_flt = np.ones(len(freq), dtype= bool) if f_lims is None else (min(f_lims) <= freq) & (freq < max(f_lims))
    f_flt[0] = False # don't include freq=0

    # Sufficient statistics
    Af = np.fft.rfft(A, norm='ortho') # Fourier transform with 1/sqrt(N) included
    Bf = np.fft.rfft(B, norm='ortho')

    bLambda_A = np.mean(np.abs(Af)**2, axis=0) # Usual result of PSD
    bLambda_B = np.mean(np.abs(Bf)**2, axis=0)
    Nf = len(bLambda_A)
    AfBf_cos_w0 = np.real(Af)*np.real(Bf) + np.imag(Af)*np.imag(Bf) # Some definitions
    AfBf_sin_w0 = np.real(Af)*np.imag(Bf) - np.imag(Af)*np.real(Bf)
    Qcos = np.mean(AfBf_cos_w0, axis=0)
    Qsin = np.mean(AfBf_sin_w0, axis=0)
    Qcos[0] = np.mean(np.real(Af[:, 0]-np.mean(Af[:, 0]))*np.real(Bf[:, 0]-np.mean(Bf[:, 0])))
    Qsin[0] = 0.
    bLambda = np.sqrt(Qcos**2+Qsin**2) # Usual estimation of CSD
    bsf = np.sqrt(Qcos**2+Qsin**2)/np.sqrt(bLambda_A*bLambda_B) # Usual estimation of correlation coefficient
    bphif = np.angle(Qcos+Qsin*1j) # Usual estimation of the phase
    

#     P_Lambda = np.nan * np.ones((Nf, 2, N_y)) # [freq, Lambda or P(Lambda), i_Lambda]
    phif_vec = np.linspace(-0.*np.pi, 2*np.pi, N_y,endpoint=False) # Possible values of phase
    sf_vec = np.linspace(0., 1, N_y)[:] # Possible values of sf
    dphif = np.mean(np.diff(phif_vec))
    
    x_bins = np.linspace(np.min(np.log10(freq[:][f_flt])), np.max(np.log10(freq[:][f_flt])), N_x)
    x_dig = np.digitize(np.log10(freq[:][f_flt]), x_bins,right=True) # 1st bin to have index 0
    Lambda = np.nan * np.ones_like(x_bins)
    Lambda_MELE = np.nan * np.ones_like(x_bins)
    F_pcolor = np.zeros([len(x_bins), N_y])
    F_pcolor_norm = np.zeros([len(x_bins), len(phif_vec),len(sf_vec)])
    Qcos_pcolor = np.zeros(len(x_bins))
    Qsin_pcolor = np.zeros(len(x_bins))
    QA_pcolor = np.zeros(len(x_bins))
    QB_pcolor = np.zeros(len(x_bins))
    bin_cntx = np.zeros(len(x_bins)) # bin count
    Lambda_ci = np.zeros([len(x_bins),2]) # bin count
    sf_ci = np.zeros([len(x_bins),2]) # bin count
    phif_ci = np.zeros([len(x_bins),2]) # bin count
    
    bsf_avg_col=np.zeros(len(x_bins))
    bphif_avg_col=np.zeros(len(x_bins))

    Lambda_uplim = 10*max(bLambda[1:]) # domain upper limit, skips 0 freq component (DC shift which can be 0)
    Lambda_dolim = 0.1*min(bLambda[1:]) # domain lower limit, skips 0 freq component (DC shift which can be 0)
    Lambda_vec = np.linspace(np.log10(Lambda_dolim), np.log10(Lambda_uplim), N_y+1)
    Lambda_vec = 10**Lambda_vec
    dL=Lambda_vec[1:]-Lambda_vec[:-1]
    Lambda_vec = np.sqrt(Lambda_vec[1:]*Lambda_vec[:-1])
    logger = _configure_logger(verbose)
    for i in range(len(freq[f_flt])):
        bi = x_dig[i]
        bin_cntx[bi] += 1
        Qcos_pcolor[bi] += Qcos[i+1]
        Qsin_pcolor[bi] += Qsin[i+1]
        QA_pcolor[bi] += bLambda_A[i+1]
        QB_pcolor[bi] += bLambda_B[i+1]
    if verbose:
        logger.info('Clustered cross corr.')
    for bi in range(N_x):
        if bin_cntx[bi] != 0:
            QQc=Qcos_pcolor[bi]/bin_cntx[bi]
            QQs=Qsin_pcolor[bi]/bin_cntx[bi]
            QQa=QA_pcolor[bi]/bin_cntx[bi]
            QQb=QB_pcolor[bi]/bin_cntx[bi]
            bLambda_avg=np.sqrt(QQc**2+QQs**2)
            bsf_avg=np.sqrt(QQc**2+QQs**2)/np.sqrt(QQa*QQb)
            bphif_avg=np.angle(QQc+QQs*1j)
            bsf_avg_col[bi]=bsf_avg
            bphif_avg_col[bi]=bphif_avg
        
            Lambda_vec_clus = Lambda_vec/bLambda_avg
            
            F_pcolor[bi, :] = [(logP_LambdaAB(L,M*bin_cntx[bi],bsf_avg)) for L in Lambda_vec_clus]
            F_pcolor[bi, :] = np.exp(F_pcolor[bi, :]-logP_LambdaAB(1,M*bin_cntx[bi],bsf_avg))
            F_pcolor[bi] = F_pcolor[bi]/max(F_pcolor[bi])
            
            if mode == 'bar': Lambda[bi] = bLambda_avg
            elif mode == 'mele': Lambda[bi] = np.mean(Lambda_vec*F_pcolor[bi]*dL)/np.mean(F_pcolor[bi]*dL)
            else: Lambda[bi] = Lambda_vec[np.argmax(F_pcolor[bi, :])]
                
            Lambda_ci[bi]=confidence_ivals_log(Lambda_vec,dL,F_pcolor[bi],tol/2,Lambda[bi])
            
        for bj in range(N_y):
            F_pcolor_norm[bi,bj]=P_sf_phif_D(sf_vec, phif_vec[bj], bsf_avg, bphif_avg, M*bin_cntx[bi],Nf//2,Nf)
        F_pcolor_norm[bi]=F_pcolor_norm[bi]/(sf_vec[1]*dphif*np.nansum(F_pcolor_norm[bi],axis=(0,1)))
        
        if verbose:
            logger.info('Clustered cross corr. %d/%d', bi + 1, N_x)
                
    P_sf=np.sum(F_pcolor_norm,axis=1)*dphif
    for i in range(len(P_sf[:,0])):
        P_sf[i]=P_sf[i]/max(P_sf[i])
#     P_sf=P_sf/np.max(P_sf,axis=1)
    if mode == 'bar':  sf = bsf_avg_col
    elif mode == 'mele':  sf = np.mean(sf_vec * P_sf, axis = 1)/np.mean(P_sf, axis = 1)
    else: sf = sf_vec[np.argmax(P_sf,axis=1)]
    
    
    P_phif=np.sum(F_pcolor_norm,axis=2)*sf_vec[1]
    for i in range(len(P_phif[:,0])):
        P_phif[i]=P_phif[i]/max(P_phif[i])
    phif_MLE = phif_vec[np.argmax(P_phif,axis=1)]
#     phif_MLE = shift_phase_coord(phif_MLE, phase_center)
    phif_MELE = np.zeros(len(phif_MLE))
    for i_f in range(len(x_bins)):
        phif_MELE[i_f] = np.mean(shift_phase_coord(phif_vec, phif_MLE[i_f]) * P_phif[i_f])/np.mean(P_phif[i_f])
#         phif_MELE[i_f] = shift_phase_coord(phif_MELE[i_f], phasa)
        phif_ci[i_f,:] = confidence_ivals(shift_phase_coord(phif_vec, phif_MLE[i_f]), P_phif[i_f],\
                                           0.5*tol, phif_MLE[i_f])
        sf_ci[i_f,:] = confidence_ivals(sf_vec, P_sf[i_f], 0.5*tol, sf[i_f])
    
    if mode == 'bar': phif = shift_phase_coord(bphif_avg_col, phase_center)
    elif mode == 'mele': phif = shift_phase_coord(phif_MELE, phase_center)
    else: phif = shift_phase_coord(phif_MLE, phase_center)
    phif_vec = shift_phase_coord(phif_vec,phase_center)
    phif_ord = np.argsort(phif_vec)
    phif_vec, P_phif = phif_vec[phif_ord], P_phif[:,phif_ord]
    result = {'freq': 10**x_bins[bin_cntx>0], 'sf':sf[bin_cntx>0],'sf_ci':sf_ci[bin_cntx>0],\
              'phif':phif[bin_cntx>0],'phif_ci':phif_ci[bin_cntx>0],\
              'Lambda': Lambda[bin_cntx>0]*dt, 'Lambda_ci':Lambda_ci[bin_cntx>0]*dt,'M_eff':M*bin_cntx[bin_cntx>0]}        
    if return_P == True:
        result.update({'P_Lambda': F_pcolor[bin_cntx>0], 'Lambda_vec':Lambda_vec*dt,\
                      'P_sf': P_sf[bin_cntx>0], 'sf_vec':sf_vec,\
                      'P_phif': P_phif[bin_cntx>0], 'phif_vec':phif_vec})
    return result


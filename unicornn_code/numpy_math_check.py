"""
Independent NumPy check of the formulas behind unicornn.py (needs only numpy, no PyTorch).

It re-implements one UnICORNN layer (forward, hand-derived backward, exact inverse) and checks:
  1. the backward formulas against finite differences (w, c and the input x),
  2. that the exact inverse rebuilds all states from the final state (float64 and float32, N = 1000),
  3. that the one-step Jacobian of symplectic Euler has determinant 1.
Run:  python numpy_math_check.py
"""
import numpy as np
rng=np.random.default_rng(0)
N,B,m=50,4,6; dt,alpha=0.2,3.0
x=rng.standard_normal((N,B,m)); w=rng.random(m); c=rng.standard_normal(m); tgt=rng.standard_normal((N,B,m))
sig=lambda v:1/(1+np.exp(-v))
def fwd(x,w,c):
    h=dt*sig(c); y=np.zeros((B,m)); z=np.zeros((B,m)); out=np.empty_like(x)
    for n in range(N):
        z=z-h*(np.tanh(w*y+x[n])+alpha*y); y=y+h*z; out[n]=y
    return out,y,z
def loss(x,w,c): return 0.5*((fwd(x,w,c)[0]-tgt)**2).sum()
def bwd(x,w,c):
    out,y,z=fwd(x,w,c); sc=sig(c); h=dt*sc; go=out-tgt
    Yb=np.zeros((B,m)); Zb=np.zeros((B,m)); gx=np.empty_like(x); gw=np.zeros((B,m)); gh=np.zeros((B,m)); err=0
    for n in range(N-1,-1,-1):
        Yb=Yb+go[n]; zt=Zb+h*Yb
        yp=y-h*z; t=np.tanh(w*yp+x[n]); s=1-t*t; g=t+alpha*yp; zp=z+h*g
        gx[n]=-h*zt*s; gw=gw-h*zt*s*yp; gh=gh+Yb*z-zt*g
        Yb=Yb-h*zt*(s*w+alpha); Zb=zt; y,z=yp,zp
    return gx,gw.sum(0),gh.sum(0)*dt*sc*(1-sc)
gx,gw,gc=bwd(x,w,c); eps=1e-6
fw=np.array([(loss(x,w+eps*np.eye(m)[i],c)-loss(x,w-eps*np.eye(m)[i],c))/(2*eps) for i in range(m)])
fc=np.array([(loss(x,w,c+eps*np.eye(m)[i])-loss(x,w,c-eps*np.eye(m)[i]))/(2*eps) for i in range(m)])
xi=(1,2,3); xp=x.copy(); xm=x.copy(); xp[xi]+=eps; xm[xi]-=eps
fx=(loss(xp,w,c)-loss(xm,w,c))/(2*eps)
print("w  maxdiff",abs(gw-fw).max()); print("c  maxdiff",abs(gc-fc).max()); print("x  diff",abs(gx[xi]-fx))
# reversibility (float64 + float32), N=1000
for dtp in (np.float64,np.float32):
    NN=1000; xx=rng.standard_normal((NN,2,8)).astype(dtp); ww=rng.random(8).astype(dtp); h=dtp(0.05)
    y=np.zeros((2,8),dtp); z=np.zeros((2,8),dtp); ys=[];zs=[]
    for n in range(NN):
        z=z-h*(np.tanh(ww*y+xx[n])+dtp(5)*y); y=y+h*z; ys.append(y); zs.append(z)
    e=0
    for n in range(NN-1,0,-1):
        yp=y-h*z; z=z+h*(np.tanh(ww*yp+xx[n])+dtp(5)*yp); y=yp
        e=max(e,abs(y-ys[n-1]).max(),abs(z-zs[n-1]).max())
    print("reversibility",dtp.__name__,"max err",e)
# jacobian det
h,a=0.15,4.0
for _ in range(3):
    ww,xx,y0,z0=rng.standard_normal(4)
    def step(v):
        z2=v[1]-h*(np.tanh(ww*v[0]+xx)+a*v[0]); return np.array([v[0]+h*z2,z2])
    v=np.array([y0,z0]); J=np.array([(step(v+1e-6*e)-step(v-1e-6*e))/2e-6 for e in np.eye(2)]).T
    print("det J",np.linalg.det(J))

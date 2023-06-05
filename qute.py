#========================
# Qute module
# Yen Lee Loh 2023-6-4
#========================
from IPython.display import display,HTML,Markdown
display(HTML('''
<style>
h1 { background-color: #AEA; padding: 0.8ex 0.8ex 0.5ex 0.8ex; border: 2px solid #8C8; }
h2 { background-color: #AEE; padding: 0.8ex 0.8ex 0.5ex 0.8ex; border: 2px solid #9CC; }
h3 { background-color: #EEA; padding: 0.8ex 0.8ex 0.5ex 0.8ex; border: 2px solid #CC9; }
</style>'''))
display(Markdown(r'''
$\newcommand{\mean}[1]{\langle #1 \rangle}$
$\newcommand{\bra}[1]{\langle #1 \rvert}$
$\newcommand{\ket}[1]{\lvert #1 \rangle}$
$\newcommand{\adag}{a^\dagger}$
$\newcommand{\mat}[1]{\underline{\underline{\mathbf{#1}}}}$
$\newcommand{\beq}{\qquad\begin{align}}$
$\newcommand{\eeq}{\end{align}}$
$\newcommand{\half}{\frac{1}{2}}$
'''))

from collections.abc import Iterable
import numpy as np; from numpy import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import regex as re
import qiskit,qiskit_aer
import qiskit.visualization as qisvis
from qiskit.providers.fake_provider import fake_provider
vigoBackend = fake_provider.FakeVigo()
rng = random.default_rng()
cirStyle = {'fontsize':24, 'linecolor':'#999', 'displaycolor': {'cx':'#06C', 'ccx':'#06C', 'mcx':'#06c', 'cswap':'#06C'}}
cirOpts={'fold':-1, 'style':cirStyle}

def bstrFromList (bitlist):
  try:    return ''.join([str(c) for c in bitlist[::-1]])
  except: return bitlist
def bstrFromInteger (thing, numBits=-1):
  if numBits==-1:
    return bin(thing)[2:]
  else:
    return bin(thing)[2:].zfill(numBits)
def bstr (thing, numBits=-1):
  if isinstance (thing, Iterable):
    return bstrFromList(thing)
  else:
    return bstrFromInteger(thing, numBits)
def dstrFromString (thing):
  if isinstance(thing,str):
    numDigits = len(thing)
    fmt = '{:' + str(numDigits) + 'd}'
    return fmt.format (int(thing,2))
def dstr (thing):
  return dstrFromString (thing)


def run (cir, sim='auto', **kwargs):
  if sim=='auto':
    sim = qiskit.Aer.get_backend('statevector_simulator') # set up a statevector simulator
  return sim.run(qiskit.transpile(cir, sim), **kwargs).result()
  
def getCounts (res, **kwargs):   # wrapper for consistency
  return res.get_counts(**kwargs)
def padCounts(cts, numBits):
  def sanitize(count): return (0 if count==None else count)
  n = numBits
  return {bstr(i,n):sanitize(cts.get(bstr(i,n))) for i in range(2**n)}
def getStatevector (res, svName='Statevector'):
  return np.asarray(res.data() [svName])
def getStatevectors (res, pattern='psi'):
  dat = res.data()
  svNames = [key for key in dat.keys() if re.search(pattern, key)]
  svNames.sort()
  return np.asarray([dat[svName] for svName in svNames])
  
def plotStatevector(psi,ax='auto',figsize='auto',horiz=False,cmap=mpl.cm.hsv,tickInterval=1):
  #======== Determine dimensions
  nmax = int(np.log(len(psi)) / np.log(2)) # number of qubits
  lmax = 2**nmax                           # number of states
  probs = np.abs(psi)**2
  phases = np.remainder(np.angle(psi) / (2*np.pi), 1)
  #======== If user hasn't supplied an ax, create one now
  if ax=='auto':
    if figsize=='auto':
      figsize = (2,6) if horiz else (6,1)
    fig,ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure
  #======== Plot
  if horiz:
    for l in range(lmax):
      patch = patches.Rectangle([0,-l-1], probs[l],1, ec='black', fc=cmap(phases[l]))
      ax.add_patch(patch)
    ax.set_yticks([-l-1+.5 for l in range(0,lmax,tickInterval)])
    ax.set_yticklabels([bstr(l,nmax) for l in range(0,lmax,tickInterval)])
    ax.set_ylim(-lmax-.5, .5)
    ax.set_xlim(0, max(probs)*1.1)
  else:
    for l in range(lmax):
      patch = patches.Rectangle([l,0], 1,probs[l], ec='black', fc=cmap(phases[l]))
      ax.add_patch(patch)
    ax.set_xticks([l+.5 for l in range(lmax)])
    ax.set_xticklabels([bstr(l,nmax) for l in range(lmax)])
    ax.set_xlim(-.5, lmax+.5)
    ax.set_ylim(0, max(probs)*1.1)
  return fig,ax
    
def plotHistogram(counts,ax='auto',figsize='auto',horiz=False,fc='#9cf',textrot=0):
  #======== Pad counts
  if isinstance (counts, dict):
    nmax = len(list(counts.keys()) [0] )
    counts = padCounts(counts,nmax)
    counts = list(counts.values())
  #======== Determine dimensions
  nmax = int(np.log(len(counts)) / np.log(2)) # number of qubits
  lmax = 2**nmax                              # number of states
  ymax = max(counts)
  #======== If user hasn't supplied an ax, create one now
  if ax=='auto':
    if figsize=='auto':
      figsize = (2,6) if horiz else (6,1)
    fig,ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure
  #======== Plot
  if horiz:
    for l in range(lmax):
      c = counts[l]
      patch = patches.Rectangle([0,-l-1], c,1, ec='black', fc=fc)
      ax.add_patch(patch)
      if c>0:
        y = (.75*ymax if c<.5*ymax else .25*ymax)
        ax.text (y, -l-1+.5, c, va='center',ha='center')
    ax.set_yticks([-l-1+.5 for l in range(lmax)])
    ax.set_yticklabels([bstr(l,nmax) for l in range(lmax)])
    ax.set_ylim(-lmax-.5, .5)
    ax.set_xlim(0, ymax*1.02)
  else:
    for l in range(lmax):
      c = counts[l]
      patch = patches.Rectangle([l,0], 1,c, ec='black', fc=fc)
      ax.add_patch(patch)
      if c>0:
        y = (.75*ymax if c<.5*ymax else .25*ymax)
        ax.text (l+.5, y, c, va='center',ha='center')
    ax.set_xticks([l+.5 for l in range(lmax)])
    ax.set_xticklabels([bstr(l,nmax) for l in range(lmax)])  
    ax.tick_params(axis='x', labelrotation=textrot)
    ax.set_xlim(-.5, lmax+.5)
    ax.set_ylim(0, ymax*1.02)
  return fig,ax

def circuitSize (cir, scaleFactor=0.4):
  '''
  Given a qiskit.Circuit, return a tuple (width,height)
  representing the "standard size" of that circuit in inches 
  '''
  fig,ax=plt.subplots(1,1,figsize=(1,1))   # temporary
  cir.draw('mpl',ax=ax,fold=-1)
  bbox = ax.get_tightbbox(fig.canvas.get_renderer()); aspect = bbox.height/bbox.width
  yrange = np.ptp(ax.get_ylim()); h = yrange*scaleFactor; w=h/aspect
  plt.close()
  return (w,h)

def subplotRow (widths=[1,2,3,2], heights=[2,2,2,2]):
  '''
  Make a row of subplots with the given widths and heights (in inches)
  '''
  assert len(widths)==len(heights)
  numPanels = len(widths)
  width = np.sum (widths)
  height = np.max (heights)
  fig,axs=plt.subplots(1,numPanels,figsize=(width,height))
  xcur = 0
  for n in range(numPanels):
    axs[n].set_position ([xcur, .5 - heights[n]/height/2, widths[n]/width, heights[n]/height])
    xcur += widths[n]/width
  return fig,axs

def draw (cir, **kwargs):
  #print (circuitSize(cir))
  fig,ax=plt.subplots(figsize=circuitSize(cir))
  cir.draw('mpl',**kwargs,**cirOpts,ax=ax)

def initRegister (cir, register, value):
  '''
  Add instructions to circuit <cir> to XOR register <register> with an integer <value>
  '''
  if isinstance (register, qiskit.QuantumRegister): imax = register.size
  else: imax = len(register)
  for i in range(imax):
    if (value>>i)&1:
      cir.x(register[i])

sim = qiskit_aer.AerSimulator()                         # run this line if you want to use ideal simulator
#sim = qiskit_aer.AerSimulator.from_backend(vigoBackend) # uncomment and run if you want to use a noisy simulator based on IBM's Vigo machine



def axgrid (widths=4, heights=2, ha=.5, va=.5, bottomtotop=False, labels=None, removeticks=True, padl=0, padt=0):
  '''
  Make a Figure and an array of Axes, arranged in a grid layout.
  
  Examples:
  
  >>> axgrid (3,1)                      # One plot of size 3x1
  >>> axgrid ([1,4,2,3], [1])           # One row of plots, all of height 1
  >>> axgrid (6, [.2, .4, .2])          # One column of plots
  >>> axgrid ([.2,3,3], [.2,.4,.4,.4])  # Grid with unequal widths and heights
  >>> axgrid ([.2,3,3], [.2,.4,.4,.4], bottomtotop=True) # Reverse vertical order of plots
  
  If *widths* and *heights* are both 2D arrays, some of the plots may be smaller than the allotted grid cell.
  In this case, *ha* and *va* determine horizontal alignment and vertical alignment.  For example:
  
  >>> axgrid ([[2,2,3],[2,3,2]], [[1,1,1],[2,1,2]], ha='left', va='top', labels='auto')
  
  Rows are usually in top-to-bottom order.  This may be reversed using the *bottomtotop* argument:
  
  >>> axgrid ([[2,2,3],[2,3,2]], [[1,1,1],[2,1,2]], ha='right', va='center', labels='auto', bottomtotop=True)
  
  In order to address the Figure and Axes objects, one should save the return values:
  
  >>> fig,axs = axgrid ([[2,2,3],[2,3,2]], [[1,1,1],[2,1,2]], removeticks=False)
  >>> ax = axs[0,0]; ax.plot ([1,2],[1,2])
  >>> ax = axs[1,2]; ax.plot ([1,2],[1,2]);
  
  The Axes in row i and column j is axs[i,j].  It has size width[i,j] x height[i,j].
  These conventions are consistent with matrix indexing conventions (and plt.subplots and numpy.ndarray)
  Generally, where indices are concerned, row indices are quoted before column indices.
  However, where physical dimensions are concerned, widths are quoted before heights,
  according to the conventional ordering of Cartesian coordinates (x,y) (and plt.plot).
  
  Parameters
  ----------
  widths, heights : scalar, 1D, or 2D array-like
        
  Returns
  ----------------
  fig, axs : Figure object and numpy.ndarray of Axes objects
  
  Other Parameters
  ----------------
  ha : 'left', 'center', 'right', float between 0 and 1; or 1D or 2D array of such specifications
  va : 'top', 'center', 'bottom', float between 0 and 1; or 1D 2D array of such specifications
  bottomtotop : False (default) or True
  labels : 
    None                  do not draw labels
    'auto'                label each Axes as axs[rowNumber,columnNumber]
    2D array of strings   custom labels to draw in the center of each Axes
  removeticks :
    True                  set each Axes to show only the frame (and no ticks)
    False                 leave Axes tick marks intact
  removeframe : 
    TBD
  '''
  #======== Determine number of grid cells
  wij = np.array (widths) 
  hij = np.array (heights)
  if wij.ndim==0: wij = np.array([wij])
  if hij.ndim==0: hij = np.array([hij])
  jmax = wij.shape[-1]
  imax = hij.shape[0]
  if wij.ndim==1: wij = np.tile (wij, (imax,1))                # Extend 1D to 2D
  if hij.ndim==0: hij = np.tile (hij, (imax,jmax))             # Extend 0D to 2D
  if hij.ndim==1: hij = np.tile (np.array([hij]).T, (1,jmax))  # Extend 1D to 2D
  assert hij.shape == wij.shape,'ERROR: axgrid was supplied with incompatible widths and heights!'
  if not bottomtotop:
    wij = np.flipud (wij)
    hij = np.flipud (hij)
  #======== Deal with padding
  plij = np.array (padl) # padding left
  ptij = np.array (padt)    # padding top
  if plij.ndim==0: plij = np.tile (plij, (imax,jmax))
  if ptij.ndim==0: ptij = np.tile (ptij, (imax,jmax))
  #======== Determine dimensions of grid cells
  wj = np.max (wij + plij, axis=0)
  hi = np.max (hij + ptij, axis=1)
  w = np.sum (wj)
  h = np.sum (hi)
  xj = np.concatenate ([[0], np.cumsum (wj)])
  yi = np.concatenate ([[0], np.cumsum (hi)])
  uij = np.array(ha)   # Array of horizontal alignment pars
  vij = np.array(va)   # Array of vertical alignment pars
  if uij.ndim==0: uij = np.tile (uij, (imax,jmax))
  if uij.ndim==1: uij = np.tile (uij, (imax,1))
  if vij.ndim==0: vij = np.tile (vij, (imax,jmax))
  if vij.ndim==1: vij = np.tile (np.array([vij]).T, (1,jmax))
  for i in range(imax):
    for j in range(jmax):
      if isinstance(uij[i,j],str): uij[i,j] = {'left':0, 'center':0.5, 'right':1}.get(uij[i,j])
      if isinstance(vij[i,j],str): vij[i,j] = {'top':0, 'center':0.5, 'bottom':1}.get(vij[i,j])
  uij = uij.astype (np.float64)
  vij = vij.astype (np.float64)
  #======== Create Axes
  fig,axs = plt.subplots (imax, jmax, figsize=(w,h))
  axs = np.array(axs).reshape ((imax,jmax))   # ensure this is always a imax*jmax numpy array of Axes
  for i in range(imax):
    for j in range(jmax):
      i2 = i if bottomtotop else imax-1-i
      x = (xj[j] + uij[i,j]*(wj[j] - wij[i,j])) / w
      y = (yi[i] + vij[i,j]*(hi[i] - hij[i,j])) / h
      axs[i2,j].set_position ([ x, y, wij[i,j]/w, hij[i,j]/h])
  if isinstance(labels,str) and labels=='auto':
    labels = np.array([[f'axs[{i},{j}]\n{wij[i][j]}x{hij[i][j]}' for j in range(jmax)] for i in range(imax)])
  if removeticks:
    for i in range(imax):
      for j in range(jmax):
        axs[i,j].set_xticks ([])
        axs[i,j].set_yticks ([])
  if isinstance(labels,np.ndarray):
    for i in range(imax):
      for j in range(jmax):
        axs[i,j].text (.5, .5, labels[i,j], ha='center', va='center', fontsize=20)
        axs[i,j].set_facecolor ('#FFFFCC')
  return fig,axs

def modifyAxSize (ax, wnew, hnew):
  wfig,hfig = ax.figure.get_size_inches()
  x0,y0,x1,y1 = ax.get_position().extents
  x0 *= wfig; x1 *= wfig; y0 *= hfig; y1 *= hfig
  x1 = x0 + wnew
  ym = (y0+y1)/2; y0=ym-hnew/2; y1=ym+hnew/2
  x0 /= wfig; x1 /= wfig; y0 /= hfig; y1 /= hfig
  ax.set_position([x0,y0,x1-x0,y1-y0])
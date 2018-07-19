import numpy as np
import os
import math
import matplotlib
#matplotlib.use('Agg')


import matplotlib.pyplot as plt
import scipy.interpolate



def calcForm(qq, numAtom, fData):
    finalForm = []
    for tempqq in qq:
        tempForm = 0.00
        tempForm = tempForm + fData[numAtom][1]*math.exp(-fData[numAtom][2]*math.pow((tempqq/4.0/math.pi), 2.0))
        tempForm = tempForm + fData[numAtom][3]*math.exp(-fData[numAtom][4]*math.pow((tempqq/4.0/math.pi), 2.0))
        tempForm = tempForm + fData[numAtom][5]*math.exp(-fData[numAtom][6]*math.pow((tempqq/4.0/math.pi), 2.0))
        tempForm = tempForm + fData[numAtom][7]*math.exp(-fData[numAtom][8]*math.pow((tempqq/4.0/math.pi), 2.0))
        tempForm = tempForm + fData[numAtom][9]*math.exp(-fData[numAtom][10]*math.pow((tempqq/4.0/math.pi), 2.0))
        tempForm = tempForm + fData[numAtom][11]
        finalForm.append(tempForm)
    return finalForm

def calcGr(qq, iQ, rmin, rmax, dr):
    
    rr = np.linspace(rmin, rmax, int(round(((rmax-rmin)/dr+1))))
    gGr = []
    
    from scipy import integrate
    
    for i in range(len(rr)):
        tempY = 2/math.pi * np.array(qq) * np.array(iQ) * np.sin(np.array(qq) * rr[i])
        tempGr = integrate.simps(tempY, qq)
        gGr.append(tempGr)

        #tempGr = integrate.cumtrapz(tempY, qq)
        #gGr.append(tempGr[-1])
    
    return rr, gGr

def calcGrLorch(qq, iQ, rmin, rmax, dr):
    rr = np.linspace(rmin, rmax, int(round(((rmax-rmin)/dr+1))))
    gGr = []
    
    delta_r = math.pi / qq[len(qq)-1]

    from scipy import integrate
    
    for i in range(len(rr)):
        tempY = 2/math.pi * np.array(qq) * np.array(iQ) * np.sin(np.array(qq) * rr[i]) * np.sin(np.array(qq) * delta_r) / (np.array(qq) * delta_r)
        tempGr = integrate.simps(tempY, qq)
        gGr.append(tempGr)

        #tempGr = integrate.cumtrapz(tempY, qq)
        #gGr.append(tempGr[-1])
    
    return rr, gGr

def calcNeutronB(qq, bList):
    tempb = 0*np.array(qq) + 1.0
    bqlist = []

    for i in range(len(bList)):
        tempbq = bList[i] * tempb
        bqlist.append(tempbq)

    return bqlist

# Calculating simulated T(r) pattern
# Noting that exported T(r) is calculated using Lorch
def calcPairFunction(qmin, qmax, dq, numAtomList, atomConc, bList, centerAtom, targetAtom, numCN, sigma, bondLength, delta=2, lorch=1):
    normConC = np.array(atomConc)/np.sum(atomConc)
    qq = np.linspace(qmin, qmax, int(round((qmax-qmin)/dq+1)))

    
    fList = np.array(calcNeutronB(qq, bList))
    


    ff2 = []
    tes = (np.array(fList).T * normConC.T)
    for i in range(len(tes)):
        ff2.append(np.sum(tes[i]) * np.sum(tes[i]))

    if lorch==1:
        iQ_PairFunction = delta * (normConC[centerAtom] * numCN * np.array(fList[centerAtom]) * np.array(fList[targetAtom]) / np.array(ff2)) * (np.exp(-0.5 * sigma * sigma * qq * qq)) * np.sin(bondLength * qq) / bondLength / qq
        rr, gGr = calcGrLorch(qq, iQ_PairFunction, 0.01, 10, 0.01)
    else:
        iQ_PairFunction = delta * (normConC[centerAtom] * numCN * np.array(fList[centerAtom]) * np.array(fList[targetAtom]) / np.array(ff2)) * (np.exp(-0.5 * sigma * sigma * qq * qq)) * np.sin(bondLength * qq) / bondLength / qq
        rr, gGr = calcGr(qq, iQ_PairFunction, 0.01, 10, 0.01)


    return qq, iQ_PairFunction, rr, gGr, fList


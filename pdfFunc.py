import numpy as np
import os
import math
import matplotlib
#matplotlib.use('Agg')


import matplotlib.pyplot as plt
from scipy import interpolate


def GetSasakiF1F2(atomList, energyX):
    absL = []
    for i in range(len(atomList)):
        if atomList[i] == 1 or atomList[i] == 2 or atomList[i] ==3:
            tempAbs = [0.0, 0.0]
            absL.append(tempAbs)
        else:
            fname = "./sasakif1f2/f1f2_" + str(atomList[i]) + ".dat"
            ene, vabs = np.loadtxt(fname, delimiter=" ").T
            f1 = interpolate.interp1d(ene, vabs)
            absX = f1(energyX)
            absL.append(absX)
    return absL


def calcAbs(atomList, energyX):
    absL = []
    for i in range(len(atomList)):
        fname = "./abs/absorp_" + str(atomList[i]) + ".dat"
        ene, vabs = np.loadtxt(fname).T
        f1 = interpolate.interp1d(ene, vabs)
        absX = f1(energyX/1000)
        absL.append(absX)
    return absL

def calcFpFpp(atomList, energyX):
    fpfppRaw = []

    for i in range(len(atomList)):
        nameFP = "./afactor/af_" + str(atomList[i]) + ".dat"
        tes = np.loadtxt(nameFP).T
        fpfppRaw.append(tes)

  
    fpList = []
    fppList = []


    for i in range(len(atomList)):
        f1 = interpolate.interp1d(fpfppRaw[i][0], fpfppRaw[i][1])
        f2 = interpolate.interp1d(fpfppRaw[i][0], fpfppRaw[i][2])
        tempfp = f1(energyX)
        tempfpp = f2(energyX)
        fpList.append(tempfp)
        fppList.append(tempfpp)
  
    return np.array(fpList), np.array(fppList)



def calcForm2(qq, numAtomList, fData):
    finalForm = []
    for i in numAtomList:
        tempForm = fData[i-1][1]*np.exp(-fData[i-1][2]*np.power((qq/4.0/np.pi), 2.0))
        tempForm = tempForm + fData[i-1][3]*np.exp(-fData[i-1][4]*np.power((qq/4.0/np.pi), 2.0))
        tempForm = tempForm + fData[i-1][5]*np.exp(-fData[i-1][6]*np.power((qq/4.0/np.pi), 2.0))
        tempForm = tempForm + fData[i-1][7]*np.exp(-fData[i-1][8]*np.power((qq/4.0/np.pi), 2.0))
        tempForm = tempForm + fData[i-1][9]*np.exp(-fData[i-1][10]*np.power((qq/4.0/np.pi), 2.0))
        tempForm = tempForm + fData[i-1][11]
        finalForm.append(tempForm)
    return finalForm



def calcComp(qq, numAtom, cData):
    finalComp = []
    for i in range(len(qq)):
        tempComp = 0.00
        tempComp = tempComp + cData[numAtom][1]*math.exp(-cData[numAtom][2]*math.pow((qq[i]/4.0/math.pi), 2.0))
        tempComp = tempComp + cData[numAtom][3]*math.exp(-cData[numAtom][4]*math.pow((qq[i]/4.0/math.pi), 2.0))
        tempComp = tempComp + cData[numAtom][5]*math.exp(-cData[numAtom][6]*math.pow((qq[i]/4.0/math.pi), 2.0))
        tempComp = tempComp + cData[numAtom][7]

        tempComp2 = cData[numAtom][0] - math.pow(tempComp, 2.0)
        finalComp.append(tempComp2)
    
    return finalComp



def calcFseries(qq, atomList, concList, fdata, fpList, fppList):


    formList = calcForm2(qq, atomList, fdata)
    concList = np.array(concList) / np.sum(concList)

    form2sum = [] # <f2>
    formSum2 = [] # <f>2


    for i in range(len(qq)):
        tempform2sum = 0.00
        for k in range(len(atomList)):
            temp = 0.0+0.0j
            temp= concList[k] * (formList[k][i] + fpList[k] + fppList[k]*1j)**2.0
            tempform2sum += temp.real
    
        form2sum.append(tempform2sum)

    
    
    for i in range(len(qq)):
        tempformsum2 = 0.00
        for k in range(len(atomList)):
            temp = 0.0+0.0j
            temp= concList[k] * (formList[k][i] + fpList[k] + fppList[k]*1j)
            tempformsum2 += temp.real
        tempformsum2 = tempformsum2**2
    
        formSum2.append(tempformsum2)
        
    return form2sum, formSum2
        
def convToQ(tt, energy):
    lamda = 12.3984 / energy
    qq = []
    
    for i in range(len(tt)):
        tempqq = 4.0 * math.pi * math.sin(math.radians(tt[i]/2.0)) / lamda
        qq.append(tempqq)
    
    return qq


            
def corrPol(tt, intensityRaw, pol):
    polList = pol + (1-pol) * np.cos(np.radians(tt)) * np.cos(np.radians(tt))
    intCorr = np.array(intensityRaw) / polList

    return intCorr

def calcCompton(compList, concList):
    compTotal = []
    
    for i in range(len(compList[0])):
        tempSum = 0.0
        for j in range(len(compList)):
            tempSum += compList[j][i] * concList[j]
        compTotal.append(tempSum)
    
    return compTotal



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
        #tempGr = integrate.simps(tempY, qq)
        #gGr.append(tempGr)

        tempGr = integrate.cumtrapz(tempY, qq)
        gGr.append(tempGr[-1])
    
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
def calcPairFunction(qmin, qmax, dq, numAtomList, atomConc, bList, centerAtom, targetAtom, numCN, sigma, bondLength, delta=2, lorch=True):
    normConC = np.array(atomConc)/np.sum(atomConc)
    qq = np.linspace(qmin, qmax, int(round((qmax-qmin)/dq+1)))

    
    fList = np.array(calcNeutronB(qq, bList))
    


    ff2 = []
    tes = (np.array(fList).T * normConC.T)
    for i in range(len(tes)):
        ff2.append(np.sum(tes[i]) * np.sum(tes[i]))

    if lorch==True:
        iQ_PairFunction = delta * (normConC[centerAtom] * numCN * np.array(fList[centerAtom]) * np.array(fList[targetAtom]) / np.array(ff2)) * (np.exp(-0.5 * sigma * sigma * qq * qq)) * np.sin(bondLength * qq) / bondLength / qq
        rr, gGr = calcGrLorch(qq, iQ_PairFunction, 0.01, 10, 0.01)
    else:
        iQ_PairFunction = delta * (normConC[centerAtom] * numCN * np.array(fList[centerAtom]) * np.array(fList[targetAtom]) / np.array(ff2)) * (np.exp(-0.5 * sigma * sigma * qq * qq)) * np.sin(bondLength * qq) / bondLength / qq
        rr, gGr = calcGr(qq, iQ_PairFunction, 0.01, 10, 0.01)


    return qq, iQ_PairFunction, rr, gGr, fList


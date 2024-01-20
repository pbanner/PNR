"""
####################################################################################################

Detector Characterization (DC) Library

This library accompanies the paper "Number State Tomography with a Single SPAD" and contains
the set of functions used in count rate extraction and detector characterization.

Patrick Banner
RbRy Lab, University of Maryland-College Park
July 10, 2023

####################################################################################################
"""

import csv
import numpy as np
from numpy.random import default_rng
import multiprocessing as mp
from scipy import special as sp
from scipy import optimize as opt
from matplotlib import pyplot as plt
import time

"""
####################################################################################################
#
# Second-Order Histogram Generating Functions
#
####################################################################################################
"""

def coincFunc(data, delayBins):
    """
    This function is for calculating full second-order correlation histograms. 
    Inputs:
        data: A NumPy array of values indicating bins in which clicks occurred, e.g.
            np.array([7, 140, 382, 675]) indicates events in those bins. You should
            pre-set the bin size using the data before using this function.
        delayBins: The number of bins corresponding to the delay at which you want
            to calculate correlations, e.g. if your bin size is 1 ns and you want to
            calculate correlations at 17 ns, delayBins = 17.
    Outputs:
        delayBins: same as input (this is useful e.g. for plotting the results from
            many delays).
        coincData: the number of coincidences at the chosen delay.
        
    Notes: 
        • Although a one-line function, this is quite slow for reasonably large data arrays.
        • I typically use this function in parallel for generating correlations vs time,
          e.g. using the multiprocessing.Pool class:
          [pool.apply_async(pnrlib.coincFunc, args = (data, i, )) for i in range(1, maxCorrTime)]
          In this case it can be useful to include a print statement in this function to give
          a sense of progress.
    """
    
    return delayBins, len(np.intersect1d(np.copy(data), np.copy(data)+delayBins))

def createDiffHist(dataToHist, binSize):
    """
    This function creates a first-and-second histogram.
    Inputs:
        dataToHist: A NumPy array of values indicating bins in which clicks occurred, e.g.
            np.array([7, 140, 382, 675]) indicates events in those bins. You should
            pre-set the bin size using the data before using this function.
        binSize: The size of the bins, used only to return a nice "time axis" to accompany
            the histogram results.
    Outputs:
        diffsHist: The first-and-second histogram as a 2D NumPy array, with
            diffsHist[0] = times in sec and diffsHist[1] = number of correlated events.
    """
    
    # This is a first-and-second histogram, so this is easy work!
    # Get the time differences between clicks
    diffsData = dataToHist[1:] - dataToHist[:-1]
    # Sort the differences
    diffsData = diffsData[np.argsort(diffsData)]
    # numpy.unique with return_counts = True returns a 2D array of data where
    # diffsData[0] = time difference and diffsData[1] = number of times that time
    # difference appeared
    diffsData = np.unique(diffsData, return_counts = True)
    # Get the last difference and make a histogram with enough zeros for all the data
    lastClick = int(diffsData[0][-1]+1)
    diffsHist = np.zeros(lastClick)
    # Set the indices of the time differences to the number of events at that time difference
    diffsHist[np.asarray(diffsData[0]-1, dtype=np.int32)] = diffsData[1]
    # Stack a time axis above the coincidences axis
    diffsHist = np.stack((np.linspace(1, lastClick, lastClick)*binSize, diffsHist), axis=0)
    return diffsHist

def createDiffHistN(n, dataToHist, binSize):
    """
    This function creates a first-and-n histogram.
    Inputs:
        n: The "order" of the histogram, e.g. n=4 for a first-and-fourth histogram.
        dataToHist: A NumPy array of values indicating bins in which clicks occurred, e.g.
            np.array([7, 140, 382, 675]) indicates events in those bins. You should
            pre-set the bin size using the data before using this function.
        binSize: The size of the bins, used only to return a nice "time axis" to accompany
            the histogram results.
    Outputs:
        hTot: The first-and-n histogram as a 2D NumPy array, with
            hTot[0] = times in sec and hTot[1] = number of correlated events.
            
    Notes: The general algorithm here is to split the given times into several lists
    whose clicks are "adjacent" in the first-and-n histogram, then use createDiffHist()
    on each individual list, and then add all the histograms up. For instance if the
    events are in bins [1, 47, 89, 225, 612, 998] and n=3, the algorithm splits the
    data list into the "sub-lists" [1, 89, 612] and [47, 225, 998], then calculates the
    first-and-second histograms for each of these and then sums them up.
    """
    
    nEvents = len(dataToHist)
    # The indices for the first sub-list
    inds = np.arange(0, nEvents-(nEvents % (n-1)), n-1)
    lens = np.array([])
    # Since the end is likely not a multiple of n, use a dictionary
    # to accommodate varying lengths
    hists = {}
    # Go through the n-1 sub-lists and make the first-and-second histograms
    for i in range(n-1):
        hists[i] = createDiffHist(dataToHist[inds+i], binSize)
        lens = np.append(lens, len(hists[i][0]))
    # Use the first-and-second histogram lengths to set the size of the
    # final histogram, hTot
    nMax = np.argmax(lens)
    hTot = np.zeros((2, len(hists[nMax][0])))
    # Set the time axis
    hTot[0] = hists[nMax][0]
    # Add up all the sub-lists' histograms together
    for i in range(n-1):
        hTot[1] += np.append(hists[i][1], np.zeros(abs(len(hists[nMax][0]) - len(hists[i][0]))))
    return hTot
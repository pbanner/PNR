"""
####################################################################################################

Photon Number Reconstruction (PNR) Library

This library accompanies the paper "Number State Tomography with a Single SPAD" and contains
the set of functions used in number state reconstruction of light pulses, particularly the
construction of the detector matrix.

Patrick Banner
RbRy Lab, University of Maryland-College Park
January 20, 2024

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
# Recovery effects matrix construction
#
####################################################################################################
"""

def splitStrInc(strToSplit, charToSplit, flag):
    """
    For some of the various string splitting operations we need to do, the
    built-in str.split() method has some behavior we don't want (particularly,
    it sometimes removes the character we want to split by). So this custom
    string splitting method does what we want, splitting strToSplit by charToSplit.
    Inputs:
        strToSplit: The string to split
        charToSplit: The character to split by (this method doesn't work for
            charToSplit having more than one character, but we never use it that way)
        flag: This flag determines the exact splitting behavior. As an example, for
            strToSplit = '33221' and charToSplit = '3', if flag = 0 this function returns
            ['3', '3221'], but if flag = 1, this function returns ['3', '3', '221'].
            That is, flag = 0 means each substring should start with a charToSplit,
            but flag = 1 splits off all instances of charToSplit into their own substrings.
    Outputs:
        res: The NumPy array of substrings.
    """
    
    # res strores the substrings
    # The first character always starts a substring; be sure to initialize
    # the array with the right type or else weird stuff happens
    res = np.array([strToSplit[0]], dtype='<U32')
    # Loop through the rest of the string... note that k does NOT
    #     index the characters starting from 0 for the first character,
    #     i.e. char = strToSplit[k+1]
    for k, char in enumerate(strToSplit[1:]):
        if (char == charToSplit):
            # Start a new substring, always
            res = np.append(res, char)
        else:
            if ((flag == 1) and (strToSplit[k] == charToSplit)):
                # If flag=1, we want to separate the charToSplit's entirely
                # So we check if the previous character was a charToSplit, and if so,
                # start a new substring
                res = np.append(res, char)
            else:
                # If flag=0 we want the only splitting to be with a charToSplit,
                # so since this char is not that, add it to the end of the last
                # (current) substring
                res[-1] = res[-1] + char
    return res

def getStrings(s0, paperStrings = False):
    """
    This function turns what the Supplemental Material calls "strings" into
    what we call "events", that is, maps strings of 1's, 2's, and 3's onto
    physically possible events. (Use 1, 2, and 3 in place of ○, •, and ★
    respectively.)
    Inputs:
        s0: The string of 1's, 2's, and 3's to translate
        [paperStrings]: If True, the returned strings look like those of
            the paper; if False, the returned strings are optimized for
            computation (and for matrix construction, this paramter MUST
            be false). Default False.
    Outputs:
        newStrs: The set of physically possible events corresponding to s0
        
    Note that, to save computing power, we do a little processing that makes
    the returned strings look different from those in the text. Specifically,
    we get rid of all group separators ][ before 3's (as they are redundant),
    and replace the remaining ones with a single-character separator |. Then
    we give new meaning to a | before a 3: |3 means that there was a click
    that occurred before this 3 but there were no photons arriving during that
    click's recovery time, so the event string doesn't "see" it. This happens
    when there is a 2 photon (a twilight count) and no photons arrive during
    that count's recovery time. This helps smooth the way for the integrating
    later in the matrix construction. If you want to see strings like those
    of the paper, run this function with paperStrings = True.
    """
    
    # First, put in the brackets before all 3's
    s1 = (s0.replace('3', '][3') + ']')[1:]
    newStrs = np.array([], dtype='<U32')
    # Now split into groups that all start with [3 and loop through them
    # The general idea is that at each step we're going to generate all the
    # possibilities the group affords using the rules of Fig. S2 of the
    # Supplementary Material, stick all of the possibilities in newStrsTemp,
    # and at the end of each loop instance, combine all of the possibilities
    # in newStrsTemp with all of the strings we already have in newStrs, then
    # assign the list of all of the combinations to newStrs.
    # This split is Rule 1 of A.III.b. of the Supplemental Material.
    for i, s1elem in enumerate(splitStrInc(s1, '[', 0)):
        newStrsTemp = np.array([], dtype='<U32')
        if (s1elem == '[3]'):
            # If the substring is just '3', just add it to every string already
            # in the list; no new possibilities
            newStrsTemp = np.array(['[3]'])
        else:
            # The substring is a 3 followed by some 1s and 2s
            # Notice that s2 gets rid of the first 3 in the string
            s2 = splitStrInc(s1elem[2:-1], '2', 0)
            newStrsTemp = np.array(['[3'], dtype='<U32')
            # It's possible the first substring is a string of 1s...
            # This provides no ambiguity, so just add it to the new strings and move on
            if (s2[0].startswith('1')):
                newStrsTemp[0] = newStrsTemp[0] + s2[0]
                s2 = s2[1:]
            # Now implement Rule 2 of A.III.b. of the Supplemental Material.
            strsToAdd = np.array([], dtype='<U32')
            for j, s2elem in enumerate(s2):
                strsToAdd = np.array([s2elem[0:k+1] + '][' + s2elem[k+1:len(s2elem)] for k in range(len(s2elem)-1)])
                if (j+1 != len(s2)):
                    # There's more groups coming, so also add a string with ][ on the end
                    strsToAdd = np.append(strsToAdd, s2elem + '][')
                strsToAdd = np.append(strsToAdd, s2elem)
                newStrsTemp = np.array([np.array([si+sj for sj in strsToAdd]) for si in newStrsTemp]).flatten()
            # Stick the final ] on the end
            newStrsTemp = np.array([si+']' for si in newStrsTemp])
                
        if (len(newStrs) == 0):
            newStrs = newStrsTemp
        else:
            # Combine the new strings with all the previous new strings
            newStrs = np.array([np.array([si+sj for sj in newStrsTemp]) for si in newStrs]).flatten()
    
    if not paperStrings:
        # Now we're going to do a little post-processing to smooth things along
        # for the integral step. Specifically, we note that ][ before a 3 is redundant:
        # we know 3s always start their own group. So we're going to remove those, and
        # replace the remaining ][ with a single character, |. Then, we're going to
        # let |3 take on a new meaning: a 3 with a | preceding it means there is a click
        # that has no representation in the event string. This happens when a twilight
        # count (type 2 photon) occurs and then no photons arrive within that click's
        # recovery time. This can only happen prior to a 3-type event, because 1s and 2s
        # can come in that click's recovery time. (See, for instance, the last two equations
        # among the integrals at the end of the Supplemental Material section A.III.c.)
        # Thus we'll smooth the process of determining integral bounds later by doing this
        # now. Also, we'll get rid of the first 3 in every string as it's redundant.
        for nsi in range(len(newStrs)):
            newStrs[nsi] = newStrs[nsi][1:-1].replace('][', '|').replace('|3', '3')
            # Starting from the last 3...
            ind = newStrs[nsi].rfind('3')
            while True:
                # ... find the next 3 or | ...
                inda = newStrs[nsi].rfind('|', 0, ind)
                indb = newStrs[nsi].rfind('3', 0, ind)
                ind2 = max(inda, indb)
                if (ind2 == -1):
                    break
                # ... and if there's a 2 in between, turn '3' into '|3' as
                # described above...
                if (newStrs[nsi].find('2', ind2, ind) != -1):
                    newStrs[nsi] = newStrs[nsi][:ind] + '|' + newStrs[nsi][ind:]
                # and loop through all 3's until you hit the end
                ind = indb
            # Get rid of the first 3 in all strings
            newStrs[nsi] = newStrs[nsi][1:]
            
    return newStrs

def getNums(s):
    """
    This function interprets event strings as given by getStrings() (when paperStrings
    is False) and returns the number of photons and the number of clicks in the event.
    Inputs:
        s: the event string to interpret.
    Outputs:
        nPh: the number of photons in the event.
        nCl: the number of clicks in the event.
    """
    sArr = s.split('|')
    # sRec below exploits the fact that str.split removes all instances of the separator, 
    # soby joining we're just left with the numbers in the string
    sRec = ''.join(sArr)
    # The first 3 in the event is dropped, but it's both a photon and a click, so
    # add 1 to the formulas below
    nPh = 1 + len(sRec)
    # The number of 3's plus the number of groups that have at least one 2 in them
    # (remember more than one 2 in the same group only contributes one click)
    nCl = 1 + len(sRec.split('3'))-1 + len([i for i in sArr if (i.find('2') != -1)])
    return nPh, nCl

def constructRTmatrix(size, order, tD, bw, normedSignalData, window_width, backRefKeepOrder = 0, verbose = True, newStrArr=np.array([])):
    """
    This function constructs the recovery time effects matrix R.
    It's a big function. If you need to figure out how it works, I suggest you first get up,
    get a coffee/energy drink or take a walk or listen to music or whatever relaxes you.
    
    Inputs:
        size: the size of the matrix to construct. Unlike the other matrix construction functions,
            I explicitly used np.zeros((size, size)), so size here is equivalent to n_max + 1.
        order: the order o_R of corrections to keep. All event strings with more than order 1's and 2's
            total are discarded.
        tD: a 2-tuple [dead time (sec), recovery time (sec)]
        bw: the bin width (sec)
        normedSignalData: the photon profile gamma(t) to integrate with. I typically use the histogram
            of experimental runs where only one click occurred divided by the total number of experimental
            runs.
        window_width: the width of the data collection window (sec)
        [backRefKeepOrder]: In doing the integrals, this function keeps partial integrals of previous event
            strings, to use as a starting point for future integrals. This parameter determines the events
            to keep by specifying that the number of photons in the event must be less than or equal to this
            parameter to keep it. Default is zero, which means "keep all events." The reason you might not
            want to keep all events is that this feature is memory-intensive, particularly for high
            orders or large data collection windows. 
        [verbose]: Prints out extra diagnostic messages if True. Can be helpful if the calculation is long
            and you're impatient like me. Default True.
        [newStrArr]: Pass in a list of event strings in advance, and this method will use that array instead
            of computing it. If np.array([]), will compute the array. Defaults to np.array([]). This is
            mostly useful when you're running parallelized uncertainty calculation; the methods below
            pre-compute the event string array so they can save time by doing it once and then passing it in.
            
    The function implements the discussion of Sec. A.III. of the Supplemental Material.
    """
    
    # Some setup steps
    if verbose: print("Starting the dead time matrix construction.", end=" ")
    n0 = len(normedSignalData)
    if (backRefKeepOrder == 0):
        backRefKeepOrder = size
    
    # Here we construct the quantum efficiency profile D(tau)
    n1s = int(np.floor(tD[0]/bw))
    nRec = int(np.ceil(tD[1]/bw))
    cPDdata = np.append(np.append(np.ones(n1s), np.array([1-(n-n1s+1)/(nRec-n1s+1) for n in np.arange(n1s,nRec)])), np.zeros(n0))[0:n0]
    
    ### Step one: compute normalizations
    # These are Eqn. S9 of the Supplementary Material, done in nested fashion by
    # building on the previous integral to save time
    if verbose: print("Computing normalization factors...", end=" ")
    norms = np.zeros(size+1)
    nArrTemp = normedSignalData
    norms[1] = np.sum(normedSignalData)
    for ind in range(2,size):
        nArrTemp = normedSignalData*np.flip(np.cumsum(np.flip(nArrTemp)))
        norms[ind] = np.sum(nArrTemp)
        
    ### Step two: generate strings
    # Only do so if newStrArr isn't already built
    if (len(newStrArr) == 0):
        # Here we first build all possible combinations of 1's, 2's, and 3's given
        # the size and order parameter
        # I found this manual for loop to be better than e.g. itertools
        if verbose: print("Generating strings...", end=" ")
        # Seed the string array with the initial cases
        # Remember that all possible events begin with a 3-type photon
        strArr = np.array(['31', '32', '33'])

        newStrs = np.array([])
        for i in range(2, size-1):
            # Find the strings where there are already enough 1s and 2s for the order parameter (inds12)
            # For strings where there are enough, we can only add 3s; for strings where we haven't
            #     hit the limit (inds3), we can add 1, 2, or 3
            inds12 = np.where(np.array([s.count('1') + s.count('2') for s in strArr]) >= order)[0]
            inds3 = np.setdiff1d(np.arange(len(strArr)), inds12)
            newStrs = np.concatenate(([s + '3' for s in strArr[inds12]],
                            [s + '1' for s in strArr[inds3]], [s + '2' for s in strArr[inds3]], [s + '3' for s in strArr[inds3]]
                           ))
            # After each step, take care to produce an array that's sorted by length,
            # then sorted by 1,2,3, AND contains only unique strings
            # This helps later when we generate integrals based on previous strings
            strArr = np.append(strArr, np.sort(newStrs))
            indexes = np.unique(strArr, return_index=True)[1]
            strArr = np.array([strArr[index] for index in sorted(indexes)])

        # Now get the event strings using getStrings()
        t0 = time.time()
        newStrArr = np.array([])
        for i, s in enumerate(strArr):
            newStrArr = np.append(newStrArr, getStrings(s))
            if ((len(strArr) > 10000) and i%10000 == 0):
                print(i, end=" ")
        if verbose: print("Completed in", time.time()-t0, "seconds.", end=" ")

    ### Step three: perform the integrals and add them to the matrix
    # The way we perform integrals here is by building up the arrays one step at a time.
    # This provides a massive speedup over nested loops, but it's nontrivial to explain.
    # I hope to post some documentation in the GitHub to explain it.
    if verbose: print("Beginning integrals. We have", len(newStrArr), "to do...", end=" ")
    t0 = time.time()
    
    # aBase and bBase are 2D arrays representing integrands gamma(t) D(t-t_b) dt for aBase
    # (and bBase replaces D with 1-D); the first index is t_b and the second index is t_b-t
    # which varies from 0 to t_rec
    aBase = np.zeros((n0,nRec))
    bBase = np.zeros((n0,nRec))
    # cBase = normedSignalData
    for i in range(n0):
        aBase[i][:min(nRec,n0-i)] = normedSignalData[i:min(i+nRec,n0)]*(cPDdata[:min(nRec,n0-i)])
        bBase[i][:min(nRec,n0-i)] = normedSignalData[i:min(i+nRec,n0)]*(1-cPDdata[:min(nRec,n0-i)])

    # The sums and sumulative sums here effectively perform integration to the end time
    # so arr1 represents int_{t_a}^{t_b + t_rec} gamma(t) D(t-t_b) dt, with the first index
    # of arr1 being t_b and the second one being t_b - t_a
    # arr3 is a little different since it's only the integral of gamma(t), so there's only one
    # parameter; it represents int_{t_a}^T gamma(t) dt and the index of arr3 is t_a
    # This referencing is important to understand, because it's how we're going to build up the
    # integral in a reasonably time-optimized fashion: we multiply 1D and 2D arrays by aBase and
    # bBase and normedSignalData to create new arrays, and sum appropriately at the very end
    arr1 = np.zeros((n0,nRec))
    arr2 = np.zeros((n0,nRec))
    arr3 = np.zeros(n0)
    for i in range(n0):
        arr1[i] = np.flip(np.cumsum(np.flip(aBase[i])))
        arr2[i] = np.flip(np.cumsum(np.flip(bBase[i])))
        arr3[i] = np.sum(normedSignalData[i+nRec:])
    
    # Dictionary for later use
    startArrs = {'1': arr1, '2': arr2, '3': arr3}
    baseArrs = {'1': aBase, '2': bBase, '3': normedSignalData}

    # Initialize the matrix; recovery time effects have no effect on 0- and 1-photon
    # experimental runs
    mat = np.zeros((size, size))
    mat[0][0] = 1
    mat[1][1] = 1
    
    # Initialize the dictionary and array used to keep track of the event integrals we've
    # already done to see if we can build off of anything
    dataArrs = {}
    backRefStrArr = np.array([], dtype='<U32')

    # For every event string... perform the integration.
    for s0ind, s0 in enumerate(newStrArr):
        # We're going to loop over every character of the event string, in REVERSE order,
        # and arr is going to keep track of the integral as we progress through the string.
        # Kick us off with the last character of the string, and then set to start the loop-
        # through at the penultiamte character.
        arr = startArrs[s0[-1]]
        startInd = len(s0)-2

        # But we have a chance to speed things up if we can find a previous
        # event whose string matches ours and start from that array instead.
        # Successively chop off characters until we find something
        backStr = s0[1:]
        backInds = np.where(backRefStrArr == backStr)[0]
        # Note that we stop when len(backStr) = 1 because we've already
        # covered that by using one of the startArrs.
        while (len(backInds) == 0 and len(backStr) > 1):
            backStr = backStr[1:]
            backInds = np.where(backRefStrArr == backStr)[0]
        if (len(backInds) > 0):
            # We found something!
            arr = dataArrs[newStrArr[backInds[0]]]
            startInd = len(s0)-(len(backStr)+1)

        # This is the big loop through each character.
        for j in range(startInd, -1, -1):
            # At each 1, 2, or 3 we look to the next character (the previous one in the loop
            # but the next one looking at the string left to right) to see if there's a | there
            # to screw with us. So when we encounter one, we'll just skip it.
            # I found this to not noticeably change the time complexity compared to setting
            # some status variable when we found one of these, and using that to determine bounds.
            if (s0[j] == '|'): continue
            elif (s0[j] == '3'):
                # 3's are special because they're a 1D matrix, while the others are 2D matrices
                if (s0[j+1] == '3'):
                    # The next character (the one we just did in the loop) was a 3, so
                    # arr is a 1D matrix already, so we're multiplying 2 1D matrices together.
                    # 3's naturally add a t_rec to the integral bounds because they have to wait
                    # for all previous clicks to finish.
                    arr = np.concatenate((
                            np.array([np.sum(normedSignalData[i+nRec:n0]*arr[i+nRec:n0]) for i in range(n0-nRec)]),
                            np.zeros(nRec)), axis=0)
                else:
                    # The next character (the one we just did in the loop) was a 1 or 2, so
                    # arr is a 2D matrix, so we have to multiply a 1D matrix into a 2D one.
                    # 3's naturally add a t_rec to the integral bounds because they have to wait
                    # for all previous clicks to finish.
                    arr = np.concatenate((
                        np.array([np.sum(normedSignalData[i+nRec:n0]*arr[i+nRec:n0,0]) for i in range(n0-nRec)]),
                        np.zeros(nRec)), axis=0)
            else:
                # We're on a 1 or 2
                # The following if/elif implements Rule 4 of the integral start rules in Sec. A.III.c.
                if (s0[j+1] == '3'):
                    # The next character (the one we just did) was a 3, so arr is 1D, so we
                    # have to multiply a 2D matrix into a 1D matrix.
                    arr = np.concatenate((
                        np.array([np.flip(np.cumsum(np.flip(baseArrs[s0[j]][i][0:nRec]*arr[i]))) for i in range(n0-nRec)]),
                        np.zeros((nRec,nRec))), axis=0)
                elif (s0[j+1:j+3] == '|3'):
                    # The next photon character (the last photon we did) was a 3, so arr is 1D, so we
                    # have to multiply a 2D matrix into a 1D matrix. But there's a |, so per the discussions
                    # in the comments above, the bounds get an extra t_rec to account for the click that
                    # isn't "visible" in the event string.
                    arr = np.concatenate((
                        np.array([np.flip(np.cumsum(np.flip(baseArrs[s0[j]][i][0:nRec]*arr[i+nRec]))) for i in range(n0-nRec)]),
                        np.zeros((nRec,nRec))), axis=0)
                else:
                    # The current photon is a 1 or a 2, and the next photon (the previous one in the loop)
                    # is a 1 or a 2, so multiply 2 2D matrices together.
                    if (s0[j+1] == '|'):
                        # There's a group separator, so set the bounds accordingly
                        # using Rule 3 of the integral start rules in Sec. A.III.c.
                        arr = np.concatenate((
                            np.array([np.flip(np.cumsum(np.flip(baseArrs[s0[j]][i][0:nRec]*arr[i+nRec][0]))) for i in range(n0-nRec)]),
                            np.zeros((nRec,nRec))), axis=0)
                    else:
                        # There's not a group separator, so set the bounds accordingly
                        # using Rule 2 of the integral start rules in Sec. A.III.c.
                        arr = np.array([np.flip(np.cumsum(np.flip(baseArrs[s0[j]][i][0:nRec]*arr[i][0:nRec]))) for i in range(n0)])

        # All done with the integral!
        # What we're left with is an array that's a function of t_1, the
        # time of the first click.
        
        # Get the corresponding photon and click numbers.
        # Also, if it meets the criterion set by backRefKeepOrder,
        # store the array in dataArrs and index it in backRefStrArr
        # so we can use it later.
        nPh, nCl = getNums(s0)
        if (nPh <= backRefKeepOrder):
            dataArrs[s0] = arr
            backRefStrArr = np.append(backRefStrArr, s0)
            
        # There is some filtering for the order previously, but it's incomplete
        # in subtle ways, so this is an extra redundancy for the sake of orderliness
        if (nPh - nCl <= order):
            # Figure out whether we need to sum over a 1D or a 2D array for the
            # t_1 integral, and do the t_1 integral
            if (s0[0] == '3'):
                tot = np.sum(normedSignalData*arr/norms[nPh])
            else:
                tot = np.sum(normedSignalData*arr[:,0]/norms[nPh])
            # Add the integral result to the proper matrix element
            mat[nCl][nPh] = mat[nCl][nPh] + tot
            
        # Update us if there are a lot of integrals and verbose = True
        if (verbose and len(newStrArr) > 1000):
            if ((s0ind+1) % 1000 == 0):
                print(s0ind+1, "completed...", end = " ")
    
    # Set the off-diagonal corresponding to the first uncalculated order
    # so that columns sum to 1, to preserve probability
    for i in range(1, size-(order+1)):
        mat[i][i+order+1] = 1-np.sum(mat[:,i+order+1])
        
    # Phew!
    if verbose: print("Completed in", time.time()-t0, "seconds.")
    return mat

"""
####################################################################################################
#
# Other detector effects matrix constructions
#
####################################################################################################
"""

def getBernoulliMat(eta, nMax):
    """
    This function returns the loss matrix L that accounts for non-unit
    detector efficiency. 
    Inputs:
        eta: the detector efficiency (extrapolated to zero input photon rate)
        nMax: the size of the basis
    Outputs:
        mat: the matrix (NumPy array of size [nMax+1, nMax+1])
        
    The function implements the discussion of Sec. A.I. of the Supplemental Material.
    """
    mat = np.zeros((nMax+1,nMax+1))

    for i in range(nMax+1):
        for j in range(nMax+1):
            if i > j:
                mat[i][j] = 0
            else:
                mat[i][j] = sp.binom(j,i)*(eta**i)*((1-eta)**(j-i))
                
    return mat

def constructBCMatrix(pB, nMax, BCorder = 0, th = 1e-6):
    """
    This function returns the background counts matrix B that accounts
    for background counts.
    Inputs:
        pB: the probability of a background count in the data window
        nMax: the size of the basis
        (BCorder) and (th): ignore; the defaults are good
    Outputs:
        mat: the matrix (NumPy array of size [nMax+1, nMax+1])
        
    The function implements the discussion of Sec. A.II. of the Supplemental Material.
    """
    mat = np.zeros((nMax+1+BCorder,nMax+1))
    
    if (BCorder == -1):
        BC_dist = np.array([pB**(k)*np.exp(-pB)/sp.factorial(k) for k in np.arange(1,nMax*2)])
        BCorder = np.where([BC_dist < th])[1][0]
    
    for i in range(nMax+1+BCorder-1):
        for j in range(nMax+1):
            if i<j:
                mat[i][j] = 0
                continue
            mat[i][j] = pB**(i-j)*np.exp(-pB)/sp.factorial(i-j)

    # Set the last row such that columns sum to 1, to
    # preserve probability
    mat[-1,:] = 1-np.sum(mat, axis=0)
    
    return mat

def getSums(sumGoal, n, nMax):
    """
    This function is used by the afterpulsing matrix construction function
    constructAPmatrix() to get relevant exponents. It numerically finds all
    the lists of integers of length n that exclude all integers strictly
    greater than nMax and that sum to sumGoal.
    Outputs: sums, the list of sums meeting the above criteria
    """
    # Initialize as an array so that vstack is sensible
    sums = np.array([np.zeros(n)])
    if (n == 1):
        # Base case - all lists of length 1 that sum to sumGoal
        sums = np.vstack((sums, np.array([[sumGoal]])))
    elif (n == 2):
        # Base case - all lists of length 2 that sum to sumGoal
        for i in range(sumGoal+1):
            sums = np.vstack((sums, np.array([[i, sumGoal-i]])))
    else:
        # Recursively seek sums of smaller length
        for i in range(sumGoal+1):
            # Get the sums of length n-1 that sum to sumGoal-i
            sums0 = getSums(sumGoal-i, n-1, nMax)
            # Create a list of i's...
            firstRow = np.array([np.ones(len(sums0))*i])
            # Stack it onto the transpose of sums0, then transpose it back
            # These array manipulations produce the lists of integers [i, {sums0}]
            # which is what we want
            sums1 = np.transpose(np.vstack((firstRow, np.transpose(sums0))))
            # Stack it and keep going
            sums = np.vstack((sums, sums1))
    
    # Remove those 0s that we used to initialize, and remove all sums
    # with anything greater than nMax
    sums = sums[1:]
    sums = np.delete(sums, np.where([sums > nMax])[1], axis=0)
    sums = np.array(sums, dtype=np.int32)
    
    return sums

def constructAPMatrix(pA, nMax, APorder = 2):
    """
    This functions constructs the afterpulsing matrix A.
    Inputs:
        pA: the probability of a click in the data window to be an afterpulse.
        nMax: the size of the basis
        (APorder): the number of afterpulses that can occur in any data window.
            Default is 2 (almost always sufficient since the lowest probabilities
            are then on the order pA^2).
    Outputs:
        mat: the afterpulsing matrix (NumPy array of size [nMax+1, nMax+1])
        
    The function implements the discussion of Sec. A.IV. of the Supplemental Material.
    """
    mat = np.zeros((nMax+1, nMax+1))
    mat[0][0] = 1
    
    for i in np.arange(1, nMax+1):
        for j in np.arange(i, min(i+(APorder+1),nMax+1)):
            s = getSums(j-i, i, j)
            mat[j][i] = len(s)*((1-pA)**i)*(pA**(j-i))

    # Set the diagonal corresponding to the first uncalculated order
    # such that the column sums to 1 (in some columns, this means
    # the element in the last row) to preserve probability
    for i in range(1, nMax+1):
        if (min(i+(APorder+1),nMax) != nMax):
            mat[i+(APorder+1)][i] = 1-np.sum(mat[:,i])
        else:
            mat[nMax][i] = 1-np.sum(mat[:-1,i])
    
    return mat

"""
####################################################################################################
#
# Reconstruction functions: Get the detector matrix and perform the EME algorithm
#
####################################################################################################
"""

def getExpMatrix(nMax, eta, pB, pA, tD, bin_width, p1data, window_width, RTorder = 2, RTbrko = 0, RTverbose = True, RTstrArr = np.array([]), APorder = 2):
    """
    This function calculates the detector matrix.
    
    Inputs:
        nMax: The size of the basis (determines the size of the matrix)
        eta: The detector efficiency (when the detector is fully armed)
        pB: The probability of a background count in the data window
        pA: The probability that a click in the data window is an afterpulse
        tD: a 2-tuple [dead time (sec), recovery time (sec)]
        bin_width: the bin width (sec)
        p1data: the photon profile gamma(t) to integrate with. I typically use the histogram
            of experimental runs where only one click occurred divided by the total number of experimental
            runs.
        window_width: the width of the data collection window (sec)
        [RTorder]: the order of recovery time corrections to keep. Default 2.
        [RTbrko]: See the documentation of constructRTmatrix for backRefKeepOrder. Default 0.
        [RTverbose]: See the documentation of constructRTmatrix for verbose. Default True.
        [APorder]: the order of afterpulsing effects to keep. Default 2.
    Outputs:
        The detector effects matrix (NumPy array of size [nMax+1, nMax+1])
    """

    matAP = constructAPMatrix(pA, nMax, APorder=APorder)
    matRT = constructRTmatrix(nMax+1, RTorder, tD, bin_width, p1data, window_width, backRefKeepOrder=RTbrko, verbose=RTverbose, newStrArr=RTstrArr)
    matBC = constructBCMatrix(pB, nMax)
    matL = getBernoulliMat(eta, nMax)
    
    return np.matmul(matAP, np.matmul(matRT, np.matmul(matBC, matL)))

# EME = expectation-maximization entropy
def getInputDist_EME(expDist, matD, l=0.5e-2, iterations=1e10, epsilon=1e-12):
    """
    This function implements the EME algorithm as discussed in the paper.
    Inputs: 
        expDist: the experimental click number distribution (size should match that of matD)
        matD: the detector effects matrix
        [l]: the entropy regularization strength parameter (denoted alpha in the paper);
            default 0.5e-2.
        [iterations]: the maximum number of iterations to perform; default 1e10.
        [epsilon]: the convergence condition parameter; default 1e-12.
    Outputs:
        EME: the reconstructed number state distribution.
    """
    nMax = len(expDist)
    EME = np.zeros(nMax)

    # Initial guess is a uniform distribution
    pn = np.array([1./float(nMax)] * nMax)
    iteration = 0
    while (iteration < iterations):
        # This is the expectation-maximization part...
        EM = np.dot(expDist/np.dot(matD, pn), matD)*pn
        # ... and this is the additional entropy part
        E = l*(np.log(pn) - np.sum(pn*np.log(pn)))*pn
        E[np.isnan(E)] = 0.0
        EME = EM - E

        # Check for convergence... dist = distance
        dist = np.sqrt(np.sum((EME-pn)**2))
        if (dist <= epsilon):
            break
        else:
            pn = EME
            iteration += 1
    
    return EME


"""
####################################################################################################
#
# Reconstruction Evaluation Functions
#
####################################################################################################
"""

def coherentStateFitFunc(n, n0):
    """ Returns the population of a coherent state with average number n0 in Fock state n """
    return np.exp(-n0)*(n0**n)/sp.gamma(n+1)

def getG2(dist):
    """ Returns the g(2)(tau = 0) of a number distribution """
    numer = 0
    denom = 0
    for i, frac in enumerate(dist):
        #print(i, frac)
        if i == 0:
            continue
        numer = numer + i*(i-1)*frac
        denom = denom + i*frac
    denom = denom**2
    
    return numer/denom

def getDelta(expDist, thDist):
    """
    Returns the total variational distance Delta between two distributions.
    Although the arguments are named suggestively, Delta is symmetric in the arguments.
    """
    return np.sum(abs(expDist-thDist))*0.5

def getFidelity(expDist, thDist):
    """
    Returns the overlap (or fidelity) between two distributions.
    Although the arguments are named suggestively, Delta is symmetric in the arguments.
    """
    return np.sum(np.sqrt(expDist*thDist))


"""
####################################################################################################
#
# Getting and simulating uncertainties
#
####################################################################################################
"""

def getCohDistErrors(nBar, u_nBar, n, numExps = 1000):
    """
    This function simulates the uncertainty in each number component
    of a Poisson distribution whose average number has some uncertainty.
    It truly "simulates" because it runs a Monte Carlo simulation.
    Inputs:
        nBar: the average number of the Poisson distribution to simulate
        u_nBar: the uncertainty in nBar
        n: the size of the basis to simulate on
        numExps: number of Monte Carlo sims to run; default 1000
    Outputs:
        dist: the Poisson distribution with average number nBar
        distErrs: the error bars on each component of the distribution
        (i.e. the results are dist[i] ± distErrs[i] for i in [0, n]
    """
    rng = default_rng()
    dist = coherentStateFitFunc(np.array(range(n)), nBar)
    # Normalize in case the basis truncation causes an error
    dist = dist/np.sum(dist)
    
    distData = np.array([])
    # Select some nBars using nBar ± u_nBar and a normal distribution
    rns = rng.normal(nBar, u_nBar, numExps)
    for i in range(numExps):
        distData = np.append(distData, coherentStateFitFunc(np.array(range(n)), rns[i]))
    distData = np.transpose(np.reshape(distData, (-1, n)))
    # Get the uncertainties as the standard deviations of each component
    distErrs = np.std(distData, axis=1)
    
    return dist, distErrs

def getReconDist(expDist, nMax, eta, pB, pA, tD, tRec, bin_width, p1data, window_width, RTorder = 2, RTbrko = 0, RTverbose = True, RTstrArr = np.array([]), APorder = 2, l=0.5e-2, iterations=1e10, epsilon=1e-12):
    """
    This function combines getExpMatrix() and getInputDist_EME() to output a reconstructed distribution
    given all the relevant inputs. The inputs are all the same as the above two functions; see their
    docstrings for details.
    Outputs:
        A reconstructed distribution, as a 1D NumPy array of size nMax+1.
    """
    
    # Get the detector matrix
    matD = getExpMatrix(nMax, eta, pB, pA, np.array([tD, tRec]), bin_width, p1data, window_width, RTorder = RTorder, RTbrko = RTbrko, RTverbose = RTverbose, RTstrArr = RTstrArr, APorder = APorder)
    # Reconstruct the distribution
    recon_eme = getInputDist_EME(expDist, matD, l=l, iterations=iterations, epsilon=epsilon)
    
    return recon_eme

def getReconDistErrorsP(nExp, expDist, nMax, nTot, eta0, pB, pA, tD, tRec, bin_width, p1data, window_width, CPUs = 0, detOnly = 0, RTorder = 2, RTbrko = 0, RTverbose = True, APorder = 2, l=0.5e-2, iterations=1e10, epsilon=1e-12):
    """
    This function performs a Monte Carlo simulation of the uncertainties on the reconstructed distribution
    given uncertainties on all the detector parameters.
    Inputs:
        nExp: The number of Monte Carlo runs ("experiments") to perform
        expDist: the experimental click number distribution
        nMax: the max n of the truncated basis
        nTot: the total number of experimental runs
        eta0: a 2-tuple [detector efficiency, uncertainty on detector efficiency]
        pB: a 2-tuple [probability of a background count in the window, uncertainty in that probability]
        pA: a 2-tuple [probability of an afterpulse, uncertainty in that probability]
        tD: a 2-tuple [dead time, uncertainty in the dead time] (sec)
        tRec: a 2-tuple [recovery time, uncertainty in the recovery time] (sec)
        bin_width: the bin width (sec)
        p1data: the photon profile gamma(t) to integrate with. I typically use the histogram
            of experimental runs where only one click occurred divided by the total number of experimental
            runs.
        window_width: the width of the data collection window (sec)
        [CPUs]: the number of parallel threads to open. If set to zero, uses multiprocessing.cpu_count().
            Default 0.
        [detOnly]: if 1, will simulate errors only due to detector parameter uncertainties; if 0, will also
            approximate sampling error by treating the distribution of each component of the click
            distribution as Poissonian. Default 0.
        [RTorder]: the order of recovery time corrections to keep. Default 2.
        [RTbrko]: See the documentation of constructRTmatrix for backRefKeepOrder. Default 0.
        [RTverbose]: See the documentation of constructRTmatrix for verbose. Default True.
        [APorder]: the order of afterpulsing effects to keep. Default 2.
        [l]: the entropy regularization strength parameter (denoted alpha in the paper);
            default 0.5e-2.
        [iterations]: the maximum number of iterations to perform; default 1e10.
        [epsilon]: the convergence condition parameter; default 1e-12.
    Outputs:
        reconDists: a 2D NumPy array of shape (nExp, nMax+1) containing the reconstructed
            distributions from all of the runs. You can then use np.mean and np.std with
            axis = 0 to obtain statistics.
    
    """
    if (CPUs == 0):
        CPUs = mp.cpu_count()
    
    # Initialize the random number generator (RNG)
    rng = np.random.default_rng()
    
    # Get the number distributions
    nds = np.zeros((nMax+1, nExp))
    if not detOnly:
        for i in range(nMax+1):
            if (expDist[i] != 0):
                # Otherwise leave it zero
                nds[i] = rng.poisson(lam = round(expDist[i]*nTot), size=nExp)
        nds = np.transpose(nds)
        for j in range(nExp):
            nds[j] = nds[j]/np.sum(nds[j])
    
    # Sample from Gaussians with mean and width as given
    etas = rng.normal(eta0[0], eta0[1], nExp)
    pBs = rng.normal(pB[0], pB[1], nExp)
    pAs = rng.normal(pA[0], pA[1], nExp)
    tDs = rng.normal(tD[0], tD[1], nExp)
    tRs = rng.normal(tRec[0], tRec[1], nExp)
    
    # Pre-build the list of event strings for the dead time effects, and
    #     pass it in via the keyword arguments so we only ahve to compute it once
    # This code is the same as in constructRTmatrix(); refer to comments there for details
    print("Getting the dead time correction strings...", end=" ")
    strArr = np.array(['31', '32', '33'])

    newStrs = np.array([])
    for i in range(2, nMax):
        inds12 = np.where(np.array([s.count('1') + s.count('2') for s in strArr]) >= RTorder)[0]
        inds3 = np.setdiff1d(np.arange(len(strArr)), inds12)
        newStrs = np.concatenate(([s + '3' for s in strArr[inds12]],
                        [s + '1' for s in strArr[inds3]], [s + '2' for s in strArr[inds3]], [s + '3' for s in strArr[inds3]]
                       ))
        strArr = np.append(strArr, np.sort(newStrs))
        indexes = np.unique(strArr, return_index=True)[1]
        strArr = np.array([strArr[index] for index in sorted(indexes)])

    print("Got", len(strArr), "strings...", end=" ")
    t0 = time.time()
    newStrArr = np.array([])
    for i, s in enumerate(strArr):
        newStrArr = np.append(newStrArr, getStrings(s))
        if ((len(strArr) > 10000) and i%10000 == 0):
            print(i, end=" ")
    print("Completed in", time.time()-t0, "seconds.")
        
    print("Starting the parallelized computations.")
    
    # Initialize the worker pool and assign the tasks
    pool = mp.Pool(CPUs)
    distResObjs = [pool.apply_async(getReconDist, args = ((1-detOnly)*nds[i] + detOnly*expDist, nMax, etas[i], pBs[i], pAs[i], tDs[i], tRs[i], bin_width, p1data, window_width, ), kwds = {'RTorder': RTorder, 'RTbrko': RTbrko, 'RTverbose': RTverbose, 'RTstrArr': newStrArr, 'APorder': APorder, 'l': l, 'iterations': iterations, 'epsilon': epsilon}) for i in range(nExp)]
    pool.close()
    pool.join()
    
    # Get the results and collect them in the array to return
    reconDists = np.zeros((nExp, nMax+1))
    for i, res in enumerate(distResObjs):
        reconDists[i] = res.get()

    del pool
    print("Done.")
    return reconDists
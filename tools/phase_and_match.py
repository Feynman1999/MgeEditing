import math
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import argparse

def PhaseCongruency3(InputImage, NumberScales, NumberAngles):

    # nscale           4    - Number of wavelet scales, try values 3-6   小波的尺度
    # norient          6    - Number of filter orientations.             filter 方向
    # minWaveLength    3    - Wavelength of smallest scale filter.       最小尺度filter的波长
    # mult             2.1  - Scaling factor between successive filters.
    # sigmaOnf         0.55 - Ratio of the standard deviation of the Gaussian
    #                         describing the log Gabor filter's transfer function
    #                         in the frequency domain to the filter center frequency.
    # k                2.0  - No of standard deviations of the noise energy beyond
    #                         the mean at which we set the noise threshold point.
    #                         You may want to vary this up to a value of 10 or
    #                         20 for noisy images
    # cutOff           0.5  - The fractional measure of frequency spread
    #                         below which phase congruency values get penalized.
    # g                10   - Controls the sharpness of the transition in
    #                         the sigmoid function used to weight phase
    #                         congruency for frequency spread.
    # noiseMethod      -1   - Parameter specifies method used to determine
    #                         noise statistics.
    #                           -1 use median of smallest scale filter responses
    #                           -2 use mode of smallest scale filter responses
    #                            0+ use noiseMethod value as the fixed noise threshold
    minWaveLength = 3
    mult = 2.1
    sigmaOnf = 0.55
    k = 2.0
    cutOff = 0.5
    g = 10
    noiseMethod = -1


    epsilon = .0001 # Used to prevent division by zero.

    # [rows,cols] = size(im);
    # imagefft = fft2(im);              % Fourier transform of image
    #
    # zero = zeros(rows,cols);
    # EO = cell(nscale, norient);       % Array of convolution results.
    # PC = cell(norient,1);
    # covx2 = zero;                     % Matrices for covariance data
    # covy2 = zero;
    # covxy = zero;
    #
    # EnergyV = zeros(rows,cols,3);     % Matrix for accumulating total energy
    #                                   % vector, used for feature orientation
    #                                   % and type calculation
    #
    # pcSum = zeros(rows,cols);

    f_cv = cv2.dft(np.float32(InputImage),flags=cv2.DFT_COMPLEX_OUTPUT) #傅里叶变换

    #------------------------------
    nrows, ncols = InputImage.shape
    zero = np.zeros((nrows,ncols))
    EO = np.zeros((nrows,ncols,NumberScales,NumberAngles),dtype=complex)
    PC = np.zeros((nrows,ncols,NumberAngles))
    covx2 = np.zeros((nrows,ncols))
    covy2 = np.zeros((nrows,ncols))
    covxy = np.zeros((nrows,ncols))
    EnergyV = np.zeros((nrows,ncols,3)) #能量
    pcSum = np.zeros((nrows,ncols))



    # Matrix of radii 半径
    cy = math.floor(nrows/2)
    cx = math.floor(ncols/2)
    y, x = np.mgrid[0:nrows, 0:ncols]
    y = (y-cy)/nrows
    x = (x-cx)/ncols

    # y = y - cy
    # x = x - cx
    radius = np.sqrt(x**2 + y**2)
    radius[cy, cx] = 1

    # Matrix values contain polar angle.
    # (note -ve y is used to give +ve anti-clockwise angles)
    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Initialise set of annular bandpass filters
    #  Here I use the method of scale selection from the code I used to generate
    #   stimuli for my latest experiments (spatial feature scaling):
    #   /Users/carl/Studies/Face_Projects/features_wavelet
    #NumberScales = 3 # should be odd
    annularBandpassFilters = np.empty((nrows,ncols,NumberScales))
    #p = np.arange(NumberScales) - math.floor(NumberScales/2)
    #fSetCpo = CriticalBandCyclesPerObject*mult**p
    #fSetCpi = fSetCpo * ObjectsPerImage

    # Number of filter orientations.
    #NumberAngles = 6
    """ Ratio of angular interval between filter orientations and the standard deviation
        of the angular Gaussian function used to construct filters in the freq. plane.
    """
    dThetaOnSigma = 1.3
    filterOrient = np.arange(start=0, stop=math.pi - math.pi / NumberAngles, step = math.pi / NumberAngles)

    # The standard deviation of the angular Gaussian function used to construct filters in the frequency plane.
    thetaSigma = math.pi / NumberAngles / dThetaOnSigma;

    BandpassFilters = np.empty((nrows,ncols,NumberScales,NumberAngles))
    evenWavelets = np.empty((nrows,ncols,NumberScales,NumberAngles))
    oddWavelets  = np.empty((nrows,ncols,NumberScales,NumberAngles))

    # The following implements the log-gabor transfer function
    """ From http://www.peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html
        The filter bandwidth is set by specifying the ratio of the standard deviation
        of the Gaussian describing the log Gabor filter's transfer function in the
        log-frequency domain to the filter center frequency. This is set by the parameter
        sigmaOnf . The smaller sigmaOnf is the larger the bandwidth of the filter.
        I have not worked out an expression relating sigmaOnf to bandwidth, but
        empirically a sigmaOnf value of 0.75 will result in a filter with a bandwidth
        of approximately 1 octave and a value of 0.55 will result in a bandwidth of
        roughly 2 octaves.
    """
    # sigmaOnf = 0.74  # approximately 1 octave
    # sigmaOnf = 0.55  # approximately 2 octaves
    """ From Wilson, Loffler and Wilkinson (2002 Vision Research):
        The bandpass filtering alluded to above was used because of ubiquitous evidence
        that face discrimination is optimal within a 2.0 octave (at half amplitude)
        bandwidth centered upon 8–13 cycles per face width (Costen et al., 1996;
        Fiorentini et al., 1983; Gold et al., 1999; Hayes et al., 1986; Näsänen, 1999).
        We therefore chose a radially symmetric filter with a peak frequency of 10.0
        cycles per mean face width and a 2.0 octave bandwidth described by a difference
         of Gaussians (DOG):"""

    # Lowpass filter to remove high frequency 'garbage'
    filterorder = 15  # filter 'sharpness'
    cutoff = .45
    normradius = radius / (abs(x).max()*2)
    lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff)**(2*filterorder))
    #
    # Note: lowpassbutterworth is currently DC centered.
    #
    #
    #annularBandpassFilters[:,:,i] = logGabor * lowpassbutterworth
    #logGabor = np.empty((nrows,ncols,NumberScales)) --> same as annularBandpassFilters
    for s in np.arange(NumberScales):
        wavelength = minWaveLength*mult**s
        fo = 1.0/wavelength                  # Centre frequency of filter.
        logGabor = np.exp((-(np.log(radius/fo))**2) / (2 * math.log(sigmaOnf)**2))
        annularBandpassFilters[:,:,s] = logGabor*lowpassbutterworth  # Apply low-pass filter
        annularBandpassFilters[cy,cx,s] = 0          # Set the value at the 0 frequency point of the filter
                                                     # back to zero (undo the radius fudge).
    # main loop
    for o in np.arange(NumberAngles):
        # Construct the angular filter spread function
        angl = o*math.pi/NumberAngles # Filter angle.
        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        # ds = sintheta * cos(angl) - costheta * sin(angl);    % Difference in sine.
        # dc = costheta * cos(angl) + sintheta * sin(angl);    % Difference in cosine.
        # dtheta = abs(atan2(ds,dc));                          % Absolute angular distance.

        # % Scale theta so that cosine spread function has the right wavelength and clamp to pi
        # dtheta = min(dtheta*norient/2,pi);
        # % The spread function is cos(dtheta) between -pi and pi.  We add 1,
        # % and then divide by 2 so that the value ranges 0-1
        # spread = (cos(dtheta)+1)/2;
        #
        # sumE_ThisOrient   = zero;          % Initialize accumulator matrices.
        # sumO_ThisOrient   = zero;
        # sumAn_ThisOrient  = zero;
        # Energy            = zero;
        #angl = filterOrient[o]
        """ For each point in the filter matrix calculate the angular distance from the
            specified filter orientation.  To overcome the angular wrap-around problem
            sine difference and cosine difference values are first computed and then
            the atan2 function is used to determine angular distance.
        """
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)      # Difference in sine.
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)      # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds,dc))                              # Absolute angular distance.

        # Scale theta so that cosine spread function has the right wavelength
        #   and clamp to pi
        dtheta = np.minimum(dtheta*NumberAngles/2, math.pi)

        #spread = np.exp((-dtheta**2) / (2 * thetaSigma**2));  # Calculate the angular
                                                              # filter component.
        # The spread function is cos(dtheta) between -pi and pi.  We add 1,
        #   and then divide by 2 so that the value ranges 0-1
        spread = (np.cos(dtheta)+1)/2

        sumE_ThisOrient   = np.zeros((nrows,ncols))  # Initialize accumulator matrices.
        sumO_ThisOrient   = np.zeros((nrows,ncols))
        sumAn_ThisOrient  = np.zeros((nrows,ncols))
        Energy            = np.zeros((nrows,ncols))

        maxAn = []
        for s in np.arange(NumberScales):
            filter = annularBandpassFilters[:,:,s] * spread # Multiply radial and angular
                                                            # components to get the filter.

            criticalfiltershift = np.fft.ifftshift( filter )
            criticalfiltershift_cv = np.empty((nrows, ncols, 2))
            for ip in range(2):
                criticalfiltershift_cv[:,:,ip] = criticalfiltershift

            # Convolve image with even and odd filters returning the result in EO
            MatrixEO = cv2.idft( criticalfiltershift_cv * f_cv )
            EO[:,:,s,o] = MatrixEO[:,:,1] + 1j*MatrixEO[:,:,0]

            An = cv2.magnitude(MatrixEO[:,:,0], MatrixEO[:,:,1])    # Amplitude of even & odd filter response.

            sumAn_ThisOrient = sumAn_ThisOrient + An             # Sum of amplitude responses.
            sumE_ThisOrient = sumE_ThisOrient + MatrixEO[:,:,1] # Sum of even filter convolution results.
            sumO_ThisOrient = sumO_ThisOrient + MatrixEO[:,:,0] # Sum of odd filter convolution results.

            # At the smallest scale estimate noise characteristics from the
            # distribution of the filter amplitude responses stored in sumAn.
            # tau is the Rayleigh parameter that is used to describe the
            # distribution.
            if s == 0:
            #     if noiseMethod == -1     # Use median to estimate noise statistics
                tau = np.median(sumAn_ThisOrient) / math.sqrt(math.log(4))
            #     elseif noiseMethod == -2 # Use mode to estimate noise statistics
            #         tau = rayleighmode(sumAn_ThisOrient(:));
            #     end
                maxAn = An
            else:
                # Record maximum amplitude of components across scales.  This is needed
                # to determine the frequency spread weighting.
                maxAn = np.maximum(maxAn,An)
            # end

        # complete scale loop
        # next section within mother (orientation) loop
        #
        # Accumulate total 3D energy vector data, this will be used to
        # determine overall feature orientation and feature phase/type
        EnergyV[:,:,0] = EnergyV[:,:,0] + sumE_ThisOrient
        EnergyV[:,:,1] = EnergyV[:,:,1] + math.cos(angl)*sumO_ThisOrient
        EnergyV[:,:,2] = EnergyV[:,:,2] + math.sin(angl)*sumO_ThisOrient

        # Get weighted mean filter response vector, this gives the weighted mean
        # phase angle.
        XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
        # using dot and cross products between the weighted mean filter response
        # vector and the individual filter response vectors at each scale.  This
        # quantity is phase congruency multiplied by An, which we call energy.

        for s in np.arange(NumberScales):
            # Extract even and odd convolution results.
            E = EO[:,:,s,o].real
            O = EO[:,:,s,o].imag

            Energy = Energy + E*MeanE + O*MeanO - np.abs(E*MeanO - O*MeanE)

        ## Automatically determine noise threshold
        #
        # Assuming the noise is Gaussian the response of the filters to noise will
        # form Rayleigh distribution.  We use the filter responses at the smallest
        # scale as a guide to the underlying noise level because the smallest scale
        # filters spend most of their time responding to noise, and only
        # occasionally responding to features. Either the median, or the mode, of
        # the distribution of filter responses can be used as a robust statistic to
        # estimate the distribution mean and standard deviation as these are related
        # to the median or mode by fixed constants.  The response of the larger
        # scale filters to noise can then be estimated from the smallest scale
        # filter response according to their relative bandwidths.
        #
        # This code assumes that the expected reponse to noise on the phase congruency
        # calculation is simply the sum of the expected noise responses of each of
        # the filters.  This is a simplistic overestimate, however these two
        # quantities should be related by some constant that will depend on the
        # filter bank being used.  Appropriate tuning of the parameter 'k' will
        # allow you to produce the desired output.

        # if noiseMethod >= 0:     % We are using a fixed noise threshold
        #     T = noiseMethod;    % use supplied noiseMethod value as the threshold
        # else:
        # Estimate the effect of noise on the sum of the filter responses as
        # the sum of estimated individual responses (this is a simplistic
        # overestimate). As the estimated noise response at succesive scales
        # is scaled inversely proportional to bandwidth we have a simple
        # geometric sum.
        totalTau = tau * (1 - (1/mult)**NumberScales)/(1-(1/mult))

        # Calculate mean and std dev from tau using fixed relationship
        # between these parameters and tau. See
        # http://mathworld.wolfram.com/RayleighDistribution.html
        EstNoiseEnergyMean = totalTau*math.sqrt(math.pi/2)        # Expected mean and std
        EstNoiseEnergySigma = totalTau*math.sqrt((4-math.pi)/2)   # values of noise energy

        T =  EstNoiseEnergyMean + k*EstNoiseEnergySigma # Noise threshold
        # end

        # Apply noise threshold,  this is effectively wavelet denoising via
        # soft thresholding.
        Energy = np.maximum(Energy - T, 0)

        # Form weighting that penalizes frequency distributions that are
        # particularly narrow.  Calculate fractional 'width' of the frequencies
        # present by taking the sum of the filter response amplitudes and dividing
        # by the maximum amplitude at each point on the image.   If
        # there is only one non-zero component width takes on a value of 0, if
        # all components are equal width is 1.
        width = (sumAn_ThisOrient/(maxAn + epsilon) - 1) / (NumberScales-1)

        # Now calculate the sigmoidal weighting function for this orientation.
        weight = 1.0 / (1 + np.exp( (cutOff - width)*g))

        # Apply weighting to energy and then calculate phase congruency
        PC[:,:,o] = weight*Energy/sumAn_ThisOrient   # Phase congruency for this orientatio

        pcSum = pcSum + PC[:,:,o]

        # Build up covariance data for every point
        covx = PC[:,:,o]*math.cos(angl)
        covy = PC[:,:,o]*math.sin(angl)
        covx2 = covx2 + covx**2
        covy2 = covy2 + covy**2
        covxy = covxy + covx*covy
        # above everyting within orientaiton loop
    # ------------------------------------------------------------------------
    # current work
    # Edge and Corner calculations
    # The following is optimised code to calculate principal vector
    # of the phase congruency covariance data and to calculate
    # the minimumum and maximum moments - these correspond to
    # the singular values.

    # # First normalise covariance values by the number of orientations/2
    covx2 = covx2/(NumberAngles/2)
    covy2 = covy2/(NumberAngles/2)
    covxy = 4*covxy/NumberAngles   # This gives us 2*covxy/(norient/2)
    denom = np.sqrt(covxy**2 + (covx2-covy2)**2)+epsilon
    M = (covy2+covx2 + denom)/2          # Maximum moment
    m = (covy2+covx2 - denom)/2          # ... and minimum moment
    #
    # # Orientation and feature phase/type computation
    # ORM = np.arctan2(EnergyV[:,:,2], EnergyV[:,:,1])
    # ORM[ORM<0] = ORM[ORM<0]+math.pi       # Wrap angles -pi..0 to 0..pi
    # ORM = np.round(ORM*180/math.pi)        # Orientation in degrees between 0 and 180
    #
    # OddV = np.sqrt(EnergyV[:,:,1]**2 + EnergyV[:,:,2]**2)
    # featType = np.arctan2(EnergyV[:,:,0], OddV)  # Feature phase  pi/2 <-> white line,
                                            # 0 <-> step, -pi/2 <-> black line
    # ------------------------------------------------------------------------

    #return M, m, ORM, EO, T, annularBandpassFilters, lowpassbutterworth
    return M, m

def templateMatching(src, template, method):
    result = cv2.matchTemplate(src, template, method)  # (289, 289)
    # print(result)
    
    # print(result)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # 查找数组中最小和最大元素值及其位置
    location = [0, 0]
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  # 平方差匹配 值越小越好
        location = min_loc
    else:
        location = max_loc
    return location

def grayhist(img):
    median=np.mean(img)
    # print(median)
    img[img>(median+0.01)]=1
    img[img<=(median+0.01)]=0
    return img

tong =[]

if __name__ == '__main__':
    file_sar_path= '/opt/data/private/datasets/stage1/test2/sar'
    file_opt_path= '/opt/data/private/datasets/stage1/test2/optical'
    test_result_path = './tools/result1.txt'
    margin = 1
    f = open(test_result_path, 'r')
    f1 = open("./result.txt", 'w+')

    for i in range(2*margin+1):
        tong.append([0]*(2*margin+1))

    test_results = f.readlines()

    for i in range(0,len(test_results)):
        if test_results[i].strip() == "":
            continue

        if i==0:
            f1.write(test_results[i])
        else:
            res = test_results[i].split(" ")
            assert len(res) == 5
            full_sar_path = os.path.join(file_sar_path, res[2])
            full_opt_path = os.path.join(file_opt_path, res[1])
            
            sar_img = cv2.imread(full_sar_path,0)
            M1_sar,_ = PhaseCongruency3(sar_img, 4, 6)
            M1_sar = grayhist(M1_sar)
            M1_sar = M1_sar.astype(np.uint8)*255  # unit8的范围是0~255 为什么非要这样？？？

            opt_img = cv2.imread(full_opt_path,0)
            M1_opt,_ = PhaseCongruency3(opt_img, 4, 6)
            M1_opt = grayhist(M1_opt)
            M1_opt = M1_opt.astype(np.uint8)*255
            gt_w = int(res[3])
            gt_h = int(res[4])
            gt_w = max(margin, gt_w)
            gt_h = max(margin, gt_h)
            gt_w = min(288-margin, gt_w)
            gt_h = min(288-margin, gt_h)
            # print(M1_opt.shape, M1_sar.shape)
            locations = templateMatching(M1_opt[gt_h-margin:gt_h+margin+512, gt_w-margin:gt_w+margin+512], M1_sar, cv2.TM_CCORR_NORMED)
            
            loc_x=int(locations[0])
            loc_y=int(locations[1])
            tong[loc_x][loc_y] += 1


            ans = " ".join(res[0:3])
            ans += " "
            ans += str(gt_w+loc_x-margin)
            ans += " "
            ans += str(gt_h+loc_y-margin)
            ans += "\n"
            print(ans)
            f1.write(ans)

        if i%10 ==0:
            print(tong)

    f1.close()
    f.close()

import copy
import math

import numpy as np
import openturns as ot
import scipy as sp
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.stats import norm



#Function to calculate a safety factor:
def calc_gamma(mechanism,TrajectInfo):
    if mechanism == 'Piping' or mechanism == 'Heave' or mechanism == 'Uplift':
        Pcs = (TrajectInfo['Pmax'] * TrajectInfo['omegaPiping'] * TrajectInfo['bPiping']) /( TrajectInfo['aPiping'] * TrajectInfo['TrajectLength'])
        betacs = pf_to_beta(Pcs)
        betamax = pf_to_beta(TrajectInfo['Pmax'])
        if mechanism == 'Piping':
            gamma = 1.04*np.exp(0.37*betacs-0.43*betamax)
        elif mechanism == 'Heave':
            gamma = 0.37*np.exp(0.48*betacs-0.3*betamax)
        elif mechanism == 'Uplift':
            gamma = 0.48 * np.exp(0.46 * betacs - 0.27 * betamax)
        else:
            print('Mechanism not found')
    return gamma

#Function to calculate the implicated reliability from the safety factor
def calc_beta_implicated(mechanism,SF,TrajectInfo=None):
    if SF == 0:
        # print('SF for ' + mechanism + ' is 0')
        beta = 0.5
    elif SF == np.inf:
        beta = 8
    else:
        if mechanism == 'Piping':
            beta = (1 / 0.37) * (np.log(SF / 1.04) + 0.43 * TrajectInfo['beta_max']) #-norm.ppf(TrajectInfo['Pmax']))
        elif mechanism == 'Heave':
            #TODO troubleshoot the RuntimeWarning errors with invalid values in log.
            beta = (1 / 0.48) * (np.log(SF / 0.37) + 0.30 * TrajectInfo['beta_max']) #-norm.ppf(TrajectInfo['Pmax']))
        elif mechanism == 'Uplift':
            beta = (1 / 0.46) * (np.log(SF / 0.48) + 0.27 * TrajectInfo['beta_max']) #-norm.ppf(TrajectInfo['Pmax']))
        else:
            print('Mechanism not found')
    return beta




#Calculates total probability from list of sections for a mechanism or for all mechanisms that can be found (to be programmed)
def calc_traject_prob(sections, mechanism):
    if isinstance(sections[0], float):
        # traject_prob = sum(sections)
        traject_prob = 1-(np.prod(np.subtract(1,sections)))
    else:
        Psections = []
        if mechanism == 'Piping':
            for i in range(0,len(sections)):
                if isinstance(sections[i].Reliability.Piping.Pf,float):
                    betaCS = pf_to_beta(sections[i].Reliability.Piping.Pf)
                else:
                    betaCS = max((sections[i].Reliability.Piping.beta_cs_h, sections[i].Reliability.Piping.beta_cs_p, sections[i].Reliability.Piping.beta_cs_u))
                Psections.append(beta_to_pf(betaCS))
        # traject_prob = sum(Psections)
        traject_prob = 1-(np.prod(np.subtract(1,Psections)))
    return traject_prob
def compute_decimation_height(h,p, n=2):
    #computes the average decimation height for the lower parts of a distribution: h are water levels, p are exceedence probabilities. n is the number of 'decimations'
    hp = interp1d(p, h)
    h_low = hp(p[0])       #lower limit
    h_high = hp((p[0])/(10 * n))
    return (h_high-h_low)/n

def MHWtoGumbel(MHW,p,d):
    a = MHW + d * np.log(-(np.log(1-p))) / (np.log(-np.log(1-p))-np.log(-np.log(1-p/10)))
    b = d /(np.log(-np.log(1-p))-np.log(-np.log(1-p/10)))
    return a, b

class TableDist(ot.PythonDistribution):
    def __init__(self, x=[0,1], p=[1,0], extrap = False,isload = False,gridpoints = 2000):
        super(TableDist, self).__init__(1)
        #Check the input
        if len(x) != len(p):
            raise ValueError('Input arrays have unequal lengths')
        if not extrap:
            if p[0] != 1 or p[-1:] != 0:
                raise ValueError('Probability bounds are not equal to 0 and 1. Allow for extrapolation or change input')
        for i in range(1,len(x)):
            if x[i-1] > x[i]:
                raise ValueError('Values should be increasing')
            if p[i-1] > p[i]:
                raise ValueError('Non-exceedance probabilities should be increasing')
        #Define the distribution
        pp1 = 1; pp0 = 0
        if isload:
            pgrid = 1-np.logspace(0,-8,gridpoints)
            # we add a zero point to prevent excessive extrapolation. We do this based on the decimation height from the inserted points.
            d10 = compute_decimation_height(x,1-p)
            p_low = 1-p[0]
            #determine water level with 100\% chance of occuring in a year
            p = np.concatenate(([0.], p))
            x_low = x[0] - (1/p_low) * (d10/10)
            x = np.concatenate(([x_low],x))
        else:
            pgrid = np.logspace(0, -8, gridpoints)
            # pgrid = 1-np.logspace(0,-16,500)

        # spline order: 1 linear, 2 quadratic, 3 cubic ...
        order = 1
        # do inter/extrapolation
        s = InterpolatedUnivariateSpline(p, x, k=1)
        xgrid = s(pgrid)
        if xgrid[0]-xgrid[-1:]>0:
            self.x = np.flip(xgrid,0)
            self.xp = np.flip(pgrid, 0)
            self.xp[0] = 0.

        else:
            self.x = xgrid
            self.xp = pgrid
            self.xp[-1:] = 1.
    def getParameterDescription(self):
        descr1 = []
        descr2 = []
        for i in range(0,len(self.x)):
            descr1.append('x_' + str(i))
            descr2.append('xp_' + str(i))
        return(descr1+descr2)

    def getParameter(self):
        x = []; xp = []
        for i in range(0,len(self.x)):
            x.append(self.x[i])
            xp.append(self.xp[i])
        return ot.Point(x+xp)
    def computeCDF(self, X):
        if X < self.x[0]:
            return 0.0
        elif X >= self.x[-1:]:
            return 1.0
        else:
            #find first value that is larger:
            #Option 1, seems to be slightly slower:
            # idx_up = min(np.argwhere(self.x>X))
            # xx = self.x[int(idx_up)-1:int(idx_up)+1]
            # pp = self.xp[int(idx_up)-1:int(idx_up)+1]
            # f = interp1d(xx,pp)
            # p = f(X)
            X = X[0]

            # idx_up = np.min(np.argwhere(self.x > X))
            idx_up = np.argmax(self.x>X)
            xx = self.x[idx_up - 1:idx_up + 1]
            pp = self.xp[idx_up - 1:idx_up + 1]
            dp = pp[1] - pp[0]
            dx = xx[1] - xx[0]
            p = pp[0] + dp * ((X - xx[0]) / dx)

            return p
    def computeQuantile_alternative(self,p,tail=False):
            if tail:            #if input p is to be interpreted as exceedence probability
                p = 1-p
            #Linearly interpolate between two values


            # idx_up = np.min(np.argwhere(self.x > X))
            #find index above
            idx_up = np.argmax(self.xp>p)

            xx = self.x[idx_up - 1:idx_up + 1]
            pp = self.xp[idx_up - 1:idx_up + 1]
            dp = pp[1] - pp[0]
            dx = xx[1] - xx[0]
            x = xx[0] + dx * ((p - pp[0]) / dp)
            return x
    def getMean(self):
        high = np.min(np.argwhere(self.xp > 0.53))
        low = np.min(np.argwhere(self.xp>0.47))
        # high = np.min(np.argwhere(self.xp > 0.50))
        # low = high-1
        index = low+(np.abs(0.5 - self.xp[low:high])).argmin()
        mu = np.interp(0.5, self.xp[index - 1:index + 1], self.x[index - 1:index + 1])
        return [mu]
    def getRange(self):
        return ot.Interval([self.x[0]], [float(self.x[-1:])], [True], [True])

    def getRealization(self):
        X = []
        p = ot.RandomGenerator.Generate()
        idx_up = min(np.argwhere(self.xp > p)) #CHECK
        pp = self.xp[int(idx_up) - 1:int(idx_up) + 1]
        xx = self.x[int(idx_up) - 1:int(idx_up) + 1]
        f = interp1d(pp, xx)
        X = float(f(p))
        return ot.Point(1,X)

        # sample = h.getSample(50000)
        # from openturns.viewer import View
        # graph = ot.VisualTest_DrawEmpiricalCDF(sample)
        # orig = ot.Curve(wls, p_nexc)
        # graph.add(orig)
        # View(graph).show()
    def getSample(self, size):
        X = []
        for i in range(size):
            X.append(self.getRealization())
        return X

def FragilityIntegration(FragilityCurve, WaterLevelDist, WaterLevelChange = False, N=1600,  PrintResults=False):
    if WaterLevelChange:
        if (WaterLevelChange.getClassName() == 'Dirac') and (WaterLevelDist.distribution.getName() =='TableDist'):
            pass
            half = int(0.5 * len(WaterLevelDist.distribution.getParameter()))
            x = np.asarray(WaterLevelDist.distribution.getParameter()[0:half]) + np.asarray(WaterLevelChange.getParameter())
            p = WaterLevelDist.distribution.getParameter()[half:]
            # plt.semilogy(x, np.subtract(1, p))
            # plt.xlim(left=5)
            # new_dist = TableDist(x, p)
        else:
            raise Exception('not implemented')

    ux = np.linspace(-8.0, 8.0, N+1)
    px = norm.cdf(ux)
    interpolator = sp.interpolate.interp1d(p, x, fill_value='extrapolate')
    h = interpolator(px)
    cdf_hc = GetValueFromFragilityCurve(FragilityCurve, Value='pf', x=h)
    dx = ux[1] - ux[0]
    pdf_h = norm.pdf(ux)
    Pfs = cdf_hc * pdf_h * dx
    Pf = Pfs.sum()
    beta = -1*norm.ppf(Pf)
    import matplotlib.pyplot as plt

    # plt.plot(h, pdf_h, label='pdf h')
    # plt.plot(h, cdf_hc, label='fragility curve')
    # plt.plot(h, Pfs / dx, label='multiplied (no scaling)')
    # plt.legend()
    if PrintResults:
        print('\n'
              'Integrated Results Numerical Integration \n'
              '======================================== \n'
              'Beta = %0.2f \n'
              'Pf   = %0.2e \n' %(beta,Pf)
              )

    return Pf, beta

def GetValueFromFragilityCurve(FragilityCurve, Value, x):
    if Value=='h':
        interpolator = sp.interpolate.interp1d(FragilityCurve['beta'], FragilityCurve['h'], fill_value='extrapolate')
        h = interpolator(x)
        return h
    elif Value=='beta' or Value=='pf':
        interpolator = sp.interpolate.interp1d(FragilityCurve['h'], FragilityCurve['beta'], fill_value='extrapolate')
        beta_hc = interpolator(x)
        if Value=='pf':
            pf = norm.cdf(-beta_hc)
            return pf
        else:
            return beta



def run_prob_calc(model,dist,method='FORM',startpoint=False):
    vect = ot.RandomVector(dist)
    if method == 'MCS':
        model = ot.MemoizeFunction(model)
    G = ot.CompositeRandomVector(model, vect)
    event = ot.Event(G, ot.Less(), 0)

    if method == 'FORM':
        solver = ot.AbdoRackwitz()
        solver.setMaximumAbsoluteError(1e-2)
        solver.setMaximumRelativeError(1e-2)
        solver.setMaximumIterationNumber(200)
        if startpoint:
            algo = ot.FORM(solver, event, startpoint)
        else:
            algo = ot.FORM(solver, event, dist.getMean())

        # else:
        #     startpoint = 0
        algo.run()
        result = algo.getResult()
        Pf = result.getEventProbability()
        beta = result.getHasoferReliabilityIndex()
        alfas_sq = (np.array(result.getStandardSpaceDesignPoint())/beta)**2
    elif method =='DIRS':
        result, algo = run_DIRS(event, approach=ot.MediumSafe(), samples=1000)
        Pf = result.getProbabilityEstimate()
    elif method == 'MCS':
        ot.RandomGenerator.SetSeed(5000)
        print('Warning, Random Generator state is currently fixed!')
        experiment = ot.MonteCarloExperiment()
        algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
        algo.setMaximumCoefficientOfVariation(0.05)
        algo.setMaximumStandardDeviation(1e-3)
        algo.setMaximumOuterSampling(int(1000))
        algo.setBlockSize(100)
        algo.run()
        result = algo.getResult()
        Pf = result.getProbabilityEstimate()
        beta = pf_to_beta(Pf)
        if beta != float('Inf'):
            alfas_sq = np.array(result.getImportanceFactors())
        else:
            alfas_sq = np.empty((1,vect.getDimension(),))
            alfas_sq[:] = np.nan
    return result, Pf, beta, alfas_sq


def run_DIRS(event, approach=ot.MediumSafe(),sampling=ot.OrthogonalDirection(),samples = 250):
    # start = time.time()
    algo = ot.DirectionalSampling(event,approach,sampling)
    algo.setMaximumOuterSampling(samples)
    algo.setBlockSize(4)
    algo.setMaximumCoefficientOfVariation(0.025)
    algo.run()
    result = algo.getResult()
    probability = result.getProbabilityEstimate()
    # end = time.time()
    print(event.getName())
    print('%15s' % 'Pf = ', "{0:.2E}".format(probability))
    print('%15s' % 'CoV = ', "{0:.2f}".format(result.getCoefficientOfVariation()))
    print('%15s' % 'N = ', "{0:.0f}".format(result.getOuterSampling()))
    # print('%15s' % 'Time elapsed = ', "{0:.2f}".format(end-start), 's')
    return result, algo

def IterativeFC_calculation(marginals,WL, names, zFunc, method, step=0.5, lolim = 10e-4, hilim=0.999):
    marginals[len(marginals) - 1] = ot.Dirac(float(WL))
    dist = ot.ComposedDistribution(marginals)
    dist.setDescription(names)
    result, P,beta,alpha = run_prob_calc(ot.PythonFunction(len(marginals), 1, zFunc), dist, method)
    wl_list = []; result_list = []; P_list = [];
    while P >hilim or P < lolim:
        print('changed start value')
        WL = WL - 1 if P > hilim else WL + 1
        marginals[len(marginals) - 1] = ot.Dirac(float(WL))
        dist = ot.ComposedDistribution(marginals)
        dist.setDescription(names)
        result, P,beta,alpha = run_prob_calc(ot.PythonFunction(len(marginals), 1, zFunc), dist, method)

    result_list.append(result)
    wl_list.append(WL)
    P_list.append(P)
    count = 0
    while P > lolim:
        WL -=step
        count += 1
        marginals[len(marginals) - 1] = ot.Dirac(float(WL))
        dist = ot.ComposedDistribution(marginals); dist.setDescription(names)
        result, P,beta,alpha = run_prob_calc(ot.PythonFunction(len(marginals), 1, zFunc), dist, method)
        result_list.append(result)
        wl_list.append(WL)
        P_list.append(P)
        print(str(count) + ' calculations made for fragility curve')
    WL = max(wl_list)
    while P < hilim:
        WL +=step
        count += 1
        marginals[len(marginals) - 1] = ot.Dirac(float(WL))
        dist = ot.ComposedDistribution(marginals); dist.setDescription(names)
        result, P,beta,alpha = run_prob_calc(ot.PythonFunction(len(marginals), 1, zFunc), dist, method)
        result_list.append(result)
        wl_list.append(WL)
        P_list.append(P)
        print(str(count) + ' calculations made for fragility curve')

    indices = list(np.argsort(wl_list))
    wl_list = [wl_list[i] for i in indices]
    result_list = [result_list[i] for i in indices]
    P_list = [P_list[i] for i in indices]
    indexes = np.where(np.diff(P_list) == 0)
    rm_items = 0
    if len(indexes[0])>0:
        for i in indexes[0]:
            wl_list.pop(i-rm_items)
            result_list.pop(i-rm_items)
            P_list.pop(i-rm_items)
            rm_items +=1
    #remove the non increasing values
    print()
    return result_list, P_list, wl_list

def TemporalProcess(input, t,makePlot='off'):
    #TODO check of input == float.
    if isinstance(input, float):
        input = ot.Dirac(input*t) #make distribution
   #This function derives the distribution parameters for the temporal process governed by the annual distribution 'input' for year 't'
    elif input.getClassName() == 'Gamma':
        params = input.getParameter()
        mu = params[0] / params[1]
        var = params[0] / (params[1] ** 2)
        input.setParameter(ot.GammaMuSigma()([mu*float(t), np.sqrt(var)*float(t),0]))
        if makePlot=='on':
            gr = input.drawPDF()
            from openturns.viewer import View
            view = View(gr)
            view.show()
    elif input.getClassName() == 'Dirac':
        input.setParameter(input.getParameter()*t)
    else:
        raise Exception('Distribution type for temporal process not recognized.')
    return input

def UpscaleCDF(dist,t=1,testPlot='off' ,change_dist = None,change_step = 1,Ngrid = None):
    #function to upscale an exceedance probability curve to another time scale
    #change_dist provides the opportunity to include an (uncertain) temporally changing PDF (e.g. sea level rise)
    #if that is added the function will provide the CDF of the water level for period t given a change over time
    freq = []
    pnew = []
    if dist.getName() == 'TableDist':
        params = dist.getParameter()
        if Ngrid == None:
            x = np.split(np.array(params),2)[0]
        else:
            x =np.linspace(np.min(np.split(np.array(params),2)[0]),np.max(np.split(np.array(params),2)[0]),Ngrid)
        tgrid = np.arange(1,t+1,change_step)
        if isinstance(change_dist,type(None)):
            for i in x:
                freq.append(-np.log(dist.computeCDF(i))*t)
                pnew.append(np.exp(-freq[-1:][0]))
        else:
            distcoll = []

            for j in tgrid:
                original = copy.deepcopy(change_dist)
                distcoll.append(ot.RandomMixture([dist,TemporalProcess(original,j,makePlot='off')],[1.0,1.0],0.0))
                pass
            #derive factors:
            factors = []
            for i in range(0, len(tgrid)):
                if i == 0 or i == len(tgrid) - 1:
                    factors.append((change_step / 2) + 0.5)
                else:
                    factors.append((change_step))
            if tgrid[-1:] < t:
                factors[-1:]=factors[-1:]+(t-tgrid[-1:])
                print('Warning: range is not optimal. Last point has been extended to end of time window')
            #calculate frequencies
            freq = np.empty((len(x),len(distcoll)))
            for i in range(0,len(distcoll)):
                for j in range(0,len(x)):
                    freq[j,i] = -np.log(distcoll[i].computeCDF(x[j]))*factors[i]
            frequencies = np.sum(freq,axis=1)
            pnew = np.exp(-frequencies)
        newdist = ot.Distribution(TableDist(list(x),list(pnew),extrap=True, isload=True))
        if testPlot == 'on':
            wl = np.arange(1, 10, 0.1)
            pold = []
            pnuevo = []
            for i in wl:
                pold.append(1 - dist.computeCDF(i))
                pnuevo.append(1 - newdist.computeCDF(i))
            import matplotlib.pyplot as plt
            plt.plot(wl, pold)
            plt.plot(wl, pnuevo)
            plt.yscale('log')
            plt.show()

    return newdist

def getDesignWaterLevel(load,p):
    return np.array(load.distribution.computeQuantile(1 - p))[0]

def addLoadCharVals(input,load=None,p_h = 1./1000, p_dh=0.5,year = 0):
    #TODO this function should be moved elsewhere
    #input = list of all strength variables

    if load != None:
        if isinstance(load.distribution,dict):
            if str(np.int32(year+config.t_0)) in list(load.distribution.keys()):
                h_norm = np.array(load.distribution[str(np.int32(year+config.t_0))].computeQuantile(1 - p_h))[0]
            else:
                #for each year, compute WL
                years = [np.int32(i) for i in list(load.distribution.keys())]
                wls = []
                for j in years:
                    wls.append(load.distribution[str(j)].computeQuantile(1-p_h)[0])
                h_norm = interp1d(years,wls,fill_value='extrapolate')(year+config.t_0)
                #then interpolate for given year
        else:
            h_norm = np.array(load.distribution.computeQuantile(1 - p_h))[0]
        input['h'] = h_norm

    if hasattr(load, 'dist_change'):
        if isinstance(load.dist_change,float):      #for SAFE input
            # this is only for piping and stability. For overflow it should be extended with use of the HBN factor
            input['dh'] = load.dist_change * year
        else:
            p = 0.5
            dh = np.array(load.dist_change.computeQuantile(p_dh))[0]
            input['dh'] = dh * year
    else:
        input['dh'] = 0.
    return input

def MarginalsforTimeDepReliability(input,load=None,year=0,type=None):
    marginals = []; names = [];

    #Strength variables:
    for i in input.input.keys():
        # Adapt temporal process variables
        if i in input.temporals:
            original = copy.deepcopy(input.input[i])
            adapt_dist = TemporalProcess(original, year)
            marginals.append(adapt_dist)
            names.append(i)
        else:
            marginals.append(input.input[i])
            names.append(i)

    #Load variables:
        if type =='ConstructFC':
            marginals.append(ot.Dirac(1.)); names.append('h') #add the load
        elif type == 'CalcFC' or type == 'Probabilistic':
            marginals.append(load.distribution)
            if hasattr(load, 'dist_change'):
                original = copy.deepcopy(load.dist_change)
                dist_change = TemporalProcess(original, year)
                marginals.append(dist_change)
                names.extend(('h','dh'))

            else:
                names.append('h')

    return marginals, names

###################################################################################################
## THESE ARE FASTER FORMULAS FOR CONVERTING BETA TO PROB AND VICE VERSA
def erf(x):
    ''' John D. Cook's implementation.http://www.johndcook.com
    >> Formula 7.1.26 given in Abramowitz and Stegun.
    >> Formula appears as 1 – (a1t1 + a2t2 + a3t3 + a4t4 + a5t5)exp(-x2)
    >> A little wisdom in Horner's Method of coding polynomials:
        1) We could evaluate a polynomial of the form a + bx + cx^2 + dx^3 by coding as a + b*x + c*x*x + d*x*x*x.
        2) But we can save computational power by coding it as ((d*x + c)*x + b)*x + a.
        3) The formula below was coded this way bringing down the complexity of this algorithm from O(n2) to O(n).'''

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    # sign = 1
    # if x < 0:
    #     sign = -1
    if np.any(np.isnan(np.where(x <0,-1,1))):
        print()
    sign = np.where(x < 0, -1, 1)
    x = abs(x)

    # Formula 7.1.26 given in Abramowitz and Stegun.
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    # y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


####################################################################################################

def phi(x):
    '''Cumulative gives a probability that a statistic
    is less than Z. This equates to the area of the
    distribution below Z.
    e.g:  Pr(Z = 0.69) = 0.7549. This value is usually
    given in Z tables.'''

    return 0.5 * (1.0 + erf(x / math.sqrt(2)))


#####################################################################################################

def phi_compcum(x):
    ''' Complementary cumulative gives a probability
    that a statistic is greater than Z. This equates to
    the area of the distribution above Z.
    e.g: Pr(Z  =  0.69) = 1 - 0.7549 = 0.2451'''

    return abs(phi(x) - 1)


#####################################################################################################

def phi_cumformu(x):
    '''Cumulative from mean gives a probability
    that a statistic is between 0 (mean) and Z.
    e.g: Pr(0 = Z = 0.69) = 0.2549'''

    return phi_compcum(0) - phi_compcum(x)


def formula(t):
    # constants
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d0 = 1.432788
    d1 = 0.189269
    d2 = 0.001308

    # Formula
    p = t - ((c2 * t + c1) * t + c0) / (((d2 * t + d1) * t + d0) * t + 1.0)
    return p


def phi_inv(p):
    x1 = -formula(np.sqrt(-2.0 * np.log(p)))
    x2 = formula(np.sqrt(-2.0 * np.log(1 - p)))
    if np.any(np.isnan(np.where(p<0.5,x1,x2))):
        print()
    return np.where(p<0.5,x1,x2)
    #
    # if (p < 0.5):
    #     # F^-1(p) = - G^-1(p)
    #     return -formula(math.sqrt(-2.0 * math.log(p)))
    # else:
    #     # F^-1(p) = G^-1(1-p)
    #     return formula(math.sqrt(-2.0 * math.log(1 - p)))

    return q

def beta_to_pf(beta):
    #alternative: use scipy
    return norm.cdf(-beta)
    # if isinstance(beta,np.ndarray):
    #     pf = phi(-beta.astype(np.float64))
    # else:
    #     pf = phi(-np.float64(beta))
    # return pf

def pf_to_beta(pf):
    #alternative: use scipy
    return -norm.ppf(pf)

    # if isinstance(pf,np.ndarray):
    #     beta = -phi_inv(pf.astype(np.float64))
    #     # if np.any(np.isnan(beta)):
    #
    # else:
    #     beta = -phi_inv(np.float64(pf))
    # return beta

def main():
    pass

if __name__ == "__main__":
    main()
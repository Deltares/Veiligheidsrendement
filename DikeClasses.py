import ProbabilisticFunctions
import Mechanisms
#Class for the input of a mechanism. Function only available for piping for now
class MechanismInput:
    def __init__(self,mechanism):
        self.mechanism = mechanism

    def fill_piping_data(self, dict, type):
        if type == 'SemiProb':
            self.inputtype = type
            #Input needed for multiple submechanisms
            self.h           = dict['Input']['Scenario 1']['j 0 '][1]
            self.h_exit      = dict['Input']['Scenario 1']['hp = j 3'][1]
            self.gamma_w     = dict['Input']['Scenario 1']['gw'][1]
            self.r_exit      = (dict['Input']['Scenario 1']['j2'][1] - self.h_exit)/(dict['Input']['Scenario 1']['j 0 '][1]-self.h_exit)
            self.d_cover     = dict['Input']['Scenario 1']['d,pot'][1]
            self.gamma_schem = dict['Input']['Scenario 1']['gb,u'][1]
            #Input parameter specifically for piping
            self.D           = dict['Input']['Scenario 1']['D'][1]
            self.k           = dict['Input']['Scenario 1']['kzand,pip'][1]
            self.L_voorland  = dict['Input']['Scenario 1']['L1:pip'][1]
            self.L_dijk      = dict['Input']['Scenario 1']['L2'][1]
            self.L_berm      = dict['Input']['Scenario 1']['L3'][1]
            self.L           = self.L_berm + self.L_dijk + self.L_voorland
            self.d70         = dict['Input']['Scenario 1']['d70'][1]
            self.d_cover_pip = dict['Input']['Scenario 1']['d,pip'][1]
            self.m_Piping    = 1.0
            self.theta       = 37.
            #Input specific for heave
            self.scherm      = dict['Results']['Scenario 1']['Heave']['kwelscherm aanwezig']

            #Input specific for uplift
            self.gamma_sat   = dict['Input']['Scenario 1']['Gemiddeld volumegewicht:'][1]
            print(self.gamma_sat)
            self.gamma_sat = 18 if self.gamma_sat == 0. else self.gamma_sat
            print(self.gamma_sat)
            # self.d70m        = dict['Input']['Scenario 1']['d70m'] NB: nu and eta are also already defined
        else:
            print('unknown type')

#Class describing an assessment (functions for piping, heave & uplift available)
class Assessment:
    def __init__(self,mechanism,type):
        self.mechanism = mechanism
        self.type = 'SemiProb'
    def __clearvalues__(self):
        keys = self.__dict__.keys()
        for i in keys:
            if i is not 'mechanism':
                setattr(self,i,None)

        print('runned it')
    def Assess(self,DikeSection, MechanismInput):
        #First calculate the SF without gamma for the three submechanisms
        #Piping:
        Z, self.p_dh, self.p_dh_c = Mechanisms.LSF_sellmeijer(MechanismInput)                   #Calculate hydraulic heads
        self.gamma_pip   = ProbabilisticFunctions.calc_gamma('Piping',DikeSection)                                         #Calculate needed safety factor
        # Check if it is OK, NB: Schematization factor IS NOT included here. Which is correct because a scenario approach is taken.
        self.SF_p = (self.p_dh_c/self.gamma_pip)/self.p_dh
        self.assess_p  = 'voldoende' if (self.p_dh_c/self.gamma_pip)/self.p_dh > 1 else 'onvoldoende'
        self.beta_cs_p = ProbabilisticFunctions.calc_beta_implicated('Piping',self.p_dh_c/self.p_dh,DikeSection)     #Calculate the implicated beta_cs

        #Heave:
        Z, self.h_i, self.h_i_c = Mechanisms.LSF_heave(MechanismInput)                                  #Calculate hydraulic heads
        self.gamma_h   = ProbabilisticFunctions.calc_gamma('Heave',DikeSection)                                            #Calculate needed safety factor
        # Check if it is OK, NB: Schematization factor IS included here
        self.SF_h = (self.h_i_c/(MechanismInput.gamma_schem*self.gamma_h))/self.h_i
        self.assess_h  = 'voldoende' if (self.h_i_c/(MechanismInput.gamma_schem*self.gamma_h))/self.h_i > 1 else 'onvoldoende'
        self.beta_cs_h = ProbabilisticFunctions.calc_beta_implicated('Heave',self.h_i_c/self.h_i,DikeSection)                 #Calculate the implicated beta_cs

        #Uplift
        Z, self.u_dh, self.u_dh_c = Mechanisms.LSF_uplift(MechanismInput)                                  #Calculate hydraulic heads
        self.gamma_u   = ProbabilisticFunctions.calc_gamma('Uplift',DikeSection)                                            #Calculate needed safety factor
        # Check if it is OK, NB: Schematization factor IS included here
        self.SF_u = (self.u_dh_c/(MechanismInput.gamma_schem*self.gamma_u))/self.u_dh
        self.assess_u  = 'voldoende' if (self.u_dh_c/(MechanismInput.gamma_schem*self.gamma_u))/self.u_dh > 1 else 'onvoldoende'
        self.beta_cs_u = ProbabilisticFunctions.calc_beta_implicated('Uplift',self.u_dh_c/self.u_dh,DikeSection)                 #Calculate the implicated beta_cs


#initialize the DikeSection class, as a general class for a dike section that contains all basic information
class DikeSection:
    def __init__(self, name, traject):
        self.TrajectInfo = {}
        if len(name)==8:
            self.name = name[0:5] + '0' + name[5:]
        else:
            self.name = name

        if traject == '16-4':
            self.TrajectInfo['TrajectLength'] = 19480
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
        elif traject == '16-3':
            self.TrajectInfo['TrajectLength'] = 19899
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
            #NB: klopt a hier?????!!!!

    def fill_from_dict(self, dict):
        #First the general info (to be added: traject info, norm etc)
        self.start = dict['General']['Traject start']
        self.end   = dict['General']['Traject end']
        self.CS    = dict['General']['Cross section']
        self.MHW   = dict['Input']['Scenario 1']['j 0 ']        #TO DO: add a loop over scenarios
        self.PipingIn = MechanismInput('Piping')
        self.PipingIn.fill_piping_data(dict,'SemiProb')

    def doAssessment(self, mechanism, type):
        if mechanism == 'Piping':

            self.PipingAssessment = Assessment(mechanism, type)
            self.PipingAssessment.Assess(self,self.PipingIn)

        else:
            print('Mechanism not known')

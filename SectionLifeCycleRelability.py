## This script can calculate life-cycle reliability and costs for all measures for various mechanisms

from HelperFunctions import ld_readObject


#We read a section
#For now we make two dummy sections
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\16-4\input'
Sections = ld_readObject(pad + '\\AllSections.dta')
#For each mechanism
#We calculate current and future rate of failure without measures

#We read a set of measures that are possible
#We calculate current and future failure rates with these measures/strategies

#We combine for each set of (no) measures to a conditional failure rate on a section level. Therefore we
#Upscale for length effects in the section
#Translate to conditional failure rates
#Combine the mechanisms
print()
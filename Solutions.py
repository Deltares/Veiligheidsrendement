import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from Measure import Measure, Soilreinforcement, DiaphragmWall, StabilityScreen, VerticalGeotextile, CustomMeasure
import config

class Solutions:
    #This class contains possible solutions/measures
    def __init__(self, DikeSectionObject):
        self.SectionName = DikeSectionObject.name
        self.Length = DikeSectionObject.Length
        self.InitialGeometry = DikeSectionObject.InitialGeometry

    def fillSolutions(self,excelsheet):
        """This routine reads input for the measures from the Excel sheet for each section.
        It identifies combinables and partials and identifies possible combinations of measures this way.
        These are then stored in the MeasureTable, which is later evaluated.
        """
        data = pd.read_excel(excelsheet,'Measures')
        self.Measures = []
        combinables = []
        partials = []
        for i in data.index:
            #TODO depending on data.loc[i].type make correct sublclass
            self.Measures.append(Measure(data.loc[i]))

        self.MeasureTable = pd.DataFrame(columns=['ID', 'Name'])
        for i, measure in enumerate(self.Measures):
            if measure.parameters['available'] == 1:
                self.MeasureTable.loc[i] = [str(measure.parameters['ID']), measure.parameters['Name']]
                #also add the potential combined solutions up front
                if measure.parameters['Class'] == 'combinable':
                    combinables.append((measure.parameters['ID'],measure.parameters['Name']))
                if measure.parameters['Class'] == 'partial':
                    partials.append((measure.parameters['ID'],measure.parameters['Name']))
        count = 0
        for i in range(0,len(partials)):
            for j in range(0,len(combinables)):
                self.MeasureTable.loc[count+len(self.Measures)+1] = [str(partials[i][0]) + '+' + str(combinables[j][0]),
                                                                     str(partials[i][1]) + '+' + str(combinables[j][1])]
                count += 1



    def evaluateSolutions(self,DikeSection,TrajectInfo,preserve_slope = False):
        """This is the base routine to evaluate (i.e., determine costs and reliability) for each defined measure.
        It also gathers those measures for which availability is set to 0 and removes these from the list of measures."""
        self.trange = config.T
        removal = []
        for i, measure in enumerate(self.Measures):
            if measure.parameters['available'] == 1:
                # old: measure.evaluateMeasure(DikeSection, TrajectInfo, preserve_slope = preserve_slope)

                if measure.parameters['Type'] == 'Soil reinforcement':
                    Soilreinforcement(DikeSection, TrajectInfo, preserve_slope = preserve_slope)
                elif measure.parameters['Type'] == 'DiaphragmWall':
                    DiaphragmWall(DikeSection, TrajectInfo)
                elif measure.parameters['Type'] == 'StabilityScreen':
                    StabilityScreen(DikeSection, TrajectInfo)
                elif measure.parameters['Type'] == 'VerticalGeotextile':
                    VerticalGeotextile(DikeSection, TrajectInfo)
                elif measure.parameters['Type'] == 'Custom':
                    CustomMeasure()
            else:
                removal.append(i)
        #remove measures that are set to unavailable:
        if len(removal) > 0:
            for i in reversed(removal):
                self.Measures.pop(i)

    def SolutionstoDataFrame(self, filtering=False,splitparams = False):
        #write all solutions to one single dataframe:

        years = self.trange
        cols_r = pd.MultiIndex.from_product([config.mechanisms + ['Section'], years], names=['base', 'year'])
        reliability = pd.DataFrame(columns=cols_r)
        if splitparams:
            cols_m = pd.Index(['ID', 'type', 'class', 'year', 'yes/no', 'dcrest', 'dberm', 'cost'], name='base')
        else:
            cols_m = pd.Index(['ID', 'type', 'class', 'year', 'params', 'cost'], name='base')
        measure_df = pd.DataFrame(columns=cols_m)
        # data = pd.DataFrame(columns = cols)
        inputs_m = []
        inputs_r = []

        for i, measure in enumerate(self.Measures):
            if isinstance(measure.measures, list):
                #if it is a list of measures (for soil reinforcement): write each entry of the list to the dataframe
                typee = measure.parameters['Type']

                for j in range(len(measure.measures)):
                    measure_in = []
                    reliability_in = []
                    if typee == 'Soil reinforcement':
                        designvars = ((measure.measures[j]['dcrest'], measure.measures[j]['dberm']))

                    cost = measure.measures[j]['Cost']
                    measure_in.append(str(measure.parameters['ID']))
                    measure_in.append(typee)
                    measure_in.append(measure.parameters['Class'])
                    measure_in.append(measure.parameters['year'])
                    if splitparams:
                        measure_in.append(-999)
                        measure_in.append(designvars[0])
                        measure_in.append(designvars[1])
                    else:
                        measure_in.append(designvars)
                    measure_in.append(cost)

                    betas = measure.measures[j]['Reliability'].SectionReliability

                    for ij in config.mechanisms + ['Section']:
                        for ijk in betas.loc[ij].values:
                            reliability_in.append(ijk)

                    inputs_m.append(measure_in)
                    inputs_r.append(reliability_in)

            elif isinstance(measure.measures, dict):
                ID = str(measure.parameters['ID'])
                typee = measure.parameters['Type']
                if typee == 'Vertical Geotextile':
                    designvars = measure.measures['VZG']

                if typee == 'Diaphragm Wall':
                    designvars = measure.measures['DiaphragmWall']

                classe = measure.parameters['Class']
                yeare  = measure.parameters['year']
                cost = measure.measures['Cost']
                if splitparams:
                    inputs_m.append([ID, typee, classe, yeare, designvars, -999 , -999 ,cost])
                else:
                    inputs_m.append([ID, typee, classe, yeare, designvars, cost])
                betas = measure.measures['Reliability'].SectionReliability
                beta = []
                for ij in config.mechanisms + ['Section']:
                    for ijk in betas.loc[ij].values:
                        beta.append(ijk)
                inputs_r.append(beta)
        reliability = reliability.append(pd.DataFrame(inputs_r, columns=cols_r))
        measure_df = measure_df.append(pd.DataFrame(inputs_m, columns=cols_m))
        self.MeasureData = measure_df.join(reliability,how='inner')
        #fix multiindex:
        index = []
        for i in self.MeasureData.columns:
            index.append(i) if isinstance(i,tuple) else index.append((i,''))
        self.MeasureData.columns = pd.MultiIndex.from_tuples(index)
        if filtering: #here we could add some filtering on the measures, but it is not used right now.
            pass

    def plotBetaTimeEuro(self, measures='undefined',mechanism='Section',beta_ind = 'beta0',sectionname='Unknown',beta_req=None):
        # This function plots the relation between cost and beta in a certain year

        #measures is a list of measures that need to be plotted
        if measures == 'undefined':
            measures = list(self.Measures.keys())

        #mechanism can be used to select a single or all ('Section') mechanisms
        #beta can be used to use a criterion for selecting the 'best' designs, such as the beta at 't0'
        cols = ['type', 'parameters', 'Cost']
        [cols.append('beta' + str(i)) for i in self.trange]
        data = pd.DataFrame(columns=cols)
        num_plots = 5
        colors = sns.color_palette('hls', n_colors=num_plots)
        # colors = plt.cm.get_cmap(name=plt.cm.hsv, lut=num_plots)
        color = 0

        for i in np.unique(self.MeasureData['ID'].values):
            if isinstance(self.Measures[int(i) - 1].measures, list):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData['ID'] == i])
                # inputs = []; type = self.Measures[i].parameters['Type']
                # for j in range(0, len(self.Measures[i].measures)):
                #     inputvals = []
                #     if type == 'Soil reinforcement': designvars = str((self.Measures[i].measures[j]['dcrest'], self.Measures[i].measures[j]['dberm']))
                #     betas = list(self.Measures[i].measures[j]['Reliability'].SectionReliability.loc[mechanism])
                #     cost = self.Measures[i].measures[j]['Cost']
                #     inputvals.append(type); inputvals.append(designvars); inputvals.append(cost)
                #     for ij in range(0, len(betas)): inputvals.append(betas[ij])
                #     inputs.append(inputvals)
                # data = data.append(pd.DataFrame(inputs, columns=cols))
                # x = data.loc[data['type'] == 'Soil reinforcement']
                y = copy.deepcopy(data)
                x = data.sort_values(by=['cost'])

                steps = 20
                cost_grid = np.linspace(np.min(x['cost']), np.max(x['cost']), steps)
                envelope_beta = []
                envelope_costs = []
                indices = []
                betamax = 0

                for j in range(len(cost_grid) - 1):
                    values = x.loc[(x['cost'] >= (cost_grid[j])) & (x['cost'] <= (cost_grid[j + 1]))][(mechanism, beta_ind)]
                    if len(list(values)) > 0:
                        idd = values.idxmax()
                        if betamax < np.max(list(values)):
                            betamax = np.max(list(values))
                            indices.append(idd)
                            if isinstance(x['cost'].loc[idd], pd.Series):
                                envelope_costs.append(x['cost'].loc[idd].values[0])

                            if not isinstance(x['cost'].loc[idd], pd.Series):
                                envelope_costs.append(x['cost'].loc[idd])

                            envelope_beta.append(betamax)

                if self.Measures[np.int(i)-1].parameters['Name'][-4:] != '2045':
                    plt.plot(envelope_costs, envelope_beta, color=colors[color], linestyle='-')
                    # [plt.text(y['Cost'].loc[ij], y[beta_ind].loc[i], y['parameters'].loc[ij],fontsize='x-small') for ij in indices]

                    plt.plot(y['cost'], y[(mechanism,beta_ind)], label = self.Measures[np.int(i)-1].parameters['Name'],
                             marker='o',markersize=6, color=colors[color],markerfacecolor=colors[color],
                             markeredgecolor=colors[color], linestyle='',alpha=1)

                    color += 1
            elif isinstance(self.Measures[np.int(i)-1].measures, dict):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData['ID'] == i])
                #
                # inputs = []; type = self.Measures[np.int(i)].parameters['Type']
                # if type == 'Vertical Geotextile': designvars = self.Measures[np.int(i)].measures['VZG']
                # if type == 'Diaphragm Wall': designvars = self.Measures[np.int(i)].measures['DiaphragmWall']
                # betas = list(self.Measures[np.int(i)].measures['Reliability'].SectionReliability.loc[mechanism])
                # cost = self.Measures[np.int(i)].measures['Cost']
                # inputs.append(type); inputs.append(designvars); inputs.append(cost);
                # for ij in range(0, len(betas)): inputs.append(betas[ij])
                # data = data.append(pd.DataFrame([inputs], columns=cols))
                plt.plot(data['cost'], data[(mechanism,beta_ind)], label = self.Measures[np.int(i)-1].parameters['Name'],
                         marker='d',markersize=10,markerfacecolor=colors[color],markeredgecolor=colors[color],linestyle='')
                color += 1
        axes = plt.gca()
        plt.plot([0, axes.get_xlim()[1]], [beta_req, beta_req], 'k--', label='Norm')
        plt.xlabel('Cost');
        plt.ylabel(r'$\beta_{' + str(beta_ind+2025) + '}$')
        plt.title('Cost-beta relation for ' + mechanism + ' at ' + sectionname)
        plt.legend(loc='best')
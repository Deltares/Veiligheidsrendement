import logging

import numpy as np
import pandas as pd

from vrtool.decision_making.strategies.strategy_base import StrategyBase


class MixedIntegerStrategy(StrategyBase):
    def _flatten_list(self, list_to_flatten: list) -> list:
        _flat_list = []
        if not list_to_flatten:
            return _flat_list
        for _item_to_flatten in list_to_flatten:
            if isinstance(_item_to_flatten, (list, tuple)):
                _flat_list.extend(self._flatten_list(_item_to_flatten))
            else:
                _flat_list.append(_item_to_flatten)
        return _flat_list

    def create_optimization_model(self, BudgetLimit=False):
        """Routine to create a mixed integer optimization model in CPLEX. Note that a valid installation of the CPLEX solver is required."""
        # make a model
        # enlist all the variables
        model = cplex.Cplex()
        grN = range(self.opt_parameters["N"])
        grSh = range(self.opt_parameters["Sh"])
        grSg = range(self.opt_parameters["Sg"])
        grT = range(self.opt_parameters["T"])

        # all variables
        Cint_nd = np.array(
            [
                [
                    [
                        "C" + str(n).zfill(3) + str(sh).zfill(3) + str(sg).zfill(3)
                        for sg in grSg
                    ]
                    for sh in grSh
                ]
                for n in grN
            ]
        )
        Gint_nd = np.array(
            [
                [
                    [
                        "G" + str(n).zfill(3) + str(sh).zfill(3) + str(sg).zfill(3)
                        for sg in grSg
                    ]
                    for sh in grSh
                ]
                for n in grN
            ]
        )
        Oint_nd = np.array(
            [
                [
                    [
                        "O" + str(n).zfill(3) + str(s).zfill(3) + str(t).zfill(3)
                        for t in grT
                    ]
                    for s in grSh
                ]
                for n in grN
            ]
        )

        # names of variables
        Cint = [
            "C" + str(n).zfill(3) + str(sh).zfill(3) + str(sg).zfill(3)
            for sg in grSg
            for sh in grSh
            for n in grN
        ]
        Gint = [
            "G" + str(n).zfill(3) + str(sh).zfill(3) + str(sg).zfill(3)
            for sg in grSg
            for sh in grSh
            for n in grN
        ]
        Oint = [
            "O" + str(n).zfill(3) + str(s).zfill(3) + str(t).zfill(3)
            for t in grT
            for s in grSh
            for n in grN
        ]

        VarNames = Cint + Gint + Oint
        nvar = (
            2
            * self.opt_parameters["N"]
            * self.opt_parameters["Sh"]
            * self.opt_parameters["Sg"]
            + self.opt_parameters["N"]
            * self.opt_parameters["Sh"]
            * self.opt_parameters["T"]
        )
        if nvar != len(VarNames):
            logging.error(" ******  inconsistency with number of variables")

        # -------------------------------------------------------------------------
        #         objective function and bounds
        # ------------------------------------------------------------------------

        self.LCCOption[
            np.isnan(self.LCCOption)
        ] = 0.0  # turn nans from investment costs to 0
        CostVec1a = [
            self.LCCOption[n, sh, sg] for sg in grSg for sh in grSh for n in grN
        ]  # investment costs connected to C parameter

        # Sum the risk costs over time and sum with investment costs:
        CostVec1b = [
            np.sum(self.RiskGeotechnical[n, sg, :])
            for sg in grSg
            for sh in grSh
            for n in grN
        ]  # geotechnical risk connected to G parameter
        CostVec2 = [
            self.RiskOverflow[n, sh, t] for t in grT for sh in grSh for n in grN
        ]  # risk costs of overflow connected to O parameter

        if not BudgetLimit:
            # normal version:
            lbv = np.tile(0.0, nvar)  # lower bound 0 for all variables
            ubv = np.tile(1.0, nvar)  # upper bound 1 for all variables
            typev = "I" * nvar  # all variables are integer
            CostVec1 = list(np.add(CostVec1a, CostVec1b))
            CostVec = CostVec1 + CostVec2
        else:
            # alternative with budget limit:
            VarNames = Cint + Gint + Oint
            lbv = np.tile(0.0, len(VarNames))  # lower bound 0 for all variables
            ubv = np.tile(1.0, len(VarNames))  # upper bound 1 for all variables
            typev = "I" * len(VarNames)  # all variables are integer
            CostVec = CostVec1a + CostVec1b + CostVec2

        model.variables.add(obj=CostVec, lb=lbv, ub=ubv, types=typev, names=VarNames)
        self.CostVec = CostVec
        # -------------------------------------------------------------------------
        #         implement constraints
        # ------------------------------------------------------------------------

        # define lists that form the constraints
        A = list()  # A matrix of constraints
        b = list()  # b vector (right hand side) of equations

        # constraint XX: The initial condition Cint(n,s) = 0 for all n and s in N and S. This is not a valid constraint, it is an initial condition
        # constraint 1: There should be only 1 option implemented at each dike section: sum_s(Cint(s)=1 for all n in N
        C1 = list()
        for n in grN:
            slist = Cint_nd[n, :, :].ravel().tolist()
            nlist = [1.0] * (self.opt_parameters["Sg"] * self.opt_parameters["Sh"])
            curconstraint = [slist, nlist]
            C1.append(curconstraint)
        A = A + C1
        senseV = "E" * len(C1)
        # b = b+[1.0]*self.opt_parameters['N']
        b = b + [1.0] * len(C1)

        logging.info("constraint 1 implemented")

        # constraint 2: there is only 1 weakest section for overflow at any point in time
        C2 = list()
        for t in grT:
            # slist = Dint_nd[:,:,t].tolist()
            slist = [Oint_nd[n, s, t] for n in grN for s in grSh]
            nlist = [1.0] * (self.opt_parameters["N"] * self.opt_parameters["Sh"])
            curconstraint = [slist, nlist]
            C2.append(curconstraint)
        A = A + C2
        senseV = senseV + "E" * len(C2)
        b = b + [1.0] * len(C2)
        # b = b+ [1.0]*(self.opt_parameters['N']*self.opt_parameters['S'])

        logging.info("constraint 2 implemented")
        # Add constraints to model:
        model.linear_constraints.add(lin_expr=A, senses=senseV, rhs=b)

        # constraint 3: make sure that for overflow DY represents the weakest link
        C3 = list()
        import sys

        for t in grT:
            for n in grN:
                C3 = list()
                for nst in grN:
                    for sst in grSh:
                        # derive the index of the relevant decision variables
                        index = (
                            self.Pf["Overflow"][n, :, t]
                            > self.Pf["Overflow"][nst, sst, t]
                        )
                        index1 = np.where(index)[0]
                        ii = []
                        if np.size(index1) > 0:  # select the last. WHY THE LAST?
                            ii = index1

                        jj = {}
                        # jj = (self.opt_parameters['Sh'])*np.tile(1,self.opt_parameters['N'])
                        for kk in grN:

                            index = (
                                self.Pf["Overflow"][kk, :, t]
                                <= self.Pf["Overflow"][nst, sst, t]
                            )
                            index1 = np.where(index)[0]
                            if np.size(index1) > 0:
                                jj[kk] = [index1]
                            else:
                                jj[kk] = []
                        slist = self._flatten_list(
                            [Cint_nd[n, sh, :].tolist() for sh in ii]
                        ) + self._flatten_list(
                            [Oint_nd[nh, sh, t].tolist() for nh in grN for sh in jj[nh]]
                        )
                        nlist = [1.0] * len(slist)
                        curconstraint = [slist, nlist]
                        C3.append(curconstraint)
                        del curconstraint, slist, nlist
                senseV = "L" * len(C3)
                b = [1.0] * len(C3)
                model.linear_constraints.add(lin_expr=C3, senses=senseV, rhs=b)

        # A = A + C3
        # senseV = senseV + "L"*len(C3) # L means <=
        # b = b+[1.0]*len(C3)

        logging.info("constraint 3 implemented")
        # constraint 4: If Cint = 0 OR 1 for sh, sg, n Gint should also be 0 OR 1.
        C4 = list()
        for n in grN:
            for sh in grSh:
                for sg in grSg:
                    curconstraint = [
                        [Cint_nd[n, sh, sg], Gint_nd[n, sh, sg]],
                        [1.0, -1.0],
                    ]  # [nlist, slist]
                    C4.append(curconstraint)
        senseV = "E" * len(C4)
        b = [0.0] * len(C4)
        model.linear_constraints.add(lin_expr=C4, senses=senseV, rhs=b)
        # optional constraint 5: implement a budget limit
        if BudgetLimit:
            if BudgetLimit <= 0.0:
                raise ValueError("Invalid budget limit entered!")
            C5 = list()
            slist = Cint  # Cint_nd[:,:,:].ravel().tolist()
            nlist = CostVec1a
            curconstraint = [slist, nlist]
            C5.append(curconstraint)
            senseV = "L" * len(C5)
            b = [BudgetLimit] * len(C5)
            model.linear_constraints.add(lin_expr=C5, senses=senseV, rhs=b)

        # slist = Cint_nd[:, :, :].ravel().tolist()
        # nlist = [1.0] * (self.opt_parameters['Sg'] * self.opt_parameters['Sh'])

        # # Add constraints to model:
        # model.linear_constraints.add(lin_expr=A, senses=senseV, rhs=b)

        return model

    def read_results(self, model_results, dir=False, measure_table=None):
        N = self.opt_parameters["N"]
        Sh = self.opt_parameters["Sh"]
        Sg = self.opt_parameters["Sg"]
        T = self.opt_parameters["T"]
        grN = range(N)
        grSh = range(Sh)
        grSg = range(Sg)
        grT = range(T)
        self.results = {}
        xs = model_results["Values"]
        ind = np.argwhere(np.int32(xs))
        varnames = model_results["Names"]
        Measure_ones = np.array(varnames)[ind][:-T]
        LCCTotal = 0
        sections = []
        for i in list(Measure_ones):
            if i[0][0] == "C":
                sections.append(i[0])

        measure = {}
        for i in range(0, len(sections)):
            measure[np.int_(str(sections[i])[1:4])] = [
                np.int_(str(sections[i])[4:7]),
                np.int_(str(sections[i])[7:]),
            ]

        LCCTotal = 0
        sectionnames = list(self.options.keys())
        sections = []
        measurenames = []
        yesno = []
        dcrest = []
        dberm = []
        LCC = []
        ID = []
        ID2 = []
        for i in measure.keys():
            sections.append(sectionnames[i])
            if isinstance(measure_table, pd.DataFrame):
                if np.sum(measure[i]) != 0:
                    ID.append(
                        self.options_geotechnical[sectionnames[i]]
                        .iloc[measure[i][1] - 1]["ID"]
                        .values[0]
                    )
                    if measure[i][0] != 0:
                        ID2.append(
                            self.options_height[sectionnames[i]]
                            .iloc[measure[i][0] - 1]["ID"]
                            .values[0]
                        )  # fout
                        if ID[-1][-1] != ID2[-1]:
                            ID[-1] = ID[-1] + "+" + ID2[-1]
                else:
                    ID.append("0")
                    ID2.append("0")
                    # MeasureTable.append(pd.DataFrame([['0', 'Do Nothing']],columns=['ID','Name']))

                if len(measure_table.loc[measure_table["ID"] == ID[-1]]) == 0:
                    if len(ID[-1]) > 1:
                        splitID = ID[-1].split("+")
                        newname = (
                            measure_table.loc[measure_table["ID"] == splitID[0]][
                                "Name"
                            ].values
                            + "+"
                            + measure_table.loc[measure_table["ID"] == splitID[1]][
                                "Name"
                            ].values
                        )
                        newline = pd.DataFrame(
                            [[ID[-1], newname[0]]], columns=["ID", "Name"]
                        )
                    else:
                        newline = pd.DataFrame(
                            [[ID[-1], "Do Nothing"]], columns=["ID", "Name"]
                        )
                    measure_table = measure_table.append(newline)

                measurenames.append(
                    measure_table.loc[measure_table["ID"] == ID[-1]]["Name"].values[0]
                )
            else:
                if np.sum(measure[i]) != 0:
                    measurenames.append(
                        self.options[sectionnames[i]]
                        .iloc[measure[i][1] - 1]["type"]
                        .values[0]
                    )
                else:
                    measurenames.append("Do Nothing")
            if np.sum(measure[i]) != 0:
                yesno.append(
                    self.options_geotechnical[sectionnames[i]]
                    .iloc[measure[i][1] - 1]["yes/no"]
                    .values[0]
                )
                dcrest.append(
                    self.options_height[sectionnames[i]]
                    .iloc[measure[i][0] - 1]["dcrest"]
                    .values[0]
                )
                dberm.append(
                    self.options_geotechnical[sectionnames[i]]
                    .iloc[measure[i][1] - 1]["dberm"]
                    .values[0]
                )

            else:
                yesno.append("no")
                dcrest.append(0)
                dberm.append(0)

            LCC.append(self.LCCOption[i, measure[i][0], measure[i][1]])
            LCCTotal += self.LCCOption[i, measure[i][0], measure[i][1]]

        TakenMeasures = pd.DataFrame(
            {
                "ID": ID,
                "Section": sections,
                "LCC": LCC,
                "name": measurenames,
                "yes/no": yesno,
                "dcrest": dcrest,
                "dberm": dberm,
            }
        )
        # add year
        self.results["measures"] = measure
        TakenMeasures = TakenMeasures.sort_values("Section")
        self.TakenMeasures = TakenMeasures
        data = pd.DataFrame(
            {
                "Names": model_results["Names"],
                "Values": model_results["Values"],
                "Cost": self.CostVec,
            }
        )

        pd.set_option("display.max_columns", None)  # prevents trailing elipses
        if dir:
            TakenMeasures.to_csv(dir.joinpath("TakenMeasures_MIP.csv"))
        else:
            pass
            # TakenMeasures.to_csv('TakenMeasures_MIP.csv')
        ## reproduce objective:
        alldata = data.loc[data["Values"] == 1]
        Nsections = np.int32((len(alldata) - T) / 2)
        self.results["C_int"] = alldata.iloc[0:Nsections]
        self.results["G_int"] = data.loc[data["Values"] == 1].iloc[Nsections:-T]
        self.results["O_int"] = data.loc[data["Values"] == 1].iloc[-T:]
        self.results["TC"] = np.sum(alldata)["Cost"]
        self.results["LCC"] = np.sum(alldata.iloc[0:Nsections])["Cost"]
        self.results["GeoRisk"] = np.sum(alldata.iloc[Nsections:-T])["Cost"]
        self.results["OverflowRisk"] = np.sum(alldata.iloc[-T:])["Cost"]

    def check_constraint_satisfaction(self, Model):
        N = self.opt_parameters["N"]
        S = self.opt_parameters["S"]
        T = self.opt_parameters["T"]

        grN = range(N)
        grS = range(S)
        grT = range(T)
        # -------------------------------------------------------------------------
        #         verify if constraints are satisfied
        # ------------------------------------------------------------------------

        AllConstraintsSatisfied = True

        # C1
        GG = np.tile(1.0, N)
        for n in grN:
            GG[n] = np.sum(self.results["C_int"][n, :])

        if (GG == 1).all():
            logging.info("constraint 1 satisfied")
        else:
            logging.warning("constraint 1 not satisfied")
            AllConstraintsSatisfied = False
        # C2
        GG = np.tile(1.0, T)
        for t in range(0, T):
            GG[t] = sum(sum(self.results["D_int"][:, :, t]))

        if (GG == 1).all():
            logging.info("constraint C2 satisfied")
        else:
            logging.warning("constraint C2 not satisfied")
            AllConstraintsSatisfied = False

            # C3
        pass

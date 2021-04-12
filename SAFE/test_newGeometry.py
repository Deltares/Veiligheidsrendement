
# import dill  # pip install dill --user
# filename = 'globalsave.pkl'
# # dill.dump_session(filename)
#
# # and to load the session again:
# dill.load_session(filename)
import sys
sys.path.append('..')
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import config
from shapely.geometry import Polygon


# from sympy import Polygon as Polygon2
# from Measure import calculateArea
# script to calculate the area difference of a new geometry after reinforcement (compared to old one)

def calculateArea_old(geometry):
 extra = np.empty((1, 2))
 if geometry[-1][1] > geometry[0][1]:
     extra[0, 0] = geometry[-1][0]
     extra[0, 1] = geometry[0][1]
     geometry = np.append(geometry, np.array(extra), axis=0)
 elif geometry[-1][1] < geometry[0][1]:
     extra[0, 0] = geometry[0][0]; extra[0, 1] = geometry[-1][1]
     geometry = np.insert(geometry, [0], np.array(extra), axis=0)

 bottomlevel = np.min(geometry[:, 1])
 area = 0

 for i in range(1, len(geometry)):
     a = np.abs(geometry[i-1][0] - geometry[i][0]) * (0.5 * np.abs(geometry[i - 1][1]-geometry[i][1]) + 1.0 * (np.min((geometry[i-1][1], geometry[i][1])) - bottomlevel))
     area += a

 polypoints= []

 for i in range(len(geometry)):
     polypoints.append((geometry[i, 0], geometry[i, 1]))
 polygon = Polygon(polypoints)
 return area, polygon

def calculateArea(geometry):
    polypoints = []
    for i in range(len(geometry)):
        polypoints.append((geometry[i, 0], geometry[i, 1]))
    polygonXY = Polygon(polypoints)
    areaPol = Polygon(polygonXY).area
    return areaPol, polygonXY

def addBerm(initial, geometry, new_geometry, bermheight, dberm):
    i = int(initial[initial.type == 'innertoe'].index.values)
    j = int(initial[initial.type == 'innercrest'].index.values)
    slope_inner = (geometry[j][1] - geometry[i][1]) / (geometry[j][0] - geometry[i][0])
    extra = np.empty((1, 2))
    extra[0, 0] = new_geometry[i][0] + (1 / slope_inner) * bermheight
    extra[0, 1] = new_geometry[i][1] + bermheight
    new_geometry = np.append(new_geometry, np.array(extra), axis=0)
    extra2 = np.empty((1, 2))
    extra2[0, 0] = new_geometry[i][0] + (1 / slope_inner) * bermheight + dberm
    extra2[0, 1] = new_geometry[i][1] + bermheight
    new_geometry = np.append(new_geometry, np.array(extra2), axis=0)
    new_geometry = new_geometry[new_geometry[:, 0].argsort()]

    if (initial.type == 'extra').any():
        k = int(initial[initial.type == 'extra'].index.values)
        new_geometry[0, 0] = initial.x[i]
        new_geometry[0, 1] = initial.z[i]
        extra3 = np.empty((1, 2))
        extra3[0, 0] = initial.x[k]
        extra3[0, 1] = initial.z[k]
        new_geometry = np.append(np.array(extra3), new_geometry, axis=0)
    return new_geometry

def main():
    #input:
    # new_geometry, area_difference = DetermineNewGeometry(geometry_change, direction, initial,plot_dir = None, bermheight = 2, slope_in = False)

    #input 6:
    #DV='DV01'

    #input 4:
    DV ='DV31'
    initial = pd.read_excel(r'c:\Users\krame_n\0_WERK\SAFE\Repos\data\cases\Testcase_10sections_2021_16-4\{}.xlsx'.format(DV), sheet_name='Geometry')
    direction = 'inward' # of outward
    geometry_change = [0, 10]
    bermheight = 2
    slope_in = 4
    plot_dir = config.directory.joinpath('figures')
    config.geometry_plot = 1
    maxBermOut = 20

    #script
    if len(initial) == 6:
        noberm = False
    elif len(initial) == 4:
        noberm=True
    else:
        raise Exception ('input length dike is not 4 or 6')

    # if outertoe < innertoe
    if initial.z[int(initial[initial.type == 'innertoe'].index.values)] > initial.z[int(initial[initial.type == 'outertoe'].index.values)]:
        extra_row = pd.DataFrame([[initial.x[int(initial[initial.type == 'innertoe'].index.values)],initial.z[int(initial[initial.type == 'outertoe'].index.values)], 'extra']],columns =initial.columns)
        initial = extra_row.append(initial).reset_index(drop=True)

    # Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm = geometry_change[1]
    geometry = initial.values[:,0:2]
    cur_crest = np.max(geometry[:, 1])
    new_crest = cur_crest + dcrest
    if config.geometry_plot:
        plt.plot(geometry[:, 0], geometry[:, 1], 'k')

    if direction == 'outward':
        new_geometry = copy.deepcopy(geometry)

        if dberm < maxBermOut:
            for i in range(len(new_geometry)):
                # Run over points from the outside.
                if initial.type[i] == 'extra':
                    new_geometry[i][0] = geometry[i][0]
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innertoe':
                    new_geometry[i][0] = geometry[i][0]
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innerberm1':
                    new_geometry[i][0] = geometry[i][0]
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innerberm2':
                    new_geometry[i][0] = geometry[i][0] + dberm
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innercrest':
                    new_geometry[i][0] = geometry[i][0] + dberm
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outercrest':
                    new_geometry[i][0] = geometry[i][0] + dberm
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outertoe':
                    new_geometry[i][0] = geometry[i][0] + dberm
                    new_geometry[i][1] = geometry[i][1]
        else:
            berm_in = dberm- maxBermOut
            for i in range(len(new_geometry)):
                # Run over points from the outside.
                if initial.type[i] == 'extra':
                    new_geometry[i][0] = geometry[i][0] - berm_in
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innertoe':
                    new_geometry[i][0] = geometry[i][0] - berm_in
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innerberm1':
                    new_geometry[i][0] = geometry[i][0] - berm_in
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innerberm2':
                    new_geometry[i][0] = geometry[i][0] + maxBermOut
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innercrest':
                    new_geometry[i][0] = geometry[i][0] + maxBermOut
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outercrest':
                    new_geometry[i][0] = geometry[i][0] + maxBermOut
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outertoe':
                    new_geometry[i][0] = geometry[i][0] + maxBermOut
                    new_geometry[i][1] = geometry[i][1]

        if noberm:  # len(initial) == 4:
            new_geometry = addBerm(initial, geometry, new_geometry, bermheight, dberm)

    if direction == 'inward':
        new_geometry = copy.deepcopy(geometry)
        # we start at the outer toe so reverse:

        for i in range(len(new_geometry)):
            # Run over points from the outside.
            if initial.type[i] == 'extra':
                new_geometry[i][0] = geometry[i][0] - dberm
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'innertoe':
                new_geometry[i][0] = geometry[i][0] - dberm
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'innerberm1':
                new_geometry[i][0] = geometry[i][0] - dberm
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'innerberm2':
                new_geometry[i][0] = geometry[i][0]
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'innercrest':
                new_geometry[i][0] = geometry[i][0]
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'outercrest':
                new_geometry[i][0] = geometry[i][0]
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'outertoe':
                new_geometry[i][0] = geometry[i][0]
                new_geometry[i][1] = geometry[i][1]

        if noberm: #len(initial) == 4:   #precies hetzelfde als hierboven. def van maken.
            new_geometry = addBerm(initial, geometry, new_geometry, bermheight, dberm)

    # calculate the area difference
    area_old_oud, polygon_old_oud = calculateArea_old(geometry)
    area_new_oud, polygon_new_oud = calculateArea_old(new_geometry)

    area_old, polygon_old = calculateArea(geometry)
    area_new, polygon_new = calculateArea(new_geometry)

    if (polygon_old.intersects(polygon_new)):  # True
        poly_intsects = polygon_old.intersection(polygon_new)
        area_intersect = (polygon_old.intersection(polygon_new).area)  # 1.0
        area_excavate = area_old - area_intersect
        area_extra = area_new - area_intersect

        #difference new-old = extra
        poly_diff = polygon_new.difference(polygon_old)
        area_difference = poly_diff.area  #zou zelfde moeten zijn als area_extra
        # difference new-old = excavate
        poly_diff2 = polygon_old.difference(polygon_new)
        area_difference2 = poly_diff2.area  #zou zelfde moeten zijn als area_excavate

        #controle
        test1 = area_difference - area_extra
        test2 = area_difference2 - area_excavate
        # test3 = area_intersect + area_extra - area_excavate
        if test1>1 or test2 >1:
            raise Exception ('area calculation failed')


        if config.geometry_plot:
            plt.plot(geometry[:, 0], geometry[:, 1], 'k')
            plt.plot(new_geometry[:, 0], new_geometry[:, 1], '--r')
            if poly_diff.area > 0:
                if hasattr(poly_diff, 'geoms'):
                    for i in range(len(poly_diff.geoms)):
                        x1, y1 = poly_diff[i].exterior.xy
                        plt.fill(x1, y1, 'r--', alpha=.1)
                else:
                    x1, y1 = poly_diff.exterior.xy
                    plt.fill(x1, y1, 'r--', alpha=.1)
            if poly_diff2.area > 0:
                if hasattr(poly_diff2, 'geoms'):
                    for i in range(len(poly_diff2.geoms)):
                        x1, y1 = poly_diff2[i].exterior.xy
                        plt.fill(x1, y1, 'b--', alpha=.1)
                else:
                    x1, y1 = poly_diff2.exterior.xy
                    plt.fill(x1, y1, 'b--', alpha=.1)
            #
            # if hasattr(poly_intsects, 'geoms'):
            #     for i in range(len(poly_intsects.geoms)):
            #         x1, y1 = poly_intsects[i].exterior.xy
            #         plt.fill(x1, y1, 'g--', alpha=.1)
            # else:
            #     x1, y1 = poly_intsects.exterior.xy
            #     plt.fill(x1, y1, 'g--', alpha=.1)
            plt.show()

        # plt.text(np.mean(new_geometry[:, 0]), np.max(new_geometry[:, 1]),
        #          'Area difference = ' + '{:.4}'.format(str(area_difference)) + ' $m^2$')

        plt.savefig(plot_dir.joinpath('Geometry_' + str(dberm) + '_' + str(dcrest) + '.png'))
        plt.close()

    print(area_difference)
    print(new_geometry)
    return area_excavate, area_extra




if __name__ == '__main__':
    main()
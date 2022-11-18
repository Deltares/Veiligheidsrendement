'''This is a first attempt to properly generate a shapefile with all required data'''
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from shapely import geometry, ops
import warnings

def check_traject_shape(traject_name, shape):
    #check 1: ensure that gdb has an entry 'TRAJECTID', and that there is only 1 traject included.
    if 'TRAJECT_ID' not in shape.columns: raise Exception('Traject shape does not have field TRAJECT_ID')
    if shape['TRAJECT_ID'].unique().shape[0] != 1: raise Exception('Traject shape does not have unique TRAJECT_ID')
    TRAJECT_ID = shape['TRAJECT_ID'].unique()
    crs = shape.crs

    #check 2: ensure that it is the correct traject (TRAJECTID)
    if TRAJECT_ID[0] != traject_name: raise Exception('Traject shape does not match traject_name: shape has {}, while it should be {}'.format(TRAJECT_ID[0],traject_name))
    #STEP 3: generate right shape: ensure that it is a single traject if not, merge all sections.
    if shape.shape[0] == 1:
        #simplify:
        updated_shape = shape[['TRAJECT_ID']]
        pass
    else:
        print('Multiple sections detected in traject shape, welding to 1 single traject')
        new_geometry = ops.linemerge(geometry.MultiLineString(list(shape.geometry)))
        updated_shape = gpd.GeoDataFrame(data= TRAJECT_ID,columns=['TRAJECT_ID'],geometry=[new_geometry])
        #what if parts are missing?
    updated_shape.crs = crs
    return updated_shape
def check_vakindeling_algemeen(df_vakindeling,traject_length):
    #checks the integrity of the vakindeling as inputted.
    #cuts of any obsolete columns:
    idx_last = np.argwhere(df_vakindeling.columns.str.contains('Unnamed')).min()
    df_vakindeling = df_vakindeling[df_vakindeling.columns[0:idx_last]]

    #check if OBJECTID and NUMMER have unique values
    if any(df_vakindeling['OBJECTID'].duplicated()): raise('values in OBJECTID are not unique')
    if any(df_vakindeling['NUMMER'].duplicated()): raise('values in NUMMER are not unique')

    #sort the df by OBJECTID:
    df_vakindeling = df_vakindeling.sort_values('OBJECTID')

    #check the references to the mechanisms.
    # If they are complete: all is good
    # If they are partially completed: raise a CLEAR warning
    # If nothing is filled in, also raise a warning
    for mechanism in df_vakindeling.columns[df_vakindeling.columns.str.contains('DOORSNEDE')]:
        if all(df_vakindeling[mechanism].isna()):
            warnings.warn('WARNING: No mechanism data for {}'.format(mechanism.split('_')[1]),ImportWarning)
        elif any(df_vakindeling[mechanism].isna()):
            warnings.warn('WARNING: Some sections do not have input for {}'.format(mechanism.split('_')[1]),ImportWarning)
        else:
            pass

    df_vakindeling = df_vakindeling.fillna(value=np.nan)
    #check if, if sorted on OBJECTID MEAS_START and MEAS_END are increasing
    if any(np.diff(df_vakindeling.MEAS_END)<0): raise ValueError('MEAS_END values are not always increasing if sorted on OBJECTID')
    if any(np.diff(df_vakindeling.MEAS_START)<0): raise ValueError('MEAS_START values are not always increasing if sorted on OBJECTID')
    if any(np.subtract(df_vakindeling.MEAS_END,df_vakindeling.MEAS_START)<0.): raise ValueError('MEAS_START is higher than MEAS_END for at least 1 section')

    #check if MEAS_END.max() is approx equal to traject_length
    np.testing.assert_approx_equal(df_vakindeling.MEAS_END.max(), traject_length, significant=5)

    print()
    return df_vakindeling

def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [geometry.LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(geometry.Point(p))
        if pd == distance:
            return [
                geometry.LineString(coords[:i+1]),
                geometry.LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                geometry.LineString(coords[:i] + [(cp.x, cp.y)]),
                geometry.LineString([(cp.x, cp.y)] + coords[i:])]
        
def generate_vakindeling_shape(traject_shape,df_vakken):
    traject_geom = traject_shape.geometry[0]
    section_geom = []
    total_dist = 0.
    for count, row in df_vakken.iterrows():
        section,traject_geom = cut(traject_geom,row['MEAS_END']-total_dist)
        section_geom.append(section)
        total_dist += section.length
        #sanity check. VAKLENGTE column and section length should be almost equal
        np.testing.assert_approx_equal(section.length,row['VAKLENGTE'],significant=5)

    return gpd.GeoDataFrame(df_vakken,geometry = section_geom)

def main(traject_name, traject_shape_path,vakindeling_tabel_path):
    #first read the shape that will be used as a basis
    traject_shape = gpd.read_file(traject_shape_path)

    #check the correctness of the file
    traject_shape = check_traject_shape(traject_name,traject_shape)
    traject_length = traject_shape.geometry[0].length

    #save traject_shape
    traject_shape.to_file(traject_shape_path.parent.joinpath('ModifiedTrajectShape.shp'))
    #read the vakindeling:
    vakindeling_df = pd.read_excel(vakindeling_tabel_path,sheet_name='Algemeen',
                                         dtype = {'OBJECTID':np.int32, 'NUMMER':str,'NAAM':str,
                                                       'STBI_DOORSNEDE':str,'STPH_DOORSNEDE':str,'GEKB_DOORSNEDE':str,
                                                  'VAN_DP':str,'TOT_DP':str,'MEAS_START':np.float32,'MEAS_END':np.float32,
                                                  'VAKLENGTE':np.float32,'IN_ANALYSE':bool})

    vakindeling_df = check_vakindeling_algemeen(vakindeling_df, traject_length)
    #modify the traject shape such that it represents the vakindeling properly
    vakindeling_gdf = generate_vakindeling_shape(traject_shape,vakindeling_df)
    vakindeling_gdf.to_file(vakindeling_tabel_path.parent.joinpath('VakindelingVR.shp'))
    #read the mechanism data for each entry with DOORSNEDE_mechanism:

    #read data for different mechanisms
    print()

if __name__ == '__main__':
    working_dir = Path(r'n:\Projects\11208000\11208392\C. Report - advise\Invoer & vakindeling\Invoer & vakindeling\test_38_1')
    traject_shape_path = working_dir.joinpath('Basisvakindeling_HB_38_1_20210708.shp')
    vakindeling_tabel_path = working_dir.joinpath('Vakindeling 38-1.xlsx')
    traject_name = '38-1'
    main(traject_name,traject_shape_path,vakindeling_tabel_path)
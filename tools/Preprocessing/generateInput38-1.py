'''This is a first attempt to properly generate a shapefile with all required data'''
import geopandas as gpd
import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from shapely import geometry, ops
import warnings
import sqlite3
import glob
import re
from owslib.wfs import WebFeatureService
import shutil
import copy
sys.path.append('../../src')
from flood_defence_system.DikeSection import DikeProfile


def get_traject_shape_from_NBPW(traject,NBWP_shape_path=False):
    if not NBWP_shape_path:
        #get from WFS
        wfs_nbpw = WebFeatureService(url='https://waterveiligheidsportaal.nl/geoserver/nbpw/ows/wfs', version='1.1.0')
        NBPW = gpd.read_file(wfs_nbpw.getfeature('nbpw:dijktrajecten',outputFormat='json'))
    else:
        NBPW = gpd.read_file(NBWP_shape_path)

    traject_shape = NBPW.loc[NBPW.TRAJECT_ID==traject].reset_index(drop=True)
    return traject_shape


def check_traject_shape(traject_name, shape):
    #this entire routine can be removed? As we use the shape from NBWP now.

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
        updated_shape = shape[['TRAJECT_ID','geometry']]
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
def findWeakestGEKB(vakindeling, GEKB_shape,db_dir = Path(r'n:\Projects\11208000\11208392\C. Report - advise\Invoer & vakindeling\Invoer & vakindeling\test_38_1')):
    indices = []
    if len(GEKB_shape.TRAJECT_ID.unique())>1: raise Exception('Shape contains multiple trajects')
    traject_id = GEKB_shape.TRAJECT_ID.unique()[0]
    #first get middle of sections (we assume the cs is at that spot)
    GEKB_shape['M_MID'] = np.average(np.vstack([GEKB_shape.M_VAN.values,GEKB_shape.M_TOT.values]).T,axis=1)
    for count, vak in vakindeling.iterrows():
        #get subset of GEKB locations
        subset = GEKB_shape.loc[(GEKB_shape.M_MID > vak.MEAS_START) & (GEKB_shape.M_MID < vak.MEAS_END)]

        #grab line with highest Faalkans
        line = subset[subset.Faalkans == subset.Faalkans.max()]
        #if there is no line, grab the one closest to the midpoint of vak
        if line.shape[0] == 0:
            vak_mid = np.mean([vak.MEAS_START, vak.MEAS_END])
            distance = np.abs(np.subtract(GEKB_shape.M_MID, vak_mid))
            idx = np.argmin(distance)
            indices.append(idx)
        else:
            if line.shape[0] != 1:
                warnings.warn('Multiple cross sections have highest P_f for section {}'.format(vak.NUMMER))
            #write index to list
            indices.append(line.index.values[0])
    #get subset of GEKB_shape based on indexlist
    GEKB_selectie = GEKB_shape.iloc[indices].reset_index().rename(columns={'index':'GEKB_id'})
    #based on HR koppel, find the ID of the locations:
    #first we load the HRD Locations table
    # cnx = sqlite3.connect(r'n:\Projects\11208000\11208392\C. Report - advise\Invoer & vakindeling\Invoer & vakindeling\test_38_1\WBI2017_Bovenrijn_38-1_v04.sqlite')
    cnx = sqlite3.connect(db_dir.joinpath('WBI2017_Bovenrijn_38-1_v04.sqlite'))
    #TODO make cnx flexible
    locs = pd.read_sql_query("SELECT * FROM HRDLocations", cnx)
    loc_ids = []
    for count, line in GEKB_selectie.iterrows():
        loc_ids.append(locs.loc[locs['Name'] == line['HR koppel']]['HRDLocationId'].values[0])
    GEKB_selectie['LocationID'] = loc_ids

    #NUMMER & OBJECTID should be identical:
    GEKB_selectie['NUMMER'] = vakindeling['NUMMER']
    GEKB_selectie['OBJECTID'] = vakindeling['OBJECTID']

    #write right DOORSNEDE to vakindeling:
    vakindeling['DSN_GEKB'] = GEKB_selectie['Vaknaam']

    return GEKB_selectie, vakindeling
def write_df_to_csv(df,path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def dp_as_number(dijkpaal):
    #converts dijkpaal to number (format should be like DP012.+123)
    string_splitted = re.findall(r'\d+', dijkpaal)
    dp_num = np.float64(string_splitted[0]) * 100 + np.float64(string_splitted[1])
    return dp_num
def check_STBI_data(data):
    #TODO ensure that always an SF and a BETA are given as input.
    raise Exception('Develop the STBI check to avoid nans in SF and BETA')
    pass
def write_intermediate_data(traject_name,vakindeling_tabel_path,clear_data = False):
    #this part writes all original data to a more or less fixed format (to be more fixed in the future). This is the really tailo
    input_dir = vakindeling_tabel_path.parent    #input directory
    intermediate_dir = input_dir.parent.joinpath('intermediate')
    output_dir = input_dir.parent.joinpath('output')

    if clear_data:
        shutil.rmtree(intermediate_dir)
        intermediate_dir.mkdir(parents=True)
        intermediate_dir.joinpath('Profielen').mkdir()
    #first read the shape that will be used as a basis
    traject_shape = get_traject_shape_from_NBPW('38-1',NBWP_shape_path = Path(r'c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Prioritering WSRL\Gegevens 38-1\BasisbestandWaterkeringen\dijktrajecten.shp'))

    #check the correctness of the file TODO obsolete or not?
    traject_shape = check_traject_shape(traject_name,traject_shape)
    traject_length = traject_shape.geometry.length.values[0]

    #save traject_shape
    traject_shape.to_file(intermediate_dir.joinpath('Traject {}.shp'.format(traject_name)))
    #read the vakindeling:
    vakindeling_df = pd.read_excel(vakindeling_tabel_path,sheet_name='Algemeen',
                                         dtype = {'OBJECTID':np.int32, 'NUMMER':str,'NAAM':str,
                                                       'STBI_DOORSNEDE':str,'STPH_DOORSNEDE':str,'GEKB_DOORSNEDE':str,
                                                  'VAN_DP':str,'TOT_DP':str,'MEAS_START':np.float32,'MEAS_END':np.float32,
                                                  'VAKLENGTE':np.float32,'IN_ANALYSE':bool})

    vakindeling_df = check_vakindeling_algemeen(vakindeling_df, traject_length)
    #modify the traject shape such that it represents the vakindeling properly
    vakindeling_gdf = generate_vakindeling_shape(traject_shape,vakindeling_df)

    #read the mechanism data for each entry with DSN_mechanism:

    #first for GEKB, we use the generic xlsx used for the assessment at WSRL. We have a (broad) selection fo relevant columns, and read the Excel for that:
    relevante_kolommen = ['dijkvak', 'nr', 'HR koppel', 'overslagdebiet', 'prfl_bestandnaam', 'prfl_oriëntatie',
                          'prfl_dijkhoogte', 'zode_binnendijks_copy', 'gekb_situatie', 'Column171', 'Column182','h_1:600.000', 'h_1:10_rtd'
]

    #read properties from Excel Bijlage
    raw_GEKB_data = pd.read_excel(
        vakindeling_tabel_path.parent.joinpath('Overslag','Bijlage A DATA-HB_TRAJECT-38-1_v20211117.xlsx'),
        sheet_name='Per dijkpaal', header=15, usecols=relevante_kolommen)
    #renaming some unclear headers:
    raw_GEKB_data = raw_GEKB_data.rename(columns={'Column171': 'Golfhoogteklasse', 'Column182': 'Faalkans'})
    #replace non-numeric values (NVT means not applicable so Pf=0)
    raw_GEKB_data['Faalkans'] = raw_GEKB_data['Faalkans'].replace({'NVT': 0.})
    raw_GEKB_data['Faalkans'] = np.divide(1., raw_GEKB_data['Faalkans'].astype(np.float64))
    raw_GEKB_data['Kruindaling'] = 0.005 #TODO read from file
    #select weakest cross section
    #We load the vakindeling shape for the GEKB computations. Coincidentally this is also the traject_shape. We merge the data to it.
    if len(glob.glob(str(input_dir.joinpath('Overslag', 'Vakindeling', '*.shp'))))>1: raise Exception('Multiple shapefiles for GEKB detected.')
    gekb_shp_path = Path(glob.glob(str(input_dir.joinpath('Overslag', 'Vakindeling', '*.shp')))[0])
    GEKB_shape = gpd.read_file(gekb_shp_path)
    GEKB_shape = GEKB_shape.merge(raw_GEKB_data, left_on='Vaknaam', right_on='dijkvak').sort_values('M_VAN').drop('OBJECTID', axis=1).reset_index(drop=True).reset_index().rename(columns={'index': 'OBJECTID'})    #now we select the cross sections to obtain a filtered dataset
    GEKB_data, vakindeling_gdf = findWeakestGEKB(vakindeling_gdf,GEKB_shape,db_dir = input_dir.joinpath('Overslag'))

    #write to GEKB data file (if it does not exist in input, else compare the file
    #check if file in input:
    if input_dir.joinpath('Overslag','GEKB_data.csv').exists():
        old_GEKB_data = pd.read_csv(input_dir.joinpath('Overslag','GEKB_data.csv'),index_col=0)
        #compare:
        pd.testing.assert_frame_equal(old_GEKB_data, GEKB_data[['Vaknaam', 'prfl_bestandnaam', 'prfl_oriëntatie', 'prfl_dijkhoogte', 'gekb_situatie', 'Golfhoogteklasse','LocationID', 'Faalkans','Kruindaling']])
        GEKB_data_reduced = old_GEKB_data
    else:
        GEKB_data_reduced = GEKB_data[['Vaknaam', 'prfl_bestandnaam', 'prfl_oriëntatie', 'prfl_dijkhoogte', 'gekb_situatie', 'Golfhoogteklasse','LocationID', 'Faalkans','Kruindaling']]
        GEKB_data_reduced.to_csv(input_dir.joinpath('Overslag','GEKB_data.csv'))
    #Piping data
    Piping_data = pd.read_csv(input_dir.joinpath('Piping','PipingInput.csv'))
    #TODO coupling based on M-value (more robust)
    vakindeling_gdf['DSN_STPH'] = Piping_data.dwarsprofiel.unique()

    #STBI data
    STBI_data = pd.read_excel(vakindeling_tabel_path,sheet_name='STBI')[['DOORSNEDE','BEREKENING','SCENARIOKANS','SF','BETA','STIX']]
    STBI_data = check_STBI_data(STBI_data)
    #profielen
    raw_profile_data = pd.read_excel(input_dir.joinpath('Dijkprofielen','GEOMETRY_TEMPLATE_v3_38_1_STBI.xlsx'),sheet_name='KARAKTERISTIEKE_PUNTEN',header=9)
    #TODO improve CS_NUM (e.g. by a table that reference DP and M_value)
    raw_profile_data['CS_NUM'] = raw_profile_data['ID_GEOMETRY_LOCATION'].map(lambda x: dp_as_number(x))
    profiles = []
    for count, vak in vakindeling_gdf.iterrows():
        if vak.IN_ANALYSE:
            #ALS STBI ID in lijst met profielen pak dat profiel
            if any(raw_profile_data.ID_GEOMETRY_LOCATION.isin([vak.DSN_STBI])):
                if len(raw_profile_data.loc[raw_profile_data.ID_GEOMETRY_LOCATION == vak.DSN_STBI]['ID_GEOMETRIE']) >1: raise Exception('Multiple profiles found')
                profiles.append(raw_profile_data.loc[raw_profile_data.ID_GEOMETRY_LOCATION == vak.DSN_STBI]['ID_GEOMETRIE'].values[0])

            #NB: dit is een custom stap die weg kan als we CS_NUM goed hebben
            elif any(raw_profile_data.ID_GEOMETRY_LOCATION.isin([vak.DSN_STBI[:-2]])):
                if len(raw_profile_data.loc[raw_profile_data.ID_GEOMETRY_LOCATION == vak.DSN_STBI]['ID_GEOMETRIE'][:-2]) >1: raise Exception('Multiple profiles found')
                profiles.append(raw_profile_data.loc[raw_profile_data.ID_GEOMETRY_LOCATION == vak.DSN_STBI[:-2]]['ID_GEOMETRIE'].values[0])

            #ANDERS: Zoek op basis van getalswaarde
            else:
                warnings.warn('Voor dijkvak {} is geen profiel gevonden, beste schatting is gebruikt'.format(vak.NUMMER))
                options = raw_profile_data.loc[(raw_profile_data.CS_NUM > vak.MEAS_START) & (raw_profile_data.CS_NUM < vak.MEAS_END)]
                if len(options) > 1:
                    raise Exception('Meerdere mogelijke profielen voor dijkvak {}'.format(vak.NUMMER))
                elif len(options) == 0:
                    midpoint = np.mean([vak.MEAS_START,vak.MEAS_END])
                    idx_min = np.argmin(np.abs(np.subtract(raw_profile_data.CS_NUM,midpoint)))
                    profiles.append(raw_profile_data.reset_index(drop=True).iloc[idx_min]['ID_GEOMETRIE'])
                    warnings.warn('Geen geschikt profiel gekozen. Dichtsbijzijnde profiel in ander vak genomen ({})'.format(profiles[-1]))
                else:
                    profiles.append(options['ID_GEOMETRIE'].values[0])
        else:
            profiles.append(np.nan)
    vakindeling_gdf['DIJKPROFIEL'] = profiles
    # char_points = ['BUT', 'BUK', 'BIK', 'BBB', 'EBB', 'BIT']
    char_points = ['BUT', 'BUK', 'BIK', 'BBL', 'EBL', 'BIT']
    #TODO add buitenberm (8 point profile)
    unique_profiles = vakindeling_gdf['DIJKPROFIEL'].dropna().unique()
    for profile in unique_profiles:
        xz = []
        profile_data = raw_profile_data.loc[raw_profile_data.ID_GEOMETRIE==profile]
        dike_profile = DikeProfile(name = profile)
        for char_point in char_points:
            if char_point == 'BUT':     #custom statement, has to be removed later on.
                if np.isnan(profile_data['GEO_KAR_X_BBB'].item()): #geen buitenberm
                    xz.append((profile_data['GEO_KAR_X_{}'.format(char_point)].item(), profile_data['GEO_KAR_Z_{}'.format(char_point)].item()))
                else:
                    xz.append((profile_data['GEO_KAR_X_BBB'].item(), profile_data['GEO_KAR_Z_BBB'].item()))
            else:
                xz.append((profile_data['GEO_KAR_X_{}'.format(char_point)].item(), profile_data['GEO_KAR_Z_{}'.format(char_point)].item()))
            dike_profile.add_point(char_point,xz[-1])
        dike_profile.to_csv(intermediate_dir.joinpath('Profielen'))

    #lees bebouwing:
    bebouwing = pd.read_excel(vakindeling_tabel_path,sheet_name='Bebouwing')

    #copy files that are not modified:
    #Maatregelen
    shutil.copytree(input_dir.joinpath('Maatregelen'),intermediate_dir.joinpath('Maatregelen'))

    #Overslag/Resultaten
    shutil.copytree(input_dir.joinpath('Overslag','Resultaten_W2100'),intermediate_dir.joinpath('Overslag','Resultaten_W2100'))
    shutil.copytree(input_dir.joinpath('Overslag','Resultaten_WBI2017'),intermediate_dir.joinpath('Overslag','Resultaten_WBI2017'))

    #Waterstand/Resultaten
    shutil.copytree(input_dir.joinpath('Waterstand','Resultaten_W2100'),intermediate_dir.joinpath('Waterstand','Resultaten_W2100'))
    shutil.copytree(input_dir.joinpath('Waterstand','Resultaten_WBI2017'),intermediate_dir.joinpath('Waterstand','Resultaten_WBI2017'))

    #Data voor mechanismen:
    write_df_to_csv(GEKB_data_reduced,intermediate_dir.joinpath('Overslag','GEKB_data.csv'))
    write_df_to_csv(Piping_data,intermediate_dir.joinpath('Piping','Piping_data.csv'))
    write_df_to_csv(STBI_data,intermediate_dir.joinpath('STBI','STBI_data.csv'))

    #Bebouwing
    write_df_to_csv(bebouwing,intermediate_dir.joinpath('Bebouwing','Bebouwing_data.csv'))

    #Vakindeling:
    vakindeling_gdf.to_file(intermediate_dir.joinpath('Vakindeling_Veiligheidsrendement.shp'))

def write_dike_section_files_legacy(working_dir, vakindeling_gdf, housing_data, measureset_data, gekb_data):
    '''This routine translates /intermediate to input data that is mostly in accordance with the old format. To be replaced by the new data model.'''
    vakindeling_gdf = vakindeling_gdf.loc[vakindeling_gdf.IN_ANALYSE == 1]
    for count, section in vakindeling_gdf.iterrows():
        #write generic file for section:

        #General tab:
        general_name = ['Length','Start','End','Overflow', 'StabilityInner','Piping','Load_2025','Load_2100','Kruindaling','Kruinhoogte','SHP']
        general_type = [''] *3 + ['HRING', 'Simple', 'SemiProb'] + ['']*5

        general_values = [section['VAKLENGTE'], section['VAN_DP'], section['TOT_DP'],
                          section['DSN_GEKB'], section['DSN_STBI'], section['DSN_STPH'],
                          'Waterstand_' + section['DSN_GEKB'],'Waterstand_' + section['DSN_GEKB'],
                          gekb_data.loc[gekb_data['Vaknaam']==section['DSN_GEKB']]['Kruindaling'].item(),
                          gekb_data.loc[gekb_data['Vaknaam'] == section['DSN_GEKB']]['prfl_dijkhoogte'].item(),
                          section['geometry']]

        General = pd.DataFrame(general_values,index=general_name,columns=['Value'])
        General['Type'] = general_type

        #Measures tab
        Measures = copy.deepcopy(measureset_data)

        #DikeProfile
        DikeProfile = pd.read_csv(working_dir.joinpath('intermediate','Profielen',section['DIJKPROFIE'] + '.csv'),index_col=0)

        #Housing
        Housing = housing_data.loc[housing_data.OBJECTID == section.OBJECTID].drop(columns=['OBJECTID', 'NUMMER']).transpose().reset_index()
        Housing.columns = ['distancefromtoe','number']
        if len(section['NUMMER'])==1:
            # if replace_files & working_dir.joinpath('output','DV{:02d}.xlsx'.format(np.int32(section['NUMMER']))).exists():
            #     os.remove(working_dir.joinpath('output','DV{:02d}.xlsx'.format(np.int32(section['NUMMER']))))
            writer = pd.ExcelWriter(working_dir.joinpath('output','DV{:02d}.xlsx'.format(np.int32(section['NUMMER']))))
        else:
            # if replace_files & working_dir.joinpath('output','DV{}.xlsx'.format(section['NUMMER'])).exists():
            #     os.remove(working_dir.joinpath('output','DV{}.xlsx'.format(section['NUMMER'])))
            writer = pd.ExcelWriter(working_dir.joinpath('output','DV{}.xlsx'.format(section['NUMMER'])))

        General.to_excel(writer, sheet_name='General', index=True)
        Measures.to_excel(writer, sheet_name='Measures', index=False)
        DikeProfile.to_excel(writer, sheet_name='Geometry', index=True)
        Housing.to_excel(writer, sheet_name='Housing', index=False)
        writer.save()
def write_mechanism_data_legacy(working_dir, vakindeling_gdf, housing_data, measureset_data, piping_data, stbi_data):


    # make subfolders for Overflow and Toetspeil:
    shutil.rmtree(working_dir.joinpath('output', 'Overflow'))
    shutil.rmtree(working_dir.joinpath('output', 'Waterstand'))
    working_dir.joinpath('output', 'Overflow', '2025').mkdir(parents=True)
    working_dir.joinpath('output', 'Overflow', '2100').mkdir(parents=True)
    working_dir.joinpath('output', 'Waterstand', '2025').mkdir(parents=True)
    working_dir.joinpath('output', 'Waterstand', '2100').mkdir(parents=True)


    vakindeling_gdf = vakindeling_gdf.loc[vakindeling_gdf.IN_ANALYSE == 1]
    for count, section in vakindeling_gdf.iterrows():
        #STBI
        stbi_data.set_index('DOORSNEDE').loc[section.DSN_STBI].to_csv(working_dir.joinpath('output', 'StabilityInner', section.DSN_STBI + '.csv'), header=False)

        #Piping
        piping_data.set_index('dwarsprofiel').loc[section.DSN_STPH].drop(columns=['Naam traject']).transpose().to_csv(working_dir.joinpath('output', 'Piping', str(section.DSN_STPH) + '.csv'), header=False)



        #Overflow
        shutil.copyfile(working_dir.joinpath('intermediate','Overslag','Resultaten_W2100',section.DSN_GEKB,'designTable.txt'),working_dir.joinpath('output','Overflow','2100',section.DSN_GEKB + '.txt'))
        shutil.copyfile(working_dir.joinpath('intermediate','Overslag','Resultaten_WBI2017',section.DSN_GEKB,'designTable.txt'),working_dir.joinpath('output','Overflow','2025',section.DSN_GEKB + '.txt'))
        shutil.copyfile(working_dir.joinpath('intermediate','Waterstand','Resultaten_W2100',section.DSN_GEKB,'DESIGNTABLE_{}.txt'.format(section.DSN_GEKB)),working_dir.joinpath('output','Waterstand','2100', 'Waterstand_{}.txt'.format(section.DSN_GEKB)))
        shutil.copyfile(working_dir.joinpath('intermediate','Waterstand','Resultaten_WBI2017',section.DSN_GEKB,'DESIGNTABLE_{}.txt'.format(section.DSN_GEKB)),working_dir.joinpath('output','Waterstand','2025','Waterstand_{}.txt'.format(section.DSN_GEKB)))
        #Toetspeil

def write_tool_input_legacy(working_dir):
    '''Script to translate intermediate input to the legacy file structure (with some additions for geospatial plotting and analysis)'''

    #basic data on sections and mechanisms:
    vakindeling_gdf = gpd.read_file(working_dir.joinpath('intermediate','Vakindeling_Veiligheidsrendement.shp'))
    housing_data = pd.read_csv(working_dir.joinpath('intermediate','Bebouwing','Bebouwing_data.csv'),index_col=0)
    #NB: no customs here, should be extended.
    measureset_data = pd.read_csv(working_dir.joinpath('intermediate','Maatregelen','base_measures.csv'), delimiter=',')

    piping_data = pd.read_csv(working_dir.joinpath('intermediate','Piping','Piping_data.csv'),index_col=0)
    stbi_data = pd.read_csv(working_dir.joinpath('intermediate','STBI','STBI_data.csv'),index_col=0)
    gekb_data = pd.read_csv(working_dir.joinpath('intermediate','Overslag','GEKB_data.csv'),index_col=0)

    #Make subfolders if not exist:
    if not working_dir.joinpath('output','StabilityInner').is_dir():
        working_dir.joinpath('output','StabilityInner').mkdir(parents=True, exist_ok=True)
        working_dir.joinpath('output','Piping').mkdir(parents=True, exist_ok=True)
        working_dir.joinpath('output','Overflow').mkdir(parents=True, exist_ok=True)
        working_dir.joinpath('output','Toetspeil').mkdir(parents=True, exist_ok=True)
        working_dir.joinpath('output','Measures').mkdir(parents=True, exist_ok=True)

    #write section files:
    write_dike_section_files_legacy(working_dir, vakindeling_gdf, housing_data, measureset_data, gekb_data)

    write_mechanism_data_legacy(working_dir, vakindeling_gdf, housing_data, measureset_data, piping_data, stbi_data)

if __name__ == '__main__':
    #this code generates all the input for traject 38-1

    #set working directory:
    working_dir = Path(r'c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Prioritering WSRL\Gegevens 38-1')
    # working_dir = Path(r'n:\Projects\11208000\11208392\C. Report - advise\Invoer & vakindeling\Invoer & vakindeling\test_38_1')
    vakindeling_tabel_path = working_dir.joinpath('input','Vakindeling 38-1.xlsx')
    traject_name = '38-1'

    #write to intermediate data format
    write_intermediate_data(traject_name,vakindeling_tabel_path,clear_data = True)

    #here we could implement a bunch of checks to see if all the proper data is there

    #write intermediate data to tool input (legacy version)
    write_tool_input_legacy(working_dir)

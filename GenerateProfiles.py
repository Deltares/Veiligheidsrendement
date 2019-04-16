from DikeClasses import extractProfile
import pandas as pd
pathname = r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\Local\GIS\dwp_wouter'
traject= '16_3'
#read the dbf xls
trajectdata = pd.read_excel(pathname + '\\shapes\\' + traject + '.xls')
#read the ahnfile
ahnprofiles = pd.read_excel(pathname + '\\xz\\' + traject + '_ahn3_output.xlsx')
#run extractprofiles
for i in trajectdata['Dijkpaal']:
    profielnummer = trajectdata.loc[trajectdata['Dijkpaal'] == i]['profielnr'].values[0]
    points = ahnprofiles.loc[ahnprofiles['profielnr'] == profielnummer]
    points = points.reset_index(drop=True)
    reducedProfile = extractProfile(points,window=10,titel = i,path = pathname + '\\output')
    reducedProfile.to_csv(pathname + '\\output\\' + i + '.csv')
#save plot with name of dike section in DBF
#save csv with profile points

def extractProfile(profile,window=5,titel='Profile',path = None):
    profile_averaged = copy.deepcopy(profile)
    profile_averaged['z'] = pd.rolling_mean(profile['z'], window=window, center=True, min_periods=1)

    profile_cov = copy.deepcopy(profile_averaged)
    profile_cov['z'] = pd.rolling_cov(profile['z'], window=window, center=True, min_periods=1)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax3.set_xlim(100, 170)
    ax1.plot(profile['x'], profile['z'])
    ax1.plot(profile_averaged['x'], profile_averaged['z'], 'r')

    ax2.plot(profile_cov['x'], profile_cov['z'])

    # charpoints = [121, 128, 133, 143, 151, 161]
    # for i in charpoints:
    #     ax1.axvline(x=i, LineStyle='--', Color='k')
    #     ax2.axvline(x=i, LineStyle='--', Color='k')
        # ax3.axvline(x=i, LineStyle='--', Color='k')
    d1_peaks = np.argwhere(profile_cov['z'] > 0.02).flatten()
    ax1.plot(profile_averaged['x'].iloc[d1_peaks], profile_averaged['z'].iloc[d1_peaks], 'or')
    d1_peaks_diff = np.diff(d1_peaks)
    x = []
    takenext = 0

    #This finds crests with about 100% reliability. Rest is crap.
    for i in range(0, len(d1_peaks_diff)):
        if i == 0 or d1_peaks_diff[i] > 3 or takenext == 1:
            if takenext == 1:
                if np.absolute([d1_peaks[i]- d1_peaks[i - 1]]) < 20 and np.absolute([d1_peaks[i]- d1_peaks[i - 1]]) > 3:
                    x.append(d1_peaks[i])
                    takenext = 0
                else:
                    takenext = 0
            else:
                x.append(d1_peaks[i])
                takenext = 1
    # ax2.plot(profile_averaged['x'].iloc[x], np.ones((len(x), 1)), 'xb')
    # for i in x:
    #     ax1.axvline(x=profile_averaged['x'].iloc[i], LineStyle='--', Color='k')
    # if len(x) > 6:
    #     #selecteer de juiste punten
    # elif len (x) < 4:
    #     #not enough points found
    # else:
    #select two highest points to find crest
    #crest:
    crest = profile_averaged.iloc[x].nlargest(2,'z')
    ind_crest = crest.index.values
    x_crest = crest['x'].values
    z_crest = np.average(crest['z'].values)

    inwardside = profile_averaged.iloc[0:np.min(ind_crest)]
    toe_in_est_idx = np.int(np.round(3*z_crest/0.5))
    mean_in = np.median(inwardside['z'].iloc[-50-toe_in_est_idx:-toe_in_est_idx])
    inwardside = inwardside.iloc[-3*toe_in_est_idx:]
    inward_cov = pd.rolling_cov(inwardside['z'], window=window, center=True, min_periods=1)
    inward_crossing = inwardside.loc[inward_cov > 0.02]
    inward_crossing = inward_crossing.loc[inward_crossing['z'] < np.max([mean_in + 0.3, np.min(inward_crossing['z'])+0.01])]

    x_inner = inward_crossing['x'].iloc[-1]
    z_inner = inward_crossing['z'].iloc[-1]
    ind_inner = inward_crossing.loc[inward_crossing['z'] == z_inner].index.values[0]

    outwardside = profile_averaged.iloc[np.max(ind_crest):]
    mean_out = np.median(outwardside['z'].iloc[0:75])
    outward_crossing = outwardside.loc[outwardside['z'] < mean_out+0.1]
    x_outer = outward_crossing['x'].values[0]
    z_outer = outward_crossing['z'].values[0]

    #berm
    #2 points between crest and toe
    berm = profile_averaged.iloc[ind_inner:np.min(ind_crest)]
    berm_cov = pd.rolling_cov(berm['z'], window=window, center=True, min_periods=1)
    berm_points = berm.loc[berm_cov < 0.05]
    berm_points = berm_points.loc[berm_points['z'] > mean_in +1.5]


    if len(berm_points) > 1:
        berm_points = berm_points.iloc[[0,-1]]
        x_berm = berm_points['x'].values
        z_berm = np.average(berm_points['z'].values)
        #check if berm makes sense by verifying if slope is about ok:
        # slope lower part  and upper part should both be > 1.5
        if (np.min(x_berm)-x_inner)/(z_berm-z_inner) > 1.5 and (np.min(x_crest)-np.max(x_berm))/(z_crest-z_berm) > 1.5 and np.diff(x_berm) > 2:
            x_values = np.array([x_inner, x_berm[0], x_berm[1], np.min(x_crest), np.max(x_crest), x_outer])
            z_values = np.array([z_inner, z_berm, z_berm, z_crest, z_crest, z_outer])
        else:
            x_values = np.array([x_inner, np.min(x_crest), np.max(x_crest), x_outer])
            z_values = np.array([z_inner, z_crest, z_crest, z_outer])
            print('For ' + titel + ' estimated berm was deleted')
            print('lower slope: ' + str((np.min(x_berm)-x_inner)/(z_berm-z_inner)))
            print('upper slope: ' + str((np.min(x_crest)-np.max(x_berm))/(z_crest-z_berm)))
            print('berm length: ' + str(np.diff(x_berm)))
        #add berm
        # last step
        # filter out bogus berms where slopes are way too steep


    else:
        x_values = np.array([x_inner, np.min(x_crest), np.max(x_crest), x_outer])
        z_values = np.array([z_inner, z_crest, z_crest, z_outer])
        #not
    x_values.flatten()
    z_values.flatten()
    if titel =='VY094': x_values[0] = 122; z_values[0] = 2.5; print('Adapted inner toe for VY094')
    if titel =='VY058': x_values[0] = 131.5; z_values[0]  = 3.; x_values[-1] = 158;  z_values[-1] = 4.8; print('Adapted inner and outer toe for VY058')
    if titel =='AW216': x_values[1] = 139; z_values[1] = 5.5; x_values[2] = 149.6; z_values[2] = 5.5; print('Adapted crest for AW216')
    if titel == 'AW219':
        x_values[1] = 137.6; z_values[1] = 5.9; x_values[2] = 149; z_values[2] = 5.9; print('Corrected crest of AW219')
    if titel == 'AW240': x_values[0] = 124; z_values[0] = 1.07; print('Corrected inner toe of AW240')
    if titel == 'AW248': x_values[-1] = 152; z_values[-1] = 2.12; print('Corrected outer toe of AW248')
    if titel == 'AW276':
        x_values = np.insert(x_values,[0, 1], [100, 135])
        z_values = np.insert(z_values,[0, 1], [-0.7, z_values[0]])
        print('Corrected AW276')

    # Plot the  points with a line
    ax1.plot(x_values,z_values, color = 'k', marker = 'o', linestyle = '--')
    ax1.set_ylabel('m NAP')
    ax2.set_ylabel('CoV profiel')
    ax1.set_xlim(np.min(x_values)-30, np.max(x_values)+30)
    ax2.set_xlim(np.min(x_values)-30, np.max(x_values)+30)
    fig.suptitle(titel)
    if path != None:
        plt.savefig(path + '\\' + titel + '.png')
        plt.close()


    return pd.DataFrame(np.hstack((x_values[:,None], z_values[:,None])),columns=['x','z'])
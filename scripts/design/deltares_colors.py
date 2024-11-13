import seaborn as sns

class Deltares_colors:
    '''
    See bottom for implementation example
    '''

    def __init__(self):

         self.colors = {'zwart' : '#000000',
                        'blauw1' : '#080c80',
                        'blauw2' : '#0d38e0',
                        'blauw3' : '#0ebbf0',
                        'groen1' : '#00b389',
                        'groen2' : '#00cc96',
                        'groen3' : '#00e6a1',
                        'grijs1' : '#f2f2f2',
                        'grijs2' : '#e6e6e6',
                        'geel' : '#ffd814',
                        'academy' : '#ff960d'
                       }

         self.colorlist = {'DeltaresDefault'    : [self.colors[c] for c in ['blauw1', 'blauw2', 'blauw3', 'groen1', 'groen2', 'groen3' ]],
                          'DeltaresFull'       : [self.colors[c] for c in ['blauw1', 'blauw2', 'blauw3', 'groen1', 'groen2', 'groen3', 'academy', 'geel', 'grijs2', 'zwart' ]],
                          'DeltaresBlues'      : [self.colors[c] for c in ['blauw1', 'blauw2', 'blauw3' ]],
                          'DeltaresBlues_r'    : [self.colors[c] for c in ['blauw3', 'blauw2', 'blauw1' ]],
                          'DeltaresGreens'     : [self.colors[c] for c in ['groen1', 'groen2', 'groen3' ]],
                          'DeltaresGreens_r'   : [self.colors[c] for c in ['groen3', 'groen2', 'groen1' ]],
                          'DeltaresOranges'    : [self.colors[c] for c in ['academy', 'geel', 'grijs2' ]],
                          'DeltaresOranges_r'  : [self.colors[c] for c in ['grijs2', 'geel', 'academy' ]]
                          }

    def sns_palette(self, colorlist_name):
        '''
        :param colorlist_name: str, colorlist name in Deltares_colors.colorlist set
        '''

        if colorlist_name not in self.colorlist.keys():
            raise Exception('Colorlist not defined in Deltares color palette')

        cp = sns.color_palette(self.colorlist[colorlist_name])

        return cp


    def sns_sequential(self, color_name, shade='light', reverse=False, ncolors=None ):
        '''
        :param color_name: str, color name in Deltares_colors.colors set
        :param shade: str, options: 'light' or 'dark' palette
        :param reverse: boolean, to reverse the palette
        :param ncolors: int, number of colors in colormap, if None then a colormap is returned
        '''

        if color_name not in self.colors.keys():
            raise Exception('Color name not defined in Deltares color palette')

        if ncolors == None:
            cmap_bool=True
        else:
            cmap_bool = False

        if shade == 'dark':
            cp = sns.dark_palette(self.colors[color_name], as_cmap=cmap_bool, reverse=reverse, n_colors=ncolors)
        elif shade == 'light':
            cp = sns.light_palette(self.colors[color_name], as_cmap=cmap_bool, reverse=reverse, n_colors=ncolors)

        return cp

sns.set(style="whitegrid")
colors =  Deltares_colors().sns_palette("DeltaresFull")
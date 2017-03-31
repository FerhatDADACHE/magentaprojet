
# Ajouter le sample code dans le chemin
mypath = "../sample_code"
from sys import argv, path
from os.path import abspath
path.append(abspath(mypath))

# Routine graphique
import seaborn as sns; sns.set()

# Types de données
import pandas as pd


# Importer la classe mere
import data_manager

class DataManager(data_manager.DataManager):
    '''Cette classe lit et affiche des données.
       Avec l'héritage de classe
       '''
       
#    def __init__(self, basename="", input_dir=""):
#        ''' New contructor.'''
#        DataManager.__init__(self, basename, input_dir)
        # So something here
    
    def toDF(self, set_name):
        '''Changer un sous-ensemble de données donné à un cadre de données Panda.
            Set_name peut etre 'train', 'valid' ou 'test'.'''
        DF = pd.DataFrame(self.data['X_'+set_name])
        # For training examples, we can add the target values as
        # a last column: this is convenient to use seaborn
        # Look at http://seaborn.pydata.org/tutorial/axis_grids.html for other ideas
        if set_name == 'train':
            Y = self.data['Y_train']
            DF = DF.assign(target=Y)          
        return DF

    def DataStats(self, set_name):
        ''' Afficher des statistiques de données simples'''
        DF = self.toDF(set_name)
        print DF.describe()
    
    def ShowScatter(self, var1, var2, set_name):
        ''' Afficher les diagrammes de dispersion.(scatter plots).'''
        DF = self.toDF(set_name)
        if set_name == 'train':
            sns.pairplot(DF.ix[:, [var1, var2, "target"]], hue="target")
        else:
            sns.pairplot(DF.ix[:, [var1, var2]])
            

if __name__=="__main__":
    if len(argv)==1: 
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
        
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    
    basename = 'crime'
    D = DataManager(basename, input_dir)
    print D

    #pour les données d'entrainements
    print("pour les données d'entrainements")
    D.DataStats('train')
    D.ShowScatter(1, 2, 'train')
    
    #pour les données de validation
    print("pour les données de validation")
    D.DataStats('valid')
    D.ShowScatter(1, 2, 'valid')
    
    print("pour les données de tests")
    #pour les données de tests
    D.DataStats('test')
    D.ShowScatter(1, 2, 'test')
   
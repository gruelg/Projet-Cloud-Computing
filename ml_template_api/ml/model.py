from utils.utils import *

class model:

    def __inti__(self):
        pass

    def creatModel(self):
        data = DataHandler()
        data.get_data()
        featuresRecipe = FeatureRecipe(data.df_vgsales)
        featuresRecipe.prepareData(3)
        extractor = FeatureExtractor(featuresRecipe.df)
        extractor.train()
        buildModel = ModelBuilder("", True)
        buildModel.calculData(extractor.X, extractor.y)

    def loadModel(self):
        pass


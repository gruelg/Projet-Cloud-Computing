from utils.utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--split", type=float)
args = parser.parse_args()

data = DataHandler()
data.get_data()
featuresRecipe = FeatureRecipe(data.df_vgsales)
featuresRecipe.prepareData(3)
extractor = FeatureExtractor(featuresRecipe.df)
extractor.train()
buildModel = ModelBuilder("", True)
if(args.split):
    buildModel.calculData(extractor.X, extractor.y, args.split)
else:
    buildModel.calculData(extractor.X, extractor.y)


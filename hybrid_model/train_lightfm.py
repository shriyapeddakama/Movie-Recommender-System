from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
from utils import process_data, train_model
from lightfm.cross_validation import random_train_test_split
import joblib
import argparse
import time
parser = argparse.ArgumentParser(description ='Process some integers.')
parser.add_argument('--users',
                    type = str, default='/home/team11/project_data/data/processed/user_data.csv',
                    help ='Path to csv containing users data.')
parser.add_argument('--movies',
                    type = str, default='/home/team11/project_data/data/processed/movies_metadata.csv',
                    help ='Path to csv containing movies data.')
parser.add_argument('--ratings',
                    type = str, default='/home/team11/project_data/data/processed/ratings_snap.csv',
                    help ='Path to csv containing ratings data.')
parser.add_argument('--epochs',
                    type = int, default=1,
                    help ='number of training epochs.')
parser.add_argument('--no_components',
                    type = int, default=15,
                    help ='number of components per feature.')
parser.add_argument('--learning_schedule',
                    type = str, default='adagrad',
                    help ='learning_schedule type to train the model [adagrad, adadelta].')
parser.add_argument('--loss',
                    type = str, default='warp',
                    help ='loss type to train the model [warp, logistic, bpr, warp-kos].')
parser.add_argument('--random_state',
                    type = int, default=42,
                    help ='seed for reproducing exp.')
parser.add_argument('--frac',
                    type = float, default=0.1,
                    help ='Percentage of training data.')

args = parser.parse_args()

def main(args):
    start_all= time.time()
    df_user= pd.read_csv(args.users)
    df_movie= pd.read_csv(args.movies)
    df_rating= pd.read_csv(args.ratings)

    #split train/test based on user
    
    print(f" Users: {len(df_user)}, Ratings: {len(df_rating)}, Movies: {len(df_movie)}")
    start_process= time.time()
    #process
    all_user_ids, all_item_ids, all_user_feats, all_item_feats, user_feats_iter, item_feats_iter, triples = process_data(df_user, df_movie, df_rating)
    end_process= time.time()
    #create datasets objects
    start_data= time.time()
    dataset= Dataset()
    dataset.fit(users=all_user_ids,
                items=all_item_ids,
                user_features= all_user_feats,
                item_features= all_item_feats)
    user_features = dataset.build_user_features(user_feats_iter, normalize=False)
    item_features = dataset.build_item_features(item_feats_iter, normalize=False)
    interactions, weights = dataset.build_interactions(triples)

    interactions_train, interactions_test = random_train_test_split(
    interactions, test_percentage=args.frac, random_state=args.random_state
    )
    end_data= time.time()


    #model
    model = LightFM(no_components=args.no_components, loss=args.loss, random_state=args.random_state, learning_schedule=args.learning_schedule, item_alpha=1e-6)
    
    train_model(model, interactions_train, interactions_test, [weights, user_features, item_features], args.epochs)
    end_all= time.time()
    print(f"full pipline time: {end_all-start_all}s\tdataset pipline time: {end_data-start_data}s\tdataset processing: {end_process-start_process}s")
    #save model
    joblib.dump(model, "/home/hbukhari/Project/lightfm_model.joblib")
    joblib.dump(dataset, "/home/hbukhari/Project/lightfm_dataset.joblib")
    joblib.dump(user_features, "/home/hbukhari/Project/user_features.joblib")
    joblib.dump(item_features, "/home/hbukhari/Project/item_features.joblib")

if __name__ == "__main__":
    main(args)

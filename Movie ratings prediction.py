import pandas as pd
import scipy.spatial.distance
import numpy as np
import time


#Function bloc to read data file into pandas and include column names
def reading_csv(train_file,test_file):
    colnames = ['user', 'movie', 'rating', 'time']
    train = pd.read_csv('train1.csv', names=colnames, header=None)
    test = pd.read_csv('test1.csv', names=colnames, header=None)
    return train,test

#Function block to calculate Euclidean, cityblock and cosine distances
def distance(train_pivot,distancefunc):
    distance = scipy.spatial.distance.cdist(train_pivot.iloc[:, 1:], train_pivot.iloc[:, 1:], metric = distancefunc)
    distance_df = pd.DataFrame(distance)
    #print("euclis :", euclidean_df)
    return distance_df

#Function block to calculate users_similar is a dictionary of each movie id and all users who have watched that movie
def similar_user(train):
    users_similar= {}
    for i in train.movie:
        if i not in users_similar:
        #list2 = []
            d = train.loc[train.movie == i]
            l2 = d['user']
            l2 = l2.tolist()
            users_similar[i] = l2
    return users_similar

#Function block to find the dictionary of each users with all of their neighbours
def neighbors(distance_df):
    users_neighbor = {}
    for index, row in distance_df.iterrows():
        list3 = np.asarray(row).argsort().tolist()
        list3.remove(index)
        list3 = map(lambda x: x+1, list3)
        users_neighbor[index + 1] = list3
    return users_neighbor


#for test users, we obtain movie ratings by finding intersection among users_neighbor and users_similar

def madlist1(test,train_pivot,k):
    mean_list=[]
    MAD_list1 = []
    for index, row in test.iterrows():
        user = row['user']
        movie = row['movie']
        #neighborList = movieDict[usr]
        list_neighbor = users_neighbor[user]
        if movie not in users_similar:
            mean = 0
            mean_list.append(mean)
            continue
        list_movie = users_similar[movie]
        #print("movieList :: ", list_movie )
        s =set(list_movie)
        #list_common = set(list_neighbor).intersection(set(list_movie))
        list_common = [x for x in list_neighbor if x in s]
        #print("commonusers:: ", list_common)
        list_rating = [train_pivot.ix[i, movie] for i in list_common]
        list_rating = np.array(list_rating)
        #print("ratingList :: ", list_rating)
        mean = 0
        if len(list_rating) >= k:
            mean = np.mean(list_rating[:k])
        elif len(list_rating) > 0:
            mean = np.mean(list_rating[:len(list_rating)])
        else:
            list_rating = [train_pivot.ix[i, movie] for i in list_movie]
            if len(list_rating) > 0:
                mean = np.mean(list_rating[:len(list_rating)])
            else:
                mean = 0
        mean_list.append(mean)
        #test['predicted rating'] = mean_list
        rating_list = np.array(test.rating.tolist())
    mean_list = np.array(mean_list)
    mad_list = np.subtract(mean_list,rating_list)
    test['MAD'] = mad_list
    test['MAD'] = test['MAD'].abs()
    MAD_list1 = test.MAD.tolist()
    return MAD_list1

#Function block for computing ratings based on average over all users who rated the movie
def madlist2(test,train_pivot,k):
    mean_list=[]
    MAD_list2 = []
    for index, row in test.iterrows():
        user = row['user']
        movie = row['movie']
        #neighborList = movieDict[usr]
        #list_neighbor = users_neighbor[user]
        #print("neighborlist :: ", list_neighbor)
        if movie not in users_similar:
            mean = 0
            mean_list.append(mean)
            continue

        list_movie = users_similar[movie]

        #print("movieList :: ", list_movie )
        #list_common = set(list_neighbor).intersection(set(list_movie))
        #print("commonusers:: ", list_common)
        list_rating = [train_pivot.ix[i, movie] for i in list_movie]
        list_rating = np.array(list_rating)
        #print("ratingList :: ", list_rating)
        if len(list_rating)==0:
            mean = 0
        else:
            mean = np.mean(list_rating)
        #print("mean is:", mean)
        mean_list.append(mean)
        #test['predicted rating'] = mean_list
        rating_list = np.array(test.rating.tolist())
    mean_list = np.array(mean_list)
    mad_list = np.subtract(mean_list,rating_list)
    test['MAD'] = mad_list
    test['MAD'] = test['MAD'].abs()
    MAD_list2 = test.MAD.tolist()
    return MAD_list2

#This function calculates MAD for proper algorithm

def find_MAD1(MAD_list1,test,length):
    MAD_list1 = np.array(MAD_list1)
    total = np.sum(MAD_list1)
    MAD1 = total/length
    return MAD1

#This function calculates MAD for basic algorithm

def find_MAD2(MAD_list2,test,length):
    MAD_list2 = np.array(MAD_list2)
    total = np.sum(MAD_list2)
    MAD2 = total / length
    return MAD2


# Beginning of my program
start = time.clock()
train_file =['train1.csv','train2.csv','train3.csv','train4.csv','train5.csv']
test_file =['test1.csv','test2.csv','test3.csv','test4.csv','test5.csv']
#train_file = ['train1.csv']
#test_file = ['test1.csv']
distances = ['euclidean', 'cityblock', 'cosine']
k =50
length = 0
MAD_list1 = []
MAD_list2 = []
for i in range(len(distances)):
    for j in range(len(train_file)):
        train,test = reading_csv(train_file[j],test_file[j])
        train_pivot = train.pivot_table(index = 'user', columns='movie', values='rating',aggfunc='first',fill_value=0)
        distance_df = distance(train_pivot,distances[i])
        users_similar = similar_user(train)
        users_neighbor = neighbors(distance_df)
        MAD_list1 += madlist1(test,train_pivot,k)
        MAD_list2 += madlist2(test,train_pivot,k)
        length += len(test.index)
    MAD1 = find_MAD1(MAD_list1, test, length)
    MAD2 = find_MAD2(MAD_list2, test, length)
    #proper algorithm is the one that is designed based on user i and each movie j they did not see, top k most similar users to i who have seen j, used this to infer i's rating on j
    print("MAD of proper algorithm is",distances[i]," is::",MAD2)
    #Basic algorithm is the simple algorithm which gives each user movie pair a rating that is equal to average score over all users who rated that movie
    print("MAD of basic algorithm is",distances[i]," is::",MAD1)

print time.clock() - start

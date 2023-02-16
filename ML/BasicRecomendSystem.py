import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


#read user data
u_usercolumns = ['id','age','gender','work','zipcode']
u_user = pd.read_csv('DataMovieLen/u.user', names=u_usercolumns,sep='|',encoding='latin-1')
number_user = u_user.shape[0]
#read rating file which is our training set and test set
u_ratecolumns = ['id','movie_id','rate_point','timestamp']
u_train = pd.read_csv('DataMovieLen/ua.base', names=u_ratecolumns,sep='\t',encoding='latin-1')
u_test = pd.read_csv('DataMovieLen/ua.test', names=u_ratecolumns,sep='\t',encoding='latin-1')
training_ratenumber = u_train.shape[0]
test_ratenumber = u_test.shape[0]
#do cột timestamp không cần dùng đến nên chúng ta sẽ loại bỏ
u_train = u_train[['id','movie_id','rate_point']]
u_test = u_test[['id','movie_id','rate_point']]
#read feature of flim
item_column = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item = pd.read_csv('DataMovieLen/u.item',names=item_column,sep='|',encoding='latin-1')
item = item[['unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
film_featuresnumber = item.shape[1]
film_number = item.shape[0]
X0 = np.asmatrix(item)
X_train_counts = X0[:, -19:]
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()
print(tfidf[0][:])
print(tfidf[1][:])
#transformer dùng để tạo đặc trưng cho các bộ phim
def get_item_rated_by_user(rate_matrix,id):
    y = np.array(rate_matrix[:,0])
    start = 0
    end = 0
    for i in range(y.shape[0]):
        if y[i]==id:
            start = i
            break
    for i in range(y.shape[0]):
        if y[i]==id+1:
            end= i-1
            break
        elif i == y.shape[0]-1:
            end = i
            break
    id_place = np.array(np.linspace(start,end,end-start+1),dtype='int64')
    k = end-start+1
    score = []
    film_id = []
    for i in range(k):
        m = rate_matrix[id_place[i],2]
        score.append(m)
    for i in range(k):
        m = rate_matrix[id_place[i],1]
        film_id.append(m)
    score = np.asarray(score).reshape([k,1])
    film_id = np.asarray(film_id).reshape([k,1])
    return (score,film_id, k)
#print(get_item_rated_by_user(np.array(u_train),1))
w = np.zeros([film_featuresnumber,number_user])
b = np.zeros([1,number_user])
for i in range(number_user):
    score, film_id, numberrate = get_item_rated_by_user(np.array(u_train),i+1)
    clf = Ridge(alpha=0.1,fit_intercept=True)
    xhat = []
    for j in range(numberrate):
        k = tfidf[film_id[j][0]-1,:]
        xhat.append(k)
    xhat = np.asarray(xhat)
    clf.fit(xhat,score)
    w[:,i]=clf.coef_
    b[0,i] = clf.intercept_
Yhat = tfidf@w+b
n = 30
score, film_id, numberrate = get_item_rated_by_user(np.array(u_train),30)
print('Rated movies ids :', film_id)
print('True ratings     :', score)
print('Predicted ratings:', Yhat[film_id, n])
#kết quả nhận được chưa được tốt lắm nguyên do là đã đơn giản hóa chương trình đi khá nhiều
#chúng ta sẽ thử lại với Neighborhood-Based Collaborative Filtering

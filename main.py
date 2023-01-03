def read_file(path_data,path_json):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #Reading the data from csv file

    data=pd.read_csv(path_data)
    print(data.head(5))
    print(data.columns)

    #Reading the string from the rtf file
    from striprtf.striprtf import rtf_to_text
    file_cont=open(path_json,'r')
    rtf=file_cont.read()
    file_cont.close()
    text = rtf_to_text(rtf)
    #print(text)
    dict={text}

    #Parsing the string using json format to convert to a python object
    import json
    p_ob=json.loads(text)
    target=p_ob['design_state_data']['target']
    feature_handling=p_ob['design_state_data']['feature_handling']
    print(feature_handling.keys())

    #imputing missing values
    return(data,p_ob,target)

#to impute numerical values
def impute_num(val,data,col):
    import numpy as np

    #print("here")
    if 'average' in val.lower():
        #print("there")
        data.loc[data.loc[:,col].isna(),col]=np.mean(data.loc[:,col])

    if 'custom' in val.lower():
            #print("where")
            data.loc[data.loc[:, col].isna(), col] = np.median(data.loc[:, col])

    return (data)

#to impute categorical values
def impute_cat(val,data,col):

    new_df=pd.DataFrame()
    if "hash" in method.lower():
        obj=HashingVectorizer(n_features=5)
        new_df=obj.fit_transform(data.loc[:,col])
    return(new_df)


# Here we split the feature and target columns for distinction between the two and to conveniently proceed to model training
def split_data(data,target):

    cols=data.columns

    target=target['target']
    y=(data.loc[:,target])
    X=(data.loc[:,cols!=target])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)
    return(X_train,X_test,y_train,y_test)



def feature_reduction(method,data_X,data_y,details):



    if 'tree' in method.lower():
        n_trees=int(details['num_of_trees'])
        d_trees=int(details['depth_of_trees'])
        f_to_keep=int(details['num_of_features_to_keep'])
        print(f_to_keep)
        features=data_X.columns
        #Random forest or ensemble of decision trees for feature_reduction
        estimator=RandomForestRegressor(n_estimators=n_trees,random_state=9)
        estimator.fit(data_X,data_y)
        f_i = list(zip(features,estimator.feature_importances_))
        sel_f=RFE(estimator,n_features_to_select=f_to_keep)
        sel_f.fit(data_X,data_y)
        selected_features=np.array(features)[sel_f.get_support()]
        return(data_X.loc[:,selected_features])

    if "no reduction" in method.lower():
        return (data_X)
    if "pca" in method.lower():

        scaler = StandardScaler()
        scaler.fit(data_X)
        data_X=scaler.transform(data_X)
        pca=PCA(.95)
        pca.fit(data_X)
        data_X=pca.transform(data_X)
        return (data_X)
    if 'corr' in method.lower():
        cor_mat=data_X.corr().abs()
        print(cor_mat)
        upper_tri=cor_mat.where(np.triu(np.ones(cor_mat.shape),k=1).astype(np.bool))
        print(upper_tri)
        to_drop=[col for col in upper_tri.columns if any(upper_tri[col]>0.95)]
        print(to_drop)
        data_X=data_X.drop(data_X.columns[to_drop],axis=1)
        return (data_X)

def model_build(algo,X_train,y_train,p_ob):
    if 'randomforestregressor' in algo.lower():
        print(p_ob[algo].keys())
        strat=p_ob[algo]["feature_sampling_statergy"]
        #print(strat)
        min_trees=int(p_ob[algo]['min_trees'])
        max_trees=int(p_ob[algo]['max_trees'])
        mini_depth=int(p_ob[algo]['min_depth'])
        max_depth=int(p_ob[algo]['max_depth'])
        min_sa_per_leaf=int(p_ob[algo]['min_samples_per_leaf_min_value'])
        max_sa_per_leaf=int(p_ob[algo]['min_samples_per_leaf_max_value'])
        if p_ob[algo]['parallelism']==0:
            para=[False]
        else:
            para=[True]
        param_grid = {'n_estimators': range(min_trees,max_trees+1),
                       'max_depth': range(mini_depth, max_depth + 1),
                       'min_samples_leaf': range(min_sa_per_leaf, max_sa_per_leaf + 1),
                       'bootstrap': para}
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predi=best_grid.predict(X_test)
        mse = mean_squared_error(y_test, predi)
        rmse = np.sqrt(mse)
        return (rmse)
    if 'decisiontreeregressor' in algo.lower():
        print(p_ob[algo].keys())
        #strat=p_ob[algo]["feature_sampling_statergy"]
        #print(strat)
        splitt=[]
        if p_ob[algo]['use_best']==True:
            splitt=['best']
        if p_ob[algo]['use_random']:
            splitt=['random']


        mini_depth=int(p_ob[algo]['min_depth'])
        max_depth=int(p_ob[algo]['max_depth'])
        min_sa_per_leaf=(p_ob[algo]['min_samples_per_leaf'])
        #max_sa_per_leaf=int(p_ob[algo]['min_samples_per_leaf_max_value'])

        param_grid = {'splitter':splitt,
                       'max_depth': range(mini_depth, max_depth + 1),
                       'min_samples_leaf': min_sa_per_leaf,
                       }
        dt = DecisionTreeRegressor()
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predi=best_grid.predict(X_test)
        mse = mean_squared_error(y_test, predi)
        rmse = np.sqrt(mse)
        return (rmse)



    if 'gbtregressor' in algo.lower():#sub_Sample values should be in the range (0.0,1.0]
        print(p_ob[algo].keys())
        strat = p_ob[algo]["feature_sampling_statergy"]
        #print(strat)
        max_fea=[]
        min_sub = (p_ob[algo]['min_subsample'])
        max_sub = int(p_ob[algo]['max_subsample'])
        mini_step = int(p_ob[algo]['min_stepsize'])
        max_step = int(p_ob[algo]['max_stepsize'])
        mini_depth = int(p_ob[algo]['min_depth'])
        max_depth = int(p_ob[algo]['max_depth'])
        min_iter = int(p_ob[algo]['min_iter'])
        max_iter = int(p_ob[algo]['max_iter'])
        n_est=(p_ob[algo]['num_of_BoostingStages'])
        #cv=int(p_ob[algo]['num_of_BoostingStages'])
        if 'fixed' in p_ob[algo]['feature_sampling_statergy'].lower():
            max_fea=[int(p_ob[algo]['fixed_number'])]
        param_grid={'n_estimators': n_est,
                           'max_depth': range(mini_depth, max_depth + 1),

                            #'subsample': range(min_sub,max_sub+1),
                            'max_features':max_fea}
        tree_regressor=GradientBoostingRegressor()
        grid_search=GridSearchCV(estimator=tree_regressor, param_grid=param_grid,
                                       cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_grid = grid_search.best_estimator_
        predi = best_grid.predict(X_test)
        mse = mean_squared_error(y_test, predi)
        rmse = np.sqrt(mse)
        return (rmse)
    if 'elasticnetregression' in algo.lower():
        print(p_ob[algo].keys())
        maxi_iter=int(p_ob[algo]['max_iter'])
        min_iter=int(p_ob[algo]['min_iter'])

        min_alpha= float(p_ob[algo]['min_regparam'])
        max_alpha=float(p_ob[algo]['max_regparam'])
        min_en=float(p_ob[algo]['min_elasticnet'])
        max_en=float(p_ob[algo]['max_elasticnet'])
        param_grid = {'max_iter': range(min_iter,maxi_iter),
                      'l1_ratio': [min_en, max_en],

                      'alpha': [min_alpha, max_alpha]

                      }
        e_n = ElasticNet()
        grid_search = GridSearchCV(estimator=e_n, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predi = best_grid.predict(X_test)
        mse = mean_squared_error(y_test, predi)
        rmse = np.sqrt(mse)
        return (rmse)
    if 'lasso' in algo.lower():
        print(p_ob[algo].keys()) #similar to other model building and evaluating afterwards.
        lr=Lasso()
        maxi_iter = int(p_ob[algo]['max_iter'])
        min_iter = int(p_ob[algo]['min_iter'])

        min_alpha = float(p_ob[algo]['min_regparam'])
        max_alpha = float(p_ob[algo]['max_regparam'])

        param_grid = param_grid = {'max_iter': range(min_iter,maxi_iter),


                      'alpha': [min_alpha, max_alpha]

                      }
        grid_search=GridSearchCV(estimator=lr, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predi = best_grid.predict(X_test)
        mse = mean_squared_error(y_test, predi)
        rmse = np.sqrt(mse)
        return (rmse)
    if 'ridge' in algo.lower():
        print(p_ob[algo].keys())  # similar to other model building and evaluating afterwards.
        rr = Ridge()
        maxi_iter = int(p_ob[algo]['max_iter'])
        min_iter = int(p_ob[algo]['min_iter'])

        min_alpha = float(p_ob[algo]['min_regparam'])
        max_alpha = float(p_ob[algo]['max_regparam'])

        param_grid = param_grid = {'max_iter': range(min_iter, maxi_iter),

                                   'alpha': [min_alpha, max_alpha]

                                   }
        grid_search = GridSearchCV(estimator=rr, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predi = best_grid.predict(X_test)
        mse = mean_squared_error(y_test, predi)
        rmse = np.sqrt(mse)
        return (rmse)




if __name__=='__main__':
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFE
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor

    path_data="/Users/shivangidubey/Downloads/Screening Test/iris.csv"

    path_json="/Users/shivangidubey/Downloads/Screening Test/algoparams_from_ui.json.rtf"
    data,p_ob,target=read_file(path_data,path_json)
    feature_handling = p_ob['design_state_data']['feature_handling']
    for i in data.columns:

        tot_miss = data.loc[:, i].isna().sum() + data.loc[:, i].isnull().sum()
        print("column_name:{}".format(i))
        print("the column {} has {} missing values".format(i, tot_miss))

        type = feature_handling[i]['feature_variable_type']
        if type == 'numerical':
            print("In case of this variable we impute with," + feature_handling[i]['feature_details']['impute_with'])
            imp_with_val = feature_handling[i]['feature_details']['impute_with']
            data = impute_num(imp_with_val, data, i)
        else:
            categories = data.loc[:, i].unique()
            print(feature_handling[i]['feature_details']['text_handling'])
            method = feature_handling[i]['feature_details']['text_handling']
            n_data = impute_cat(method, data, i)
            n_data = pd.DataFrame(n_data.toarray())
            data = data.drop(columns=i)
            data = pd.concat([data, n_data], axis=1)
            print(data.columns)
    # Replacing integer column names with string names
    j = 1
    for i in data.columns:
        if i not in p_ob['design_state_data']['feature_handling'].keys():
            print("here")
            data.rename({i: "feature_" + str(j)}, inplace=True, axis='columns')
            j = j + 1
    print(data.columns)
    X_train,X_test,y_train,y_test=split_data(data,target)

    # Extracting the parameters mentioned in the json file for feature reduction: tree based
    f_way = p_ob['design_state_data']['feature_reduction']['feature_reduction_method']
    features_selected=feature_reduction(f_way, X_train, y_train, p_ob['design_state_data']['feature_reduction'])
    print(features_selected)
    algo=p_ob['design_state_data']['algorithms'].keys()
    selected_algo=[x for x in algo if 'regression' in x.lower() or 'regressor' in x.lower() ]
    print(selected_algo)
    for i in selected_algo:
        #if p_ob['design_state_data']['algorithms'][i]['is_selected'] == True:
            print(i)
            rmse_m=model_build(i,X_train,y_train,p_ob['design_state_data']['algorithms'])
            print(rmse_m)




























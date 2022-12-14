# Regression-problems-using-Neural-Network-with-Custom-Loss-Function

This is an attempt to solve a regression problem using Neural Networks. Vanilla loss function penalizes errors on either side of the actuals equally but in certain case studies it might be required to penalize losses differently. For example, early delivery and breach in delivery in case of e-commerce  certainly have different impact on customer experience.

Libraries used : Keras

PROCESS:

1.Read your dataframe 

2.Define variable like list of numerical col, target and number of principle components

3.Pass the dataframe through :

    df=data_preprocessing(df,numeric_col_list,n_components,target)
    
    main_X, main_y,test_main_X,test_main_y=(df,time_interval,target)

4.Pass the main_X, main_y, test_main_X, test_main_y found in above through createAndSaveModel function

5.Run the below line of codes:
    
    result_set=[createAndSaveModel(main_X, main_y,test_main_X,test_main_y, 'Model-v0.1'+str(i)+str(j), i,j) for i,j in choice_set]
    
    res = result_set[0]
    
    res['predicted'] = res['predicted'].map(lambda x: x[0])
    
    df.join(res[['actual','predicted']]).to_csv('Results_'+ str(dt.today().date()) + '.csv.gz', compression = 'gzip')


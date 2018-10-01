import numpy as np


def normalize_data(raw_data):
    """
        Function which normalize the data
        parameters:
        - raw_data : matrix of data (N,M), the first two columns correspond to ID and classes, the next columns correspond 
        to the features (of course, we only normalize the features). N rows correspond to N examples
        
        returns:
        - normalize_data : raw_data normalized
        - mean_vector : vector of M-2 elements, mean of the features
        - std_vector : vector of M-2 elements, standard deviation of the features
        
    """
    data_shape = raw_data.shape
    mean_vector  = np.mean (raw_data[:,2:],axis=0) # mean_vector[i] : moyenne de la colonne du feature i
    std_vector  = np.std (raw_data [:,2:],axis=0)  # std_vector[i] : écart-type de la colonne du feature i
    
    normalized_data = raw_data
    normalized_data[:,2:] = (raw_data[:,2:]-mean_vector)/std_vector
    return(normalized_data,mean_vector,std_vector)


def split_data (data,nb_train):
    """
        Function which splits the data into a test_set and a training set
        parameters :
        - data : matrix of data to split
        - nb_train : number of training examples we want to keep
        
        returns :
        - train_set : matrix of training data
        - test_set : matrix of test data
        - N_test : number of elements of the test set
    """
    np.random.shuffle(data) # mélange des exemples 
    train_set = data[:nb_train,:]
    test_set = data[nb_train:,:]
    N_test = test_set.shape[0]
    return(train_set,test_set,N_test)


def knn_algo_deux (K, train_set, test_set):
    """
        Function which operates knn algorithm.
        parameters:
        - K : number of k nearest neighbours wanted
        - train_set : (N_train,M+2) training data, N_train examples, M features, 1st column is ID, 2nd column is class
        - test_set : (N_test,M+2) test data
        
        returns : 
        - error_rate : Percentage of error
        - predic_class : vector of N_test elements, which give the predicted class of the test points.
    """
    
    
    ## Matrices, valeurs utiles
    N_train = train_set.shape[0]
    N_test = test_set.shape[0]
    train_features = train_set [:,2:] # matrice de features du training set (N_train x nb_features)
    test_features = test_set [:,2:] # matrice de features du test set (N_test x nb_features)
    class_train = train_set[:,1] # vecteur qui donne les classes(1 ou 2) du training set
    class_test = test_set[:,1] # vecteur qui donne les classes (1 ou 2) du test set
    
    
    ## Calcul matrice distances_square :
    train_features_transpose = np.transpose(train_features)
    test_features_transpose = np.transpose(test_features)
    tile_train_features  = np.tile(train_features_transpose[:,:,np.newaxis],(1,1,N_test))
    tile_test_features = np.tile(test_features_transpose[:,np.newaxis,:],(1,N_train,1))

    
    matrice_square = np.square(tile_test_features-tile_train_features) 
    # matrice_square[i,j,k] donne (x_ij-y_ik)², où x_ij est le feature i du jiè exemple du training set, y_ik du test set

    distances_square = np.sum(matrice_square,axis=0)
    # distances_square[i,j] donne la distance au carrée entre le iè exemple du training set et le jè point du test set
    

    
    ## Calcul matrice knearest : donne les indices des k plus proches voisins 
    ind_sort = np.argsort(distances_square,axis=0) 
    # ind_sort : tableau d'indices du training set ordonnés (ordre croissant des distances)
    knearest = ind_sort[:K,:]    
    
    
    ## Compute predic_class : vecteur qui contient la classe prédite pour chaque point i du test set
    knearest_class = np.zeros(knearest.shape) 
    # init knearest_class : array avec les classes correspondant aux indices de knearest (ses éléments sont 1 ou 2)
    
    count_class_1 = np.zeros(N_test)
    # init count_class_1 : vecteur qui compte le nombre de classes 1 parmi les knearest de i

    predic_class = np.zeros(N_test)
    
    
    for i in range(N_test):
        knearest_class[:,i] = class_train[knearest[:,i]]
        count_class_1 [i] = np.sum (knearest_class[:,i]==1)
        if (count_class_1[i] >= np.ceil(K/2) ):
            predic_class[i] = 1
        else:
            predic_class[i] = 2 
        # Si K est pair, et qu'il y a autant de classe 1 que de classe 2, l'algo prédit 1 (malade) : il vaut mieux avoir 
        # des faux positifs que des faux négatifs !

    accuracy = np.sum(class_test == predic_class)/N_test 
    error_rate = (1 - accuracy)*100  # en pourcentage
    return error_rate


def with_cython (K,nb, train_set,test_set):
    for i in range (nb):
        error_rate = knn_algo_deux(K,train_set,test_set)
    return (error_rate)
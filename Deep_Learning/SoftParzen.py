import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, banknote):
        return np.mean(banknote[:, 0:4], axis = 0)

    def covariance_matrix(self, banknote):
        return np.cov(np.transpose(banknote[:, 0:4])) 

    def feature_means_class_1(self, banknote):
        return np.mean(banknote[banknote[:, 4] == 1, 0:4], axis = 0)

    def covariance_matrix_class_1(self, banknote):
        return np.cov(np.transpose(banknote[banknote[:, 4] == 1, 0:4])) 


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs 
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))  

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0] 
        counts = np.ones((num_test, self.n_classes)) 
        classes_pred = np.zeros(num_test) 
        
        for(i, row) in enumerate(test_data): 
            
            distances = (np.sum(np.abs(row - self.train_inputs)**2.0, axis=1))**(1.0/2.0) 
            
            neighbour_idx = [] 
            
            neighbour_idx = np.array([j for j in range(len(distances)) if distances[j] < self.h]) 
            if len(neighbour_idx) > 0: 
                for k in neighbour_idx: 
                    counts[i, int(self.train_labels[k]) ] += 1 
                # predict 
                classes_pred[i] = np.argmax(counts[i, :]) 
            else: 
                classes_pred[i] = draw_rand_label(row, self.label_list)
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs 
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))  

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0] 
        classes_pred = np.zeros(num_test) 
        
        def gaussian(x, Y,sigma=self.sigma, d =self.train_inputs.shape[1]): 
            return (np.exp(-((x - Y))**2/(2*self.sigma**2))/(np.power(self.sigma, d) * np.power(2*np.pi, d/2)) )
        
        
        for(i, row) in enumerate(test_data): 
            kde = np.mean(gaussian(row, self.train_inputs, self.sigma), axis = 1) 
            
            # Selecting feature with the highest p(x) 
            #kde_max_ind = np.argmax(kde) 
            #classes_pred[i] = self.train_labels[kde_max_ind, ] 
            
            # sum kde for each class, and choose the class with max
            pred_result = np.zeros((len(np.unique(self.train_labels)), 2)) 
            for j in range(len(np.unique(self.train_labels))): 
                current_label = np.unique(self.train_labels)[j] 
                pred_result[j, 0] = current_label 
                pred_result[j, 1] = np.sum(np.sort(kde[self.train_labels == current_label]))
            pred_result_max_ind = np.argmax(pred_result[:, 1]) 
            classes_pred[i] = pred_result[pred_result_max_ind, 0]
            
        return classes_pred 
            


def split_dataset(banknote):
    all_inds = np.array(range(0, banknote.shape[0])) 
    valid_inds = all_inds[all_inds % 5 == 3] 
    test_inds = all_inds[all_inds % 5 == 4] 
    train_inds = np.sort(np.append(all_inds[all_inds % 5 == 0 ], 
                           np.append(all_inds[all_inds % 5 == 1 ], all_inds[all_inds % 5 == 2 ])))
    
    train_set = banknote[train_inds, :] 
    valid_set = banknote[valid_inds, :] 
    test_set = banknote[test_inds, :] 
    return (train_set, valid_set, test_set) 


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        h_parzen = HardParzen(h) 
        h_parzen.train(train_inputs= self.x_train, train_labels=self.y_train) 
        classes_pred_h_parzen = h_parzen.compute_predictions(self.x_val) 
        
        total_num = len(self.y_val) 
        num_correct = np.sum(np.array(classes_pred_h_parzen) == np.array(self.y_val)) 
        test_error = float((1.0 - (float(num_correct) / float(total_num)))) 
        return test_error 

    def soft_parzen(self, sigma):
        s_parzen = SoftRBFParzen(sigma) 
        s_parzen.train(train_inputs= self.x_train, train_labels=self.y_train) 
        classes_pred_s_parzen = s_parzen.compute_predictions(self.x_val) 
        
        total_num = len(self.y_val) 
        num_correct = np.sum(np.array(classes_pred_s_parzen) == np.array(self.y_val)) 
        test_error = float((1.0 - (float(num_correct) / float(total_num)))) 
        return test_error 


def get_test_errors(banknote):
    train_set, valid_set, test_set = split_dataset(banknote) 

    h_ast = 3
    sigma_ast = 0.4 

    q6 = ErrorRate(x_train=train_set[:, 0:4], y_train=train_set[:, 4], 
               x_val=test_set[:, 0:4], y_val=test_set[:, 4]) 
    h_parzen_errors = q6.hard_parzen(h = h_ast) 
    s_parzen_errors = q6.soft_parzen(sigma_ast) 
    
    return np.array([h_parzen_errors, s_parzen_errors])
    


def random_projections(X, A):
    return np.matmul(X, A) / np.sqrt(2) 
        
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class data:
    def __init__(self):
        self.model = None
        self.X = None
        self.Y = None
        self.group = None

    def fetch_train_data(self, train_path, train_label_path):
        data_df = pd.DataFrame()
        ret_train, train_img = cv2.imreadmulti(train_path, [])
        if ret_train is False:
            msg = 'Failed to load image from ' + train_path
            raise OSError(msg)
        ret_label, train_label = cv2.imreadmulti(train_label_path, [])
        if ret_label is False:
            msg = 'Failed to load image from ' + train_label_path
            raise OSError(msg)

        train_df = pd.DataFrame()

        for img_name, img in enumerate(train_img):
            print(f'Loading slice {img_name}')
            train_df = pd.concat([train_df, self.feature_engineering(img, str(img_name) + '.jpg')])
            
        train_label_df = pd.DataFrame()

        for img_name, mask in enumerate(train_label):
            df = pd.DataFrame()

            df['label'] = mask.reshape(-1)
            df['img_label'] = img_name

            train_label_df = pd.concat([train_label_df, df])
        data_df = pd.concat([train_df, train_label_df], axis= 1)
        self.group = self.df.groupby('img').ngroup() + 1
        self.X = data_df.drop(['img', 'img_label', 'label'], axis= 1).reset_index(drop= True)
        self.Y = data_df['label'].values

    def fetch_inference(self, inference_path):
        ret, inference_tif = cv2.imreadmulti(inference_path, [])
        if ret is False:
            msg = 'Failed to load image from ' + inference_path
            raise OSError(msg)
        for img_name, img in enumerate(inference_tif):
            print(f'Loading inference slice {img_name}')
            img_df = self.feature_engineering(img, 'infer.jpg', True)
            yield img_df

    def create_model(params):
        model_name = params['model']['name']
        threads = params['model']['threads']
        if model_name == 'random_forest':
            detail = params['model']['detail']
            n_estimators = detail.get['n_estimators', 50]
            criterion = detail.get['criterion', 'gini']
            max_depth = detail.get['max_depth', None]
            min_sample_split = detail.get['min_samples_split', 2]
            min_samples_leaf = detail.get['min_samples_leaf', 1]
            min_weight_fraction_leaf = detail.get['min_weight_fraction_leaf', 0.0]
            max_features = detail.get['max_features', 'sqrt']
            max_leaf_nodes = detail.get['max_leaf_nodes', None]


    def feature_engineering(self, img, img_name, inferrence= False):
        df = pd.DataFrame()

        df['og_pixels'] = img.reshape(-1)
        df['img'] = img_name

        num = 1
        kernels = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for lamda in np.arange(0, np.pi, np.pi / 4):
                    for gamma in (0.05, 0.5):
                        gabor_label = 'gabor_' + str(num)
                        ksize = 9
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype= cv2.CV_32F)
                        kernels.append(kernel)
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)
                        df[gabor_label] = filtered_img
                        num += 1

        edges = cv2.Canny(img, 100, 200)
        edges1 = edges.reshape(-1)
        df['canny'] = edges1

        from skimage.filters import roberts, sobel, scharr, prewitt, gaussian

        edge_roberts = roberts(img)
        edge_roberts1 = edge_roberts.reshape(-1)
        df['roberts'] = edge_roberts1

        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['sobel'] = edge_sobel1

        edge_scharr = scharr(img)
        edge_scharr1 = edge_scharr.reshape(-1)
        df['scharr'] = edge_scharr1

        edge_prewitt = prewitt(img)
        edge_prewitt1 = edge_prewitt.reshape(-1)
        df['prewitt'] = edge_prewitt1

        gaussian_img = gaussian(img, 3)
        df['gaussian_s3'] = gaussian_img.reshape(-1)

        gaussian_img2 = gaussian(img, 7)
        df['gaussian_s7'] = gaussian_img2.reshape(-1)


        median_img = cv2.medianBlur(img, 3)
        df['median_s3'] = median_img.reshape(-1)

        if inferrence:
            df = df.drop('img', axis= 1)

        return df
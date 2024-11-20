import pandas as pd
import numpy as np

def compute_intensity(image):
    return np.sum(image) / 255.0

def compute_symmetry(image):
    img = image.reshape(28, 28)
    vertical_symmetry = np.sum(np.abs(img[:, :14] - np.fliplr(img[:, 14:])))
    horizontal_symmetry = np.sum(np.abs(img[:14, :] - np.flipud(img[14:, :])))
    return (vertical_symmetry + horizontal_symmetry) / 255.0

def preprocess_data(input_csv, output_csv):
    df = pd.read_csv(input_csv, sep=';')
    new_data = {'label': [], 'intensidade': [], 'simetria': []}
    
    for _, row in df.iterrows():
        # print(df.columns)
        label = row['label']
       
        image = row.drop('label').values
        intensity = compute_intensity(image)
        symmetry = compute_symmetry(image)
        
        new_data['label'].append(label)
        new_data['intensidade'].append(intensity)
        new_data['simetria'].append(symmetry)

    reduced_df = pd.DataFrame(new_data)
    reduced_df.to_csv(output_csv, index=False)
# Executar preprocessamento
preprocess_data('train.csv', 'train_redu.csv')
preprocess_data('test.csv', 'test_redu.csv')


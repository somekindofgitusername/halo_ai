import pandas as pd
import ast

def preprocess_and_feature_engineer(file_path, output_file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Convert string representations of tuples to actual tuples
    data['MyOrient'] = data['MyOrient'].apply(ast.literal_eval)
    data['Cd'] = data['Cd'].apply(ast.literal_eval)
    data['MyLightDir'] = data['MyLightDir'].apply(ast.literal_eval)
    
    # Calculate the brightness of 'Cd'
    def calculate_brightness(rgb):
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    
    data['Brightness'] = data['Cd'].apply(calculate_brightness)
    
    # Extract individual components from 'MyOrient' and 'MyLightDir'
    data[['MyOrient_w', 'MyOrient_x', 'MyOrient_y', 'MyOrient_z']] = pd.DataFrame(data['MyOrient'].tolist(), index=data.index)
    data[['MyLightDir_x', 'MyLightDir_y', 'MyLightDir_z']] = pd.DataFrame(data['MyLightDir'].tolist(), index=data.index)
    
    # Drop the original columns to simplify the dataframe (optional, depending on your needs)
    data = data.drop(columns=['MyOrient', 'Cd', 'MyLightDir'])
    
    # Save the preprocessed and feature-engineered data to a new CSV file
    data.to_csv(output_file_path, index=False)
    print(f"Preprocessed data has been saved to {output_file_path}")

# Example usage
input_file_path = '/media/macha/HoudiniSamsungSs/Halo/data/attributes.csv' # Update this with the path to your input file
print(input_file_path)
output_file_path = '/media/macha/HoudiniSamsungSs/Halo/data/preprocessed_attributes.csv' # Define your output file name and path
preprocess_and_feature_engineer(input_file_path, output_file_path)

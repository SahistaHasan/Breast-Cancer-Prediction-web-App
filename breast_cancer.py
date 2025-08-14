import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open(r"trained_model.sav", 'rb'))

def breast_cancer_predict(input_data):
    
    input_data_as_numpy_array = np.array(input_data)
    reshaped_arr=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(reshaped_arr)
    print(prediction)
    if(prediction[0]==1):
        return "The breast cancer is benign"
    else:
        return "The breast cancer is malignant"
    

def main():
    st.title("Breast Cancer Prediction App")
    st.write("Fill in all the tumor features to predict whether it is **Benign** or **Malignant**.")
    st.markdown("---")

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["Mean Features", "Error Features", "Worst Features"])

    # ------------------ MEAN FEATURES ------------------
    with tab1:
        st.header("Mean Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            mean_radius = st.number_input("Mean Radius", min_value=0.0)
            mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0)
            mean_compactness = st.number_input("Mean Compactness", min_value=0.0)
            mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0)
        with col2:
            mean_texture = st.number_input("Mean Texture", min_value=0.0)
            mean_area = st.number_input("Mean Area", min_value=0.0)
            mean_concavity = st.number_input("Mean Concavity", min_value=0.0)
            mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0)
        with col3:
            mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0)
            mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0)

    # ------------------ ERROR FEATURES -----------------
    with tab2:
        st.header("Error Features")
        col4, col5, col6 = st.columns(3)
        with col4:
            radius_error = st.number_input("Radius Error", min_value=0.0)
            perimeter_error = st.number_input("Perimeter Error", min_value=0.0)
            compactness_error = st.number_input("Compactness Error", min_value=0.0)
            concave_points_error = st.number_input("Concave Points Error", min_value=0.0)
        with col5:
            texture_error = st.number_input("Texture Error", min_value=0.0)
            area_error = st.number_input("Area Error", min_value=0.0)
            concavity_error = st.number_input("Concavity Error", min_value=0.0)
            symmetry_error = st.number_input("Symmetry Error", min_value=0.0)
        with col6:
            smoothness_error = st.number_input("Smoothness Error", min_value=0.0)
            fractal_dimension_error = st.number_input("Fractal Dimension Error", min_value=0.0)

    # ------------------ WORST FEATURES -----------------
    with tab3:
        st.header("Worst Features")
        col7, col8, col9 = st.columns(3)
        with col7:
            worst_radius = st.number_input("Worst Radius", min_value=0.0)
            worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0)
            worst_compactness = st.number_input("Worst Compactness", min_value=0.0)
            worst_concave_points = st.number_input("Worst Concave Points", min_value=0.0)
        with col8:
            worst_texture = st.number_input("Worst Texture", min_value=0.0)
            worst_area = st.number_input("Worst Area", min_value=0.0)
            worst_concavity = st.number_input("Worst Concavity", min_value=0.0)
            worst_symmetry = st.number_input("Worst Symmetry", min_value=0.0)
        with col9:
            worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0)
            worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=0.0)
   


    diagnosis=''
    if st.button('Breast Cancer Test Result'):
        diagnosis=breast_cancer_predict([mean_radius, mean_texture, mean_perimeter, mean_area, 
        mean_smoothness, mean_compactness, mean_concavity, 
        mean_concave_points, mean_symmetry, mean_fractal_dimension, 
        radius_error, texture_error, perimeter_error, area_error, 
        smoothness_error, compactness_error, concavity_error, 
        concave_points_error, symmetry_error, fractal_dimension_error, 
        worst_radius, worst_texture, worst_perimeter, worst_area, 
        worst_smoothness, worst_compactness, worst_concavity, 
        worst_concave_points, worst_symmetry, worst_fractal_dimension
        ])

    st.success(diagnosis)


if __name__=='__main__':
    main()
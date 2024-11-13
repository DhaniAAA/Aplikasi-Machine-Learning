import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Menambahkan judul aplikasi
st.title('Aplikasi Klasifikasi Kualitas Susu Sapi dengan Machine Learning')

# Menambahkan sidebar
st.sidebar.header('Pilih Algoritma')

# Menambahkan dropdown menu di sidebar
option = st.sidebar.selectbox(
    'Pilih Algoritma',
    ['Algoritma KNN', 'Algoritma Decision Tree (DT)', 'Algoritma Naive Bayes']
)

# Menampilkan pilihan di halaman utama
st.write(f'Anda memilih {option}')

# Menampilkan penjelasan algoritma
if option == 'Algoritma KNN':
    st.write('K-Nearest Neighbors (KNN) adalah algoritma yang digunakan untuk klasifikasi atau regresi. KNN bekerja dengan mencari data terdekat dan memberikan label berdasarkan mayoritas label dari tetangga terdekat.')

elif option == 'Algoritma Decision Tree (DT)':
    st.write('Decision Tree adalah algoritma yang membuat model keputusan dalam bentuk pohon, dengan setiap cabang mewakili keputusan atau pernyataan dan setiap daun mewakili hasil akhir.')

else:
    st.write('Naive Bayes adalah algoritma klasifikasi berbasis probabilitas yang berdasarkan pada Teorema Bayes dengan asumsi independensi antara fitur-fitur.')

st.subheader('Upload Dataset')

# Fitur upload dataset
uploaded_file = st.file_uploader("Upload file dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Membaca dataset dari file yang di-upload
        df = pd.read_csv(uploaded_file)
        st.write('Preview Dataset:')

        # Menampilkan dataset dengan scroll
        st.dataframe(df, height=300)
        st.write(f'Jumlah Baris: {df.shape[0]}')
        st.write(f'Jumlah Kolom: {df.shape[1]}')

        # Menampilkan dropdown untuk memilih variabel fitur dan target
        columns = df.columns.tolist()
        target = st.selectbox('Pilih Kolom Target', options=columns)
        features = st.multiselect('Pilih Kolom Fitur', options=[col for col in columns if col != target])

        # Pembersihan dan Transformasi Data
        st.subheader('Pembersihan dan Transformasi Data')

        # Menangani nilai hilang
        if st.checkbox('Hapus Baris dengan Nilai Hilang'):
            missing_count_before = df.isnull().sum().sum()
            df = df.dropna()
            missing_count_after = df.isnull().sum().sum()
            st.write(f'Jumlah nilai hilang yang dihapus: {missing_count_before - missing_count_after}')

        if st.checkbox('Isi Nilai Hilang dengan Rata-Rata'):
            # Tampilkan informasi nilai hilang sebelum pengisian
            missing_before = df.isnull().sum()
            if missing_before.sum() > 0:
                st.write("Jumlah nilai hilang per kolom sebelum pengisian:")
                st.write(missing_before[missing_before > 0])

                # Pisahkan kolom numerik dan kategorikal
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                categorical_columns = df.select_dtypes(include=['object']).columns

                # Isi nilai hilang untuk kolom numerik dengan mean
                if len(numeric_columns) > 0:
                    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                    st.write("Kolom numerik diisi dengan nilai rata-rata")

                # Isi nilai hilang untuk kolom kategorikal dengan mode (nilai yang paling sering muncul)
                if len(categorical_columns) > 0:
                    for col in categorical_columns:
                        df[col] = df[col].fillna(df[col].mode()[0])
                    st.write("Kolom kategorikal diisi dengan nilai yang paling sering muncul")

                # Tampilkan informasi nilai hilang setelah pengisian
                missing_after = df.isnull().sum()
                if missing_after.sum() > 0:
                    st.write("Jumlah nilai hilang per kolom setelah pengisian:")
                    st.write(missing_after[missing_after > 0])
                else:
                    st.write("Semua nilai hilang telah diisi!")

        # Encoding Kategori
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        encode_cols = st.multiselect('Pilih Kolom untuk Encoding', options=categorical_cols)
        if encode_cols:
            le = LabelEncoder()
            for col in encode_cols:
                df[col] = le.fit_transform(df[col])

        st.write('Dataset Setelah Pembersihan dan Transformasi:')
        st.dataframe(df, height=300)
        st.write(f'Jumlah Baris: {df.shape[0]}')
        st.write(f'Jumlah Kolom: {df.shape[1]}')

        # Memilih proporsi data latih dan uji
        test_size = st.slider('Pilih Proporsi Data Uji (%)', min_value=10, max_value=90, value=20)

        if st.button('Latih Model'):
            if target and features:
                X = df[features]
                y = df[target]

                # Membagi data menjadi data latih dan data uji
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

                # Standarisasi fitur
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Menyiapkan model sesuai algoritma yang dipilih
                if option == 'Algoritma KNN':
                    model = KNeighborsClassifier()
                elif option == 'Algoritma Decision Tree (DT)':
                    model = DecisionTreeClassifier()
                else:
                    model = GaussianNB()

                # Melatih model
                model.fit(X_train, y_train)

                # Memprediksi hasil
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)

                st.write(f'Akurasi Model: {accuracy:.2f}')
                st.subheader('Classification Report')
                st.text(classification_report(y_test, y_pred))

                st.subheader('Confusion Matrix')
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                st.pyplot(fig)

                # Menampilkan beberapa contoh prediksi
                st.subheader('Contoh Prediksi')
                examples = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
                st.write(examples.head())

            else:
                st.error('Pilih kolom target dan fitur dengan benar.')

    except pd.errors.EmptyDataError:
        st.error('File kosong atau format tidak valid.')
    except Exception as e:
        st.error(f'Error: {e}')

# Add after st.dataframe(df, height=300) line

if st.checkbox('Show Dataset Description'):
    st.subheader('Dataset Description')
    st.write(df.describe())

    # Display missing values information
    st.write('Missing Values:')
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

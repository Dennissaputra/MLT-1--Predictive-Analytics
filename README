# Laporan Proyek Machine Learning - Dennis Saputra Ariansyah
## Domain Proyek

Topik yang diangkat dari proyek ini yaitu mengenai kemungkinan pasien terkena stroke, maka dari itu rumah sakit dapat memprediksi pasien yang kemungkinan terkena stroke.

###  Latar Belakang
Menurut Organisasi Kesehatan Dunia (WHO) stroke adalah penyebab kematian ke-2 secara global, bertanggung jawab atas sekitar 11% dari total kematian. Di indonesia pun penyebab kematian terbesar salah satunya karena stroke[[1]](https://news.detik.com/berita-jawa-barat/d-5268472/polres-karawang-bongkar-aksi-penipu-berkedok-jual-smartphone-murah). mengutip pada jurnal Ilmiah Kedoktoran "Beberapa faktor risiko yang paling penting adalah hipertensi, merokok, dislipidemia, diabetes mellitus, obesitas, dan penyakit jantung"[[2]](http://jurnal.untad.ac.id/jurnal/index.php/MedikaTadulako/article/view/12337/9621).

dalam proyek ini akan dibuat beberapa model machine learning dan mengevaluasi model mana yang cocok untuk prediksi kemungkinan seorang pasien terkena stroke berdasarkan parameter input seperti jenis kelamin, usia, berbagai penyakit, dan status merokok. Setiap baris dalam data memberikan informasi yang relevan tentang pasien.

## Business Understanding
### Problem Statements

-   Bagaimana cara melakukan pra-pemrosesan data agar dapat digunakan pada model machine learning?
- bagaimana cara membuat model machine learning untuk klasifikasi kemungkinan pasien terkena stroke?

### Goals

-   Melakukan  _pra-pemrosesan_  data agar dapat digunakan pada model machine learning.
-   Membuat model machine learning untuk menclassifikasi kemungkinan pasien terkena stroke yang memiliki tingkat akurasi > 75%.

### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
-   _Pra-pemrosesan_  dapat dilakukan beberapa teknik sebagai berikut.
    
    -   Melakukan  _Categorical Encoding_  sebagai proses mengubah data kategori menjadi data numerik menggunakan One-Hot Encoding
    -   Melakukan  _Split Data_  dengan membagi 2 dataset sebagai data latih (train data) dan data test (test data) dengan perbandingan rasio 80% : 20%.
    -   Melakukan standardisasi data pada fitur numerik dengan  _StandarScaler_.
 - untuk pembuatan model akan digunakan 2 algoritma sebagain perbandingan, diantaranya :
    
   - Algortima Random Forest
	  Pembuatan model pertama dengan algoritma Random Forest. _Random Forest_ bergantung pada sebuah nilai vector random dengan distribusi yang sama pada semua pohon yang masing masing _decision tree_ memiliki kedalaman yang maksimal. _Random forest_ adalah _classifier_ yang terdiri dari _classifier_ yang berbentuk pohon {h(**x**, θ k ), k = 1, . . .} dimana θ_k_ adalah random vector yang diditribusikan secara independen dan masing masing tree pada sebuah unit kan memilih class yang paling popular pada input x[[3]](https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest/)
![Penerapan Data Science pada Marketing (Customer Churn Prediction-Python) -  Part 2 | by Hafiz Ma'ruf | Medium](https://miro.medium.com/max/1170/1*VY3lEFysaQ0nnV_zkxyU-w.png)

   - Algoritma Decission Tree
    kedua yaitu menggunakan algortima dari decision tree. decision tree yang merupakan algoritma yang memprediksi nilai variabel target dengan mengikuti aturan keputusan sederhana dari fitur data yang tersedia[[4]](http://learningbox.coffeecup.com/05_1_decisiontree.html). _Decision Tree_ memiliki bentuk seperti pohon, dimana _tree_ memiliki node akar (_root node_), _decision node_ dan node daun (_leaf node_). _Leaf node_ adalah node akhir yang tidak dapat dipecah dan yang akan menentukan hasil prediksi _decision tree[[5]](https://student-activity.binus.ac.id/himmat/2022/06/decision-tree-in-machine-learning/).
    ![Macam-Macam Algoritma Klasifikasi Machine Learning yang Pent...](https://dqlab.id/files/dqlab/cache/a728e959354ea130b88747e547a4d47e_100_persen.png)

    

## Data Understanding
![Screenshot 2022-10-02 161354.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20161354.png?raw=true)
informasi terkait dataset || Sumber | Kaggle Dataset : [Stroke Prediction Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) || Jenis & Ukuran | CSV (69kb)|

Pada berkas yang diunduh yakni [healthcare-dataset-stroke-data.csv](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) terdapat 2.000 baris (jumlah pengamatan) dan 12 kolom dalam dataset. Berdasarkan informasi dari dataset, variabel pada Stroke Prediction Dataset sebagai berikut.
1) id: unique identifier  
2) gender: "Male", "Female" or "Other"  
3) age: age of the patient  
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension  
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease  
6) ever_married: "No" or "Yes"  
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"  
8) Residence_type: "Rural" or "Urban"  
9) avg_glucose_level: average glucose level in blood  
10) bmi: body mass index  
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*  
12) stroke: 1 if the patient had a stroke or 0 if not![](https://raw.githubusercontent.com/Dennissaputra/MLT-1/main/Screenshot%202022-10-01%20185004.png)
dari gambar di di jelaskan di dalam data terdapat ada 8 kategori bertipe object dan 3 data numerik bertipe float64, visualiasasi data pada kategori sebagai berikut:

### Export a file

You can export the current file by clicking **Export to disk** in the menu. You can choose to export the file as plain Markdown, as HTML using a Handlebars template or as a PDF.
![Screenshot 2022-10-02 125956.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20125956.png?raw=true)
![Screenshot 2022-10-02 130009.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130009.png?raw=true)

![Screenshot 2022-10-02 130023.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130023.png?raw=true)
![Screenshot 2022-10-02 130033.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130033.png?raw=true)
![Screenshot 2022-10-02 130046.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130046.png?raw=true)
![Screenshot 2022-10-02 130057.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130057.png?raw=true)
![Screenshot 2022-10-02 130109.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130109.png?raw=true)
![Screenshot 2022-10-02 130120.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130120.png?raw=true)

untuk visuallisasi dari numerik yaitu :
![Screenshot 2022-10-02 130924.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130924.png?raw=true)

untuk melihat hubungan antar fitur numerik dengan fungsi sebagai berikut:
![Screenshot 2022-10-02 130948.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20130948.png?raw=true)

Untuk visualisasi heatmap (korelasi numeric features) adalah sebagai berikut:
![Screenshot 2022-10-02 131237.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20131237.png?raw=true)
Keterangan heatmap:

-   Semakin mendekati 1 maka semakin tinggi korelasi antar fitur numerik
-   Semakin mendekati 0 maka korelasi antar fitur numerik mendekati netral
-   Semakin mendekati -1 maka semakin rendah korelasi antar fitur numerik

## Data preparetion

berikut tahapan yang di lakukan pada data preperation:

-  Melakukan  _Categorical Encoding_  Digunakan sebagai proses mengubah data kategori menjadi data numerik. Untuk teknik  _Encoding_  fitur kategori menggunakan  _One-Hot Encoding_.  _One-Hot Encoding_  untuk data nominal. Data nominal diklasifikasikan tanpa urutan atau peringkat.
- melakukan Split Data, dataset dibagi menjadi dua bagian yaitu data latih dan data uji masing masing di bagi dengan rasio 90:10. proses split data ini di lakukan dengan modul [train_test_split](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) dari library skkit-learn.
-    Melakukan standarisasi pada data latih dengan menggunakan StandardScaler dari library [sckit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

## Modeling

setelah di lakukanya data preparation, data yaang sudah siap akan di gunakan untuk di coba pada model, di sini akan dibuat 2 model sebagai bahan berbandingan.

- dibuat model dengan menggunakan algoritma [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),  algoritma Random Forest ini salah satu algoritma dari klasifikasi yang dimana Algoritma ini memberikan akurasi yang bagus dalam klasifikasi, dapat menangani data training yang jumlahnya besar, dan juga efektif untuk mengatasi data yang tidak lengkap.
- dibuat model dengan menggunakan algoritma [DecissionTree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).  alasan menggunakan algoritma Decision Tree ini adalah salah satu algortima klasifikasi objek yang mudah di pahami juga algoritma ini bekerja dengan melakukan klasifikasi berdasarkan atribut yang paling membedakan
## Evaluasi

pada proses evaluasi ini saya menggunakan *Classification Report*

- *Classification Report* 
*Classification Report* ini digunakan untuk mengukur kualitas prediksi dari algoritma klasifikasi. *Classification report* menampilkan nilai precision, recall, f1-score, dan support untuk model
  
  - Model *RandomForest*
  ![Screenshot 2022-10-02 144728.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20144728.png?raw=true)
   - Model *DecisionTree*
   ![Screenshot 2022-10-02 144746.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20144746.png?raw=true)
   
dari model yang sudah di buat di atas secara keseluruhan dapat kita simpulkan bahwa :

1) Pada model RandomForest memiliki nilai yang sangat baik yaitu 97,27% dan f1 score, _recall_, serta _precision_ cukup baik. itu menandakan dataset ini memiliki tinggkat akurasi yang baik dengan menggunakan algoritma *RandomForest*.
2) Pada model *DecisionTree* memiliki tingkat akurasi yaitu 0,06% yang menandakan model ini cocok untuk dataset yang kita gunakan.

rumus dari *Classifikasi Report* antara lain :

- *Precision*
_Precision_ adalah metrik dalam kasus klasifikasi, yang digunakan untuk menghitung efek model dalam memprediksi label positif terhadap semua label positif model. Jadi bagaimana cara menghitungnya, pertama kita perlu memahami istilah TP, TN, FP, FN. 
![Screenshot 2022-10-02 150153.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20150153.png?raw=true)
- *Recall*
_Recall_ adalah metrik dalam kasus klasifikasi, yang digunakan untuk menghitung efek model dalam memprediksi label positif untuk semua label data positif.
![Screenshot 2022-10-02 150206.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20150206.png?raw=true)
- f1-score
_f1-score_ merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik hasil prediksi model (precision) dan seberapa lengkap hasil prediksinya (recall).
![Screenshot 2022-10-02 150218.png](https://github.com/Dennissaputra/MLT-1/blob/main/Screenshot%202022-10-02%20150218.png?raw=true)

## Refrensi
[1] https://news.detik.com/berita-jawa-barat/d-5268472/polres-karawang-bongkar-aksi-penipu-berkedok-jual-smartphone-murah
[[2]](http://jurnal.untad.ac.id/jurnal/index.php/MedikaTadulako/article/view/12337/9621)Mutiarasari, D. (2019). Ischemic Stroke: Symptoms, Risk Factors, and Prevention. Jurnal Ilmiah Kedokteran Medika Tandulako, 1(1), 60–73.
[3]()https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest/
[4]http://learningbox.coffeecup.com/05_1_decisiontree.html
[5]https://student-activity.binus.ac.id/himmat/2022/06/decision-tree-in-machine-learning/

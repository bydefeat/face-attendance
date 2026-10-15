# He thong Diem danh Sinh vien bang Nhan dien Khuon mat (Face Attendance)
> Do an Mon hoc: Hoc May (Machine Learning)
> Giang vien huong dan: TS. Ngo Quoc Viet - Truong Dai hoc Su pham TP.HCM (HCMUE)

---

## 1. Gioi thieu de tai va Muc tieu
De tai tap trung vao viec ap dung cac mo hinh Hoc may co dien (Classical Machine Learning) ket hop ky thuat trich xuat dac trung Eigenfaces (PCA) de giai quyet bai toan nhan dien khuon mat va tu dong hoa diem danh sinh vien.

Du an tuan thu nghiem ngat noi dung chuong trinh hoc, khong su dung cac mo hinh Deep Learning pretrained co san (nhu FaceNet hay InsightFace), nham the hien ro ban chat toan hoc va su so sanh giua cac thuat toan duoc giang day trong mon hoc.

---

## 2. Anh xa ly thuyet vao Bai giang Mon hoc

| Giai doan | Noi dung thuc hien | Bai giang ap dung |
| :--- | :--- | :--- |
| Tien xu ly du lieu | Crop mat bang Haar Cascade, Grayscale, Resize (100x100), Histogram Equalization, Chuan hoa diem anh ve doan [0, 1] | ML-02-DATAVISUAL |
| Giam chieu du lieu | Trich xuat dac trung Eigenfaces bang Principal Component Analysis (PCA) | ML-02 & ML-04 |
| Huan luyen & Phan loai | - k-Nearest Neighbors (k-NN voi k=3, k=5)<br>- Support Vector Machine (Linear SVM, RBF SVM)<br>- Decision Tree (C4.5/CART)<br>- Ensemble: Random Forest & AdaBoost<br>- Gaussian Naive Bayes<br>- Multinomial Logistic Regression | ML-03, ML-04, ML-06, ML-07 |
| Danh gia mo hinh | Confusion Matrix (Ma tran nham lan), Accuracy, Precision, Recall, F1-Score, Thoi gian suy luan (Inference Latency) | ML-04-CLASSIFICATION |
| Trien khai thuc te | He thong diem danh real-time qua Webcam, tu dong ghi nhan file attendance.csv kem thoi gian check-in | Ung dung thuc te mon hoc |

---

## 3. Cau truc thu muc Du an

```text
face-attendance/
|-- ML-00-SYLLABUS.pdf             # De cuong mon hoc
|-- ML-01-INTRO.pdf                # Tong quan Machine Learning
|-- ML-02-DATAVISUAL.pdf           # Xu ly & truc quan hoa du lieu
|-- ML-03-REGRESSIONS.pdf          # Hoi quy tuyen tinh & Logistic
|-- ML-04-CLASSIFICATION.pdf       # Bai toan phan loai & SVM
|-- ML-06-KNN&TREE&BOOSTING.pdf    # k-NN, Decision Trees & Boosting
|-- ML-07-NB.pdf                   # Naive Bayes Classifier
|-- dataset/                       # Thu muc chua anh khuon mat tung sinh vien
|-- models/                        # Luu mo hinh PCA va mo hinh Classifier tot nhat
|-- notebooks/
|   `-- ml_experiments.ipynb       # Jupyter Notebook phan tich & ve bieu do bao cao
|-- reports/                       # Luu ma tran nham lan va ket qua benchmark (CSV/PNG)
|-- src/
|   |-- capture_faces.py           # Thu thap mau khuon mat tu webcam
|   |-- extract_features.py        # Trich xuat dac trung Eigenfaces (PCA)
|   |-- train_and_evaluate.py      # Huan luyen va doi chuan cac thuat toan
|   `-- attendance_app.py          # Ung dung diem danh thoi gian thuc
|-- attendance.csv                 # Nhat ky diem danh tu dong
|-- requirements.txt               # Danh sach thu vien can thiet
`-- README.md                      # Tai lieu huong dan
```

---

## 4. Huong dan cai dat va Thuc thi

### Buoc 1: Cai dat moi truong va thu vien
Yeu cau Python 3.10 tro len. Cai dat cac goi thu vien phu hop:
```bash
pip install -r requirements.txt
```

### Buoc 2: Thu thap du lieu khuon mat sinh vien
Chay script sau de chup anh tung sinh vien (khoang 40 - 50 anh voi cac goc nghieng nhe):
```bash
# Vi du thu thap cho sinh vien 1:
python src/capture_faces.py --id 4901104001 --name Nguyen_Van_A --samples 50

# Vi du thu thap cho sinh vien 2:
python src/capture_faces.py --id 4901104002 --name Tran_Thi_B --samples 50
```

### Buoc 3: Huan luyen va Danh gia doi chuan cac thuat toan
Script se tu dong:
1. Trich xuat vector dac trung Eigenfaces bang PCA.
2. Huan luyen tat ca cac thuat toan (k-NN, SVM, Decision Tree, Random Forest, AdaBoost, Naive Bayes, Logistic Regression).
3. Tinh toan Precision, Recall, F1-Score, ve Confusion Matrix luu vao `reports/confusion_matrix.png`.
4. Xuat bang so sanh vao `reports/benchmark_results.csv`.
5. Luu mo hinh co F1-Score cao nhat vao `models/best_classifier.pkl`.

```bash
python src/train_and_evaluate.py
```

### Buoc 4: Chay chuong trinh diem danh thoi gian thuc
Mo webcam de he thong tu dong nhan dien sinh vien va ghi danh vao `attendance.csv`:
```bash
python src/attendance_app.py
```
Nhan phim `q` tren ban phim de ket thuc chuong trinh.

---

## 5. Thuc nghiem tren Jupyter Notebook
De xem chi tiet cac bieu do truc quan hoa du lieu, do thi Elbow phuong sai tich luy PCA va hinh anh truc quan cac vector Eigenfaces, khoi chay:
```bash
jupyter notebook notebooks/ml_experiments.ipynb
```

import os # Veriyi okumak için
import numpy as np # Matematiksel işlemler 
import pandas as pd # Veri setinin okunması ve gözlemlenmesi için
import seaborn as sns #İstatistiksel veri görselleştirme kütüphanesi
import tensorflow as tf # Modeli kurmak için
import matplotlib.pyplot as plt # Veriyi ve modeli incelemek için 

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
balıklar = '/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset' 
sınıf = [i for i in os.listdir(balıklar) if '.' not in i]                    
sınıf

#Buradaki kod satırı ile balıklar veri setindeki balık sınıflarını görmüş oluyoruz
etiket = []
yol = []

for dir_name, _,filenames in os.walk(balıklar):                    
    for filename in filenames:                                 
        etiket.append(os.path.split(dir_name)[-1])     
        yol.append(os.path.join(dir_name,filename))  

data = pd.DataFrame(columns=['yol','etiket']) 
data['yol']=yol
data['etiket']=etiket
#değişkenlere yol ve tiket ataması yapar
data['etiket'].value_counts()
#Buradaki value_counts ifadesi pandas kütüphandesinden gelip veride hangi etiketden kaç tane veri olduğunu gösterir. 
data.head()
#İlk 5 data hakkında bilgi
data.yol[0]
#İlk veriyi tanımamıza yol bakımından tanıma
batch_size = 128
img_boyu = 224
img_genişlik = 224
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)  
#Veriler tam sayı şeklinde verileri daha hızlı şeklide öğrenmek açısından bu verileri 255 bölüp 0 ila 1 arasına sıkıştıralım
eğitim_verisi = train_datagen.flow_from_directory(
    directory=balıklar,
    target_size=(img_boyu, img_genişlik),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical')

test_verisi = train_datagen.flow_from_directory(
    directory=balıklar,
    target_size=(img_boyu, img_genişlik),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical')
#Kaç sınıf var onu değişken atayalım ve öğrenelim 
sınıf_sayısı= len(eğitim_verisi.class_indices)
sınıf_sayısı
model = tf.keras.models.Sequential([
  # Giriş katmanımız 
    
  tf.keras.layers.InputLayer((img_boyu, img_genişlik, 3)),
  tf.keras.layers.Flatten(),
    # Gizli Katmalarımız
  tf.keras.layers.Dense(512, activation='relu'),#Relu fonksiyonunu kullanma sebebimizi yukarıda belirtmiştik
   tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu'),
   tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.2),
  # Çıkış katamnı (Burada 9 nöon olmasının sebebi yukarıda bakktığımız gibi 9 farklı katogori sınıflandırma yapıcağımız için)
  tf.keras.layers.Dense(9,activation="softmax")
])

#Model eğitimi için derleme kısmı
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary() #Model özeti
Callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3) 
#Callback apısı kullanma sebebimiz değerlerimizi izlemek, ve eğitim sırasında iç durum ve istatistik izlemek için bunu da periyodik olarak değerleri diske kaydederek yapar
history  = model.fit(eğitim_verisi,
                validation_data  = test_verisi,
                epochs = 100, callbacks=Callback)
results = model.evaluate(eğitim_verisi)
print("LOSS:  " + "%.4f" % results[0])
print("ACCURACY:  " + "%.2f" % results[1]) 
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Egitim Lossu')
plt.plot(history.history['val_loss'], label='Test Lossu')
plt.legend()
plt.title('Model Lossu')
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Egitim Dogrulugu')
plt.plot(history.history['val_accuracy'], label='Test Dogrulugu')
plt.legend()
plt.title('Model Dogurlugu'

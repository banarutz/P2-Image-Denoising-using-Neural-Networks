'''
def reshape (x,y):
    return tf.reshape(x, [250*250*3]), tf.reshape(y, [250*250*3])

if __name__ == "__main__":

    train_generator = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\TEST", stride= (250,250),
                           patch_size= (250,250), train_test_split_percentage =0.8, train = 'train')

    val_generator = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\TEST", stride= (250,250),
                           patch_size= (250,250), train_test_split_percentage =0.8, train = 'validation')


    # print(train_data)

    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(250, 250, 3),
                                                                                             dtype=tf.float32),
                                                                               tf.TensorSpec(shape=(250, 250, 3),
                                                                                             dtype = tf.float32)))
    train_ds = train_ds.map(lambda x,y: reshape(x,y))

    val_ds =  tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(250, 250, 3),
                                                                                             dtype=tf.float32),
                                                                               tf.TensorSpec(shape=(250, 250, 3),
                                                                                             dtype = tf.float32)))

    val_ds = val_ds.map(lambda x,y: reshape(x,y))


    # AVEM 30053 DE PATCHURI DE ANTRENAMENT #

    train_ds = train_ds.shuffle(buffer_size=320)
    val_ds = val_ds.shuffle(buffer_size=320)
    train_ds = train_ds.batch(batch_size=32)
    val_ds = val_ds.batch(batch_size=32)


    input = tf.keras.layers.Input(shape = (250*250*3))
    x0 = tf.keras.layers.Flatten()(input)
    x1 = tf.keras.layers.Dense(units = 100, activation = 'relu') (x0)
    out = tf.keras.layers.Dense(units = 250*250*3) (x1)

    model = tf.keras.models.Model(input, out)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse']
    )
    callback = [tf.keras.callbacks.ModelCheckpoint('Dense_1st_try.h5', monitor='val_mse', save_best_only=True, verbose=1)]
    model_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks = callback)
    model.summary()

'''

'''
if __name__ == "__main__":
    train_generator = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\Patches_size_256",
                                 stride= (256,256), patch_size= (256,256), train_test_split_percentage =0.8,
                                 train = 'train')

    val_generator = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\Patches_size_256",
                               stride= (256,256), patch_size= (256,256), train_test_split_percentage =0.8,
                               train = 'validation')

    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
                                                                                               dtype=tf.float32),
                                                                                 tf.TensorSpec(shape=(256, 256, 3),
                                                                                               dtype=tf.float32)))

    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
                                                                                           dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(256, 256, 3),
                                                                                           dtype=tf.float32)))

    train_ds = train_ds.shuffle(buffer_size=800)
    val_ds = val_ds.shuffle(buffer_size=800)
    train_ds = train_ds.batch(batch_size=64)
    val_ds = val_ds.batch(batch_size=64)



    latent_dimension = 16

    input = tf.keras.layers.Input(shape=(256, 256, 3))
    x0 = tf.keras.layers.Conv2D(filters = 64, kernel_size=3, strides=(2, 2), activation='relu', padding = 'same')(input)
    x1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding = 'same')(x0)
    x2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(4, 4), activation='relu', padding = 'same')(x1)
    #x3 = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x2)
    x4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(4, 4), activation='relu')(x2)
    #x5 = tf.keras.layers.Flatten()(x4)

#LatentSpace:
    #x6 = tf.keras.layers.Dense(units = latent_dimension, activation = 'relu')(x5)

#Decoder:
    #x7 = tf.keras.layers.InputLayer(input_shape=(latent_dimension,))(x6)
    # x6 = tf.keras.layers.Dense(units=250 * 250 * 3, activation=tf.nn.relu)(x5)
    x8 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, strides=4, padding='same',
                                     activation='relu') (x4)
    x9 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same',
                                          activation='relu')(x8)
    x10 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                                     activation='relu') (x9)
    x11 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                          activation='relu', output_padding= (1,1))(x10)
    x12 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                          activation='relu', output_padding= (1,1))(x11)
    x13 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same',
                                          activation='sigmoid')(x12)
    
    model = tf.keras.models.Model(input, x13)

    model.compile(
        optimizer='SGD',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mean_squared_error']
    )

    callback = [tf.keras.callbacks.ModelCheckpoint('512FILTRE.h5', monitor='val_mean_squared_error', save_best_only=True, verbose=1)]
    model_history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks = callback)
    print(model.summary())



'''
'''
pozik = plt.imread('NOISY_SRGB_010.PNG')
lista_patchuri = patching(pozik, (250,250), (250,250))
new_model = tf.keras.models.load_model('v2.h5')

print(len(lista_patchuri))



lista_predictii=[]
#for i in lista_patchuri:
    #print (np.shape(i))
predictie = new_model.predict(lista_patchuri)
    #lista_predictii.append(predictie)

#poza_refacuta = tf.reshape (predictie, shape = (3250,4250,3))
poza_refacuta = restore_image(predictie, (250, 250), np.shape(pozik))
plt.figure()
plt.imshow(poza_refacuta)

plt.show()
'''




###################33 RETEAUA 3 ####################

# if __name__ == "__main__":
#     train_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patches_size_256",
#                                 stride=(256, 256), patch_size=(256, 256), train_test_split_percentage=0.8,
#                                 train='train')
#
#     val_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patches_size_256",
#                               stride=(256, 256), patch_size=(256, 256), train_test_split_percentage=0.8,
#                               train='validation')
#
#     train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
#                                                                                                dtype=tf.float32),
#                                                                                  tf.TensorSpec(shape=(256, 256, 3),
#                                                                                                dtype=tf.float32)))
#
#     val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
#                                                                                            dtype=tf.float32),
#                                                                              tf.TensorSpec(shape=(256, 256, 3),
#                                                                                            dtype=tf.float32)))
#
#     train_ds = train_ds.shuffle(buffer_size=800)
#     val_ds = val_ds.shuffle(buffer_size=800)
#     train_ds = train_ds.batch(batch_size=64)
#     val_ds = val_ds.batch(batch_size=64)
#
#
#     input = tf.keras.layers.Input(shape=(256, 256, 3))
#     x1 = tf.keras.layers.Conv2D(filters = 32,kernel_size=2, strides=(2, 2), activation='relu')(input)
#     x2 = tf.keras.layers.MaxPooling2D((2,2))(x1)
#
#
#     x3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides= 4, activation = 'relu')(x2)
#     out = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1, strides=1,
#                                           activation='sigmoid')(x3)
#
#     ### ACHTUNG: Trebuie refacute patch-urile la o dimensiune gen 384, trebuie sa intreb ca nu se impart exact si da o poza
#     # dubioasa si gresita
#
#     model = tf.keras.models.Model(input, out)
#
#     model.compile(
#         optimizer='SGD',
#         loss=tf.keras.losses.MeanSquaredError(),
#         metrics=['mean_squared_error']
#     )
#
#     callback = [
#         tf.keras.callbacks.ModelCheckpoint('RETEAUA3.h5', monitor='val_mean_squared_error', save_best_only=True,
#                                            verbose=1)]
#
#
#    #model_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callback)
#     print(model.summary())



###################33 RETEAUA 4 ####################

# if __name__ == "__main__":
#     train_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patches_size_32",
#                                 stride=(256, 256), patch_size=(256, 256), train_test_split_percentage=0.8,
#                                 train='train')
#
#     val_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patches_size_256",
#                               stride=(256, 256), patch_size=(256, 256), train_test_split_percentage=0.8,
#                               train='validation')
#
#     train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
#                                                                                                dtype=tf.float32),
#                                                                                  tf.TensorSpec(shape=(256, 256, 3),
#                                                                                                dtype=tf.float32)))
#
#     val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
#                                                                                            dtype=tf.float32),
#                                                                              tf.TensorSpec(shape=(256, 256, 3),
#                                                                                            dtype=tf.float32)))
#
#     train_ds = train_ds.shuffle(buffer_size=800)
#     val_ds = val_ds.shuffle(buffer_size=800)
#     train_ds = train_ds.batch(batch_size=64)
#     val_ds = val_ds.batch(batch_size=64)
#
#     model = tf.keras.models.Sequential()
#     # encoder network
#     model.add(tf.keras.layers.Conv2D(30, 4, activation='relu', padding='valid', input_shape=(256, 256, 3)))
#     model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
#     model.add(tf.keras.layers.Conv2D(15, 2, activation='relu', padding='valid'))
#     model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
#     # decoder network
#     model.add(tf.keras.layers.Conv2DTranspose(15, 2, activation='relu', padding='valid'))
#     model.add(tf.keras.layers.UpSampling2D(2))
#     model.add(tf.keras.layers.Conv2DTranspose(30, 2, activation='relu', padding='valid'))
#     model.add(tf.keras.layers.UpSampling2D(2))
#     model.add(tf.keras.layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='valid'))  # output layer
#     model.compile(optimizer='adam', loss='mse')
#     print(model.summary())
#
#     callback = [
#         tf.keras.callbacks.ModelCheckpoint('RETEAUAv4.1.h5', monitor='val_loss', save_best_only=True,
#                                            verbose=1)]
#
#     #model_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callback)


#################3 RETEAUA 5 - PATCHURI DE 32 X 32 ####################



# if __name__ == "__main__":
#     train_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patches_size_32",
#                                 stride=(32, 32), patch_size=(32, 32), train_test_split_percentage=0.8,
#                                 train='train')
#
#     val_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patches_size_32",
#                               stride=(32, 32), patch_size=(32, 32), train_test_split_percentage=0.8,
#                               train='validation')
#
#     train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(32, 32, 3),
#                                                                                                dtype=tf.float32),
#                                                                                  tf.TensorSpec(shape=(32, 32, 3),
#                                                                                                dtype=tf.float32)))
#
#     val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(32, 32, 3),
#                                                                                            dtype=tf.float32),
#                                                                              tf.TensorSpec(shape=(32, 32, 3),
#                                                                                            dtype=tf.float32)))
#
#     train_ds = train_ds.shuffle(buffer_size=800)
#     val_ds = val_ds.shuffle(buffer_size=800)
#     train_ds = train_ds.batch(batch_size=64)
#     val_ds = val_ds.batch(batch_size=64)
#
#     model = tf.keras.models.Sequential()
#     # encoder network
#     model.add(tf.keras.layers.Conv2D(30, 4, activation='relu', padding='valid', input_shape=(32, 32, 3)))
#     model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
#     model.add(tf.keras.layers.Conv2D(15, 2, activation='relu', padding='valid'))
#     model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
#     # decoder network
#     model.add(tf.keras.layers.Conv2DTranspose(15, 2, activation='relu', padding='valid'))
#     model.add(tf.keras.layers.UpSampling2D(2))
#     model.add(tf.keras.layers.Conv2DTranspose(30, 2, activation='relu', padding='valid'))
#     model.add(tf.keras.layers.UpSampling2D(2))
#     model.add(tf.keras.layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='valid'))  # output layer
#     model.compile(optimizer='adam', loss='mse')
#     print(model.summary())
#
#     callback = [
#         tf.keras.callbacks.ModelCheckpoint('RETEAUAv4.1.h5', monitor='val_loss', save_best_only=True,
#                                            verbose=1)]
#
#     #model_history = model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=callback)
#
#
#     #pozik = plt.imread('NOISY_SRGB_010.PNG')
#     #pozik = norm_image(pozik)
#     pozik = plt.imread('NOISY_SRGB_010.PNG')
#     poza_GT = plt.imread("GT_SRGB_010.PNG")
#     lista_patchuri = patching(pozik, (32, 32), (32, 32))
#     lista_patchuri_GT = patching(poza_GT, (32, 32), (32, 32))
#     new_model = tf.keras.models.load_model('RETEAUAv4.1.h5')
#
#     print(len(lista_patchuri))
#     # pozik_buna = restore_image(lista_patchuri, (32, 32), np.shape(pozik))
#     # plt.figure()
#     # plt.imshow(pozik_buna)
#
#     plt.show()
#
#     ##### => O imagine are 264 de patch-uri. LATELY AM AFLAT CA NU TOATE AU 264 PATCHURI, DIFERA.
#
#     lista_predictii = []
#     # for i in lista_patchuri:
#     # print (np.shape(i))
#     predictie = new_model.predict(lista_patchuri)
#
#     # lista_predictii.append(predictie)
#
#     # poza_refacuta = tf.reshape (predictie, shape = (3250,4250,3))
#     poza_refacuta = restore_image(predictie, (32, 32), np.shape(pozik))
#
#     print(poza_refacuta)
#     #restored_image = tf.reshape(predictie,(2048,6144,3))
#     # plt.figure()
#     # plt.imshow(predictie[1000])
#     # plt.show()
#
#     # plt.figure()
#
#
#     fig, axs = plt.subplots(3, 3)
#     for i in range(3):
#         for j in range(3):
#             axs[i, j].imshow(lista_patchuri_GT[40**2*i + 100*j])
#     plt.show()
#
#     # fig, axs = plt.subplots(2, 2)
#     # for col in range(2):
#     #     for row in range(2):
#     #         ax = axs[row, col]
#     #         pcm =  ax.pcolormesh(predictie[1])
#     #         fig.colorbar(pcm, ax=ax)
#     # plt.show()
#
#
#     MSEn, MAEn, SNRn, PSNRn = restoration_metrics(poza_GT, pozik)
#     MSE, MAE, SNR, PSNR = restoration_metrics(poza_GT, poza_refacuta)
#
#     print ("MSE NOISY - GT {0}, MAE NOISY - GT {1}, SNR NOISY - GT {2} \nMSE GT - PREDICT {3},"
#            " MAE GT - PREDICT {4}, SNR GT - PREDICT {5}".format(MSEn, MAEn, SNRn, MSE, MAE, SNR ))



#################3 RETEAUA 6 - PATCHURI DE 32 X 32 ####################

# Generare de patch-uri

# gen = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Small_Dataset\\", save_path= "D:\\Small_Dataset_256",
#                        stride= (256,256), patch_size= (256,256), train_test_split_percentage =0.8, train = 'train')
# gen.load_data()


########## ASTA CRED CA ERA
'''
Modelul se numeste: [model bun] TEST v3 32 x 32 - Small Dataset 10 epochs.h5
Este antrenat pe Small Dataset
Patch 100 x 100 cu 50 x 50 stride
'''

'''
if __name__ == "__main__":
    # sebastian = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\100 x 100 FARA STRIDE FULL SET GT si NOISY",
    #                        stride= (100,100), patch_size= (100,100), train_test_split_percentage =0.8, train = 'train')
    # sebastian.load_data()

    train_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\100 x 100 FARA STRIDE FULL SET GT si NOISY",
                                stride=(100,100), patch_size=(100,100), train_test_split_percentage=0.8,
                                train='train')

    val_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\100 x 100 FARA STRIDE FULL SET GT si NOISY",
                              stride=(100,100), patch_size=(100,100), train_test_split_percentage=0.8,
                              train='validation')

    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(100,100, 3),
                                                                                               dtype=tf.float32),
                                                                                 tf.TensorSpec(shape=(100,100, 3),
                                                                                               dtype=tf.float32)))

    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(100,100, 3),
                                                                                           dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(100,100, 3),
                                                                                           dtype=tf.float32)))

    # train_ds = train_ds.shuffle(buffer_size=400)
    # val_ds = val_ds.shuffle(buffer_size=400)
    train_ds = train_ds.batch(batch_size=16)
    val_ds = val_ds.batch(batch_size=16)

    model = tf.keras.models.Sequential()
    # encoder network
    model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size= 3, activation='relu', padding='same', input_shape=(100,100, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Conv2D(filters = 32,kernel_size= 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    #model.add(tf.keras.layers.MaxPooling2D(5))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    #model.add(tf.keras.layers.AveragePooling2D(2, padding='valid'))
    # decoder network
    #model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size= 3, activation='relu', padding='same'))
    #model.add(tf.keras.layers.UpSampling2D(5))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 64,kernel_size= 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 3,kernel_size= 1, activation='sigmoid', padding='same'))  # output layer

    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    callback = [
        tf.keras.callbacks.ModelCheckpoint('CAE100 FINAL BUN.h5', monitor='val_loss',
                                           save_best_only=True,
                                           verbose=1)]


################# NUME MODEL : CAE100 FINAL BUN.h5 ###################
if __name__ == "__main__":
    # gen = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\Small_Dataset_256",
    #                        stride= (256,256), patch_size= (256,256), train_test_split_percentage =0.8, train = 'train')
    # gen.load_data()

    train_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Small_Dataset_256",
                                stride=(256, 256), patch_size=(256, 256), train_test_split_percentage=0.8,
                                train='train')

    val_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Small_Dataset_256",
                              stride=(256, 256), patch_size=(256, 256), train_test_split_percentage=0.8,
                              train='validation')

    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
                                                                                               dtype=tf.float32),
                                                                                 tf.TensorSpec(shape=(256, 256, 3),
                                                                                               dtype=tf.float32)))

    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(256, 256, 3),
                                                                                           dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(256, 256, 3),
                                                                                           dtype=tf.float32)))

    #train_ds = train_ds.shuffle(buffer_size=200)
    #val_ds = val_ds.shuffle(buffer_size=200)
    train_ds = train_ds.batch(batch_size=2)
    val_ds = val_ds.batch(batch_size=2)

    model = tf.keras.models.Sequential()
    # encoder network
    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size= 2, activation='relu', padding='valid', input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters = 64,kernel_size= 2, strides = (8,8), activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu', strides = (4,4), padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
    # decoder network
    model.add(tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size= 2, strides = (4,4), activation='relu', padding='valid'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 64,kernel_size= 2, strides = (8,8), activation='relu', padding='valid'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=1, activation='relu', padding='valid'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 3,kernel_size= 1, activation='sigmoid', padding='valid'))  # output layer
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    callback = [
        tf.keras.callbacks.ModelCheckpoint('MODEL_CAE v3 256 x 256 - Small Dataset 10 epochs.h5', monitor='val_loss',
                                           save_best_only=True,
                                           verbose=1)]

    #model_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callback)


'''



'''
if __name__ == "__main__":
    # gen = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\Full_Dataset_100",
    #                        stride= (100,100), patch_size= (100,100), train_test_split_percentage =0.8, train = 'train')
    # gen.load_data()

    train_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patch de 100 x 100",
                                stride=(100, 100), patch_size=(100, 100), train_test_split_percentage=0.8,
                                train='train')

    val_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patch de 100 x 100",
                              stride=(100, 100), patch_size=(100, 100), train_test_split_percentage=0.8,
                              train='validation')

    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(100, 100, 3),
                                                                                               dtype=tf.float32),
                                                                                 tf.TensorSpec(shape=(100, 100, 3),
                                                                                               dtype=tf.float32)))

    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(100, 100, 3),
                                                                                           dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(100, 100, 3),
                                                                                           dtype=tf.float32)))

    #train_ds = train_ds.shuffle(buffer_size=200)
    #val_ds = val_ds.shuffle(buffer_size=200)
    train_ds = train_ds.batch(batch_size=16)
    val_ds = val_ds.batch(batch_size=16)

    model = tf.keras.models.Sequential()
    # encoder network
    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size= 7, strides=(5,5), activation='relu', padding='valid', input_shape=(100, 100, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters = 256,kernel_size= 5, strides=(3,3), activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation='relu', padding='valid'))
    #model.add(tf.keras.layers.MaxPooling2D(2, padding='valid'))
    # decoder network
    model.add(tf.keras.layers.Conv2DTranspose(filters = 512, kernel_size= 1, activation='relu', padding='valid'))
    #model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 256,kernel_size= 5, strides=(3,3), activation='relu', padding='valid'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(5,5), activation='relu', padding='valid'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 3,kernel_size= 1, activation='sigmoid', padding='valid'))  # output layer
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    callback = [
        tf.keras.callbacks.ModelCheckpoint('CAE MODEL STRIDE v1 100 x 100 - Small Dataset 10 epochs.h5', monitor='val_loss',
                                           save_best_only=True,
                                           verbose=1)]

    #model_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callback)
'''


################################ DnCNN ####################################
'''
if __name__ == "__main__":
    gen = Generator (load_path= "C:\\Users\\vlogw\\Desktop\\Data\\", save_path= "D:\\Full_Dataset_100",
                           stride= (100,100), patch_size= (100,100), train_test_split_percentage =0.8, train = 'train')
    gen.load_data()

    train_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patch de 100 x 100",
                                stride=(100, 100), patch_size=(100, 100), train_test_split_percentage=0.8,
                                train='train')

    val_generator = Generator(load_path="C:\\Users\\vlogw\\Desktop\\Data\\", save_path="D:\\Patch de 100 x 100",
                              stride=(100, 100), patch_size=(100, 100), train_test_split_percentage=0.8,
                              train='validation')

    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=(tf.TensorSpec(shape=(100, 100, 3),
                                                                                               dtype=tf.float32),
                                                                                 tf.TensorSpec(shape=(100, 100, 3),
                                                                                               dtype=tf.float32)))

    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=(tf.TensorSpec(shape=(100, 100, 3),
                                                                                           dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(100, 100, 3),
                                                                                           dtype=tf.float32)))

    #train_ds = train_ds.shuffle(buffer_size=200)
    #val_ds = val_ds.shuffle(buffer_size=200)
    train_ds = train_ds.batch(batch_size=16)
    val_ds = val_ds.batch(batch_size=16)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size= 3, activation='relu', padding='same', input_shape=(100, 100, 3)))
    for _ in range(6):
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same' ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters = 3, kernel_size= 3, padding = 'same'))

    print(model.summary())

    callback = [
        tf.keras.callbacks.ModelCheckpoint('CEDEZ NERVOS STRIDE v1 100 x 100 - Small Dataset 10 epochs.h5', monitor='val_loss',
                                           save_best_only=True,
                                           verbose=1)]

    #model_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callback)

'''
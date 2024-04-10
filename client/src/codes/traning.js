export const model2 = {
  fileName: "Traning.py",
  code: `
    model.compile(
      loss='categorical_crossentropy',
      metrics=['accuracy'],optimizer='adam'
    )
    
    callback = [callbacks.EarlyStopping(monitor='val_accuracy',patience=5)]
    batch_size = 40
    n_epochs = 10
    
    results = model.fit(x_train,Y_train,
                        batch_size=batch_size,epochs=n_epochs,
                        verbose=1,
                        validation_data=(x_test,Y_test),
                        callbacks=callback)
          `,
};

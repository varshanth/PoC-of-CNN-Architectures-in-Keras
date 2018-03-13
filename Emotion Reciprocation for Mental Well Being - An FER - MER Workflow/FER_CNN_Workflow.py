from FER_MER_data_set_utils import oulu_casia_ds



# Get Modified Expanded OULU CASIA Dataset
oulu_casia_dataset = oulu_casia_ds(dataset_mode = 'modified_expanded')

# Convert emotion labels to categorical
oulu_casia_dataset.labels_to_categorical()


print(oulu_casia_dataset.X_train.shape,
      oulu_casia_dataset.X_test.shape,
      oulu_casia_dataset.y_train.shape,
      oulu_casia_dataset.y_test.shape)




def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def get_jpg_string(image):
    # Gets the serialized jpg from a menpo `Image`.
    fp = BytesIO()
    mio.export_image(image, fp, extension='jpg')
    fp.seek(0)
    return fp.read()


def generate(iterator,
             store_path='./',
             record_name='inference.tfrecords',
             store_records=True):
    store_path = Path(store_path)

    if store_records:
        with tf.device("/gpu:0"):
            writer = tf.python_io.TFRecordWriter(
                str(store_path / record_name))

# for img_name, pimg in iterator:
    pimg = iterator
    img_name = iterator.path.name
# resize image to 256 * 256
    cimg = pimg.resize([256, 256])

    img_path = store_path / '{}'.format(img_name)
# print(cimg.shape)
    if store_records:
        try:
            # construct the Example proto object
            with tf.device("/gpu:0"):
                example = tf.train.Example(
                    features=tf.train.Features(
                        # Features contains a map of string to Feature proto objects
                        feature={
                            # images
                            'image': tfrecords.bytes_feature(get_jpg_string(cimg)),
                            'height': tfrecords.int_feature(cimg.shape[0]),
                            'width': tfrecords.int_feature(cimg.shape[1]),
                        }))
        # use the proto object to serialize the example to a string
                serialized = example.SerializeToString()
        # write the serialized object to disk
                writer.write(serialized)

        except Exception as e:
            print(
                'Something bad happened when processing image: "{}"'.format(img_name))
            print(e)

    if store_records:
        writer.close()


# where should the resulting TFRecords files be written to?
store_path = Path('data/images')
inference_record_name = "temp.tfrecords"
import project.hourglass.params as hgparams

params = {
    hgparams.N_FEATURES: 128,
    hgparams.N_HOURGLASS: 1,
    hgparams.N_RESIDUALS: 3,
}
# Where is the model located?
# I:/menpo/project_lv/ #/media/taopan/data/landmark/00_project-master/
model_dir = Path('models/lv/lv_1hg_lr1e-3_decay10/')
params[hgparams.N_LANDMARKS] = 34
# Instantiate Estimator
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
run_config = tf.contrib.learn.RunConfig(
    gpu_memory_fraction=0.1,
    session_config=config

)
# generate TFRecords
generate(image, store_path, inference_record_name,
         store_records=True)

# Where are the .tfrecords?
infer_data_dir = Path('data/images/')  # I:/menpo/project_lv/
infer_tfrecords = 'temp.tfrecords'

infer_data = infer_data_dir / infer_tfrecords
nn = learn.Estimator(model_dir=str(model_dir), params=params,
                     config=run_config, model_fn=estimator._model_fn)
predictions = nn.predict(input_fn=lambda: predict._input_fn(infer_data))
images_generator = visualisation.lv_predictions(predictions,
                                                show_input_images=True,
                                                show_combined_heatmap=True,
                                                show_individual_heatmaps=False)

images = menpo.base.LazyList.init_from_iterable(images_generator)


def flatten(list_of_lists): return [
    item for sublist in list_of_lists for item in sublist]


images = flatten(images)

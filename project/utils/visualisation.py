import numpy as np
from menpo.image import Image
from menpo.landmark import labeller
from menpo.shape import PointCloud
from menpo.landmark import labeller, left_ventricle_34,left_ventricle_34_trimesh,left_ventricle_34_trimesh1
import project.utils.labeller_lv as labels

def lv_predictions(predictions, show_input_images=True, show_combined_heatmap=True,
                        show_individual_heatmaps=False):
    if isinstance(predictions, dict):
        for i in range(len(predictions['images'])):
            yield _attach_predictions('lv_34',
                                      predictions['images'][i],
                                      predictions['coordinates'][i],
                                      predictions['heatmaps'][i],
                                      left_ventricle_34,
                                      show_input_images,
                                      show_combined_heatmap,
                                      show_individual_heatmaps,
                                      labels.lv_index_to_label)
    else:
        for prediction in predictions:
            yield _attach_predictions('lv_34',
                                      prediction['images'],
                                      prediction['coordinates'],
                                      prediction['heatmaps'],
                                      left_ventricle_34 ,
                                      show_input_images,
                                      show_combined_heatmap,
                                      show_individual_heatmaps,
                                      labels.lv_index_to_label)

def catface_predictions(predictions, show_input_images=True, show_combined_heatmap=True,
                        show_individual_heatmaps=False):
    if isinstance(predictions, dict):
        for i in range(len(predictions['images'])):
            yield _attach_predictions('cat',
                                      predictions['images'][i],
                                      predictions['coordinates'][i],
                                      predictions['heatmaps'][i],
                                      labels.catface_to_catface,
                                      show_input_images,
                                      show_combined_heatmap,
                                      show_individual_heatmaps,
                                      labels.catface_index_to_label)
    else:
        for prediction in predictions:
            yield _attach_predictions('cat',
                                      prediction['images'],
                                      prediction['coordinates'],
                                      prediction['heatmaps'],
                                      labels.catface_to_catface,
                                      show_input_images,
                                      show_combined_heatmap,
                                      show_individual_heatmaps,
                                      labels.catface_index_to_label)


def catpose_predictions(predictions, show_input_images=True, show_combined_heatmap=True,
                        show_individual_heatmaps=False):
    if isinstance(predictions, dict):
        for i in range(len(predictions['images'])):
            yield _attach_predictions('catbody24',
                                      predictions['images'][i],
                                      predictions['coordinates'][i],
                                      predictions['heatmaps'][i],
                                      labels.catbody24_to_catbody24,
                                      show_input_images,
                                      show_combined_heatmap,
                                      show_individual_heatmaps,
                                      labels.catpose_index_to_label)
    else:
        for prediction in predictions:
            yield _attach_predictions('catbody24',
                                      prediction['images'],
                                      prediction['coordinates'],
                                      prediction['heatmaps'],
                                      labels.catbody24_to_catbody24,
                                      show_input_images,
                                      show_combined_heatmap,
                                      show_individual_heatmaps,
                                      labels.catpose_index_to_label)


def _attach_predictions(template_name, pixels, coordinates, heatmaps, labeller_fn, show_input_images,
                        show_combined_heatmap, show_individual_heatmaps,
                        index_to_label_fn):
    input_image = Image.init_from_channels_at_back(pixels)
    input_image.landmarks['predictions'] = PointCloud(coordinates)
    labeller(input_image, 'predictions', labeller_fn)
    del input_image.landmarks['predictions']
    images = []

    if show_input_images:
        images.append(input_image)

    if show_combined_heatmap:
        combined_heatmap = np.sum(heatmaps, axis=-1) * 255.0
        combined_heatmap = Image(combined_heatmap)
        combined_heatmap.landmarks[template_name] = input_image.landmarks[template_name]
        images.append(combined_heatmap)

    if show_individual_heatmaps:
        for i in range(heatmaps.shape[-1]):
            heatmap = heatmaps[..., i] * 255.0
            heatmap = Image(heatmap)

            if index_to_label_fn is not None:
                label = index_to_label_fn(i)
                #print(label)
                heatmap.landmarks[label] = PointCloud([coordinates[i]])

            images.append(heatmap)

    return images

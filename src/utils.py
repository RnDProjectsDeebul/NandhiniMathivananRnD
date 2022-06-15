import matplotlib.pyplot as plt
import numpy as np
import config


def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    img = image[0]
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    # reshape it into the original 96×96 dimensions
    img = img.reshape(96, 96)
    plt.imshow(img, cmap='gray')

    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
        plt.text(output_keypoint[p, 0], output_keypoint[p, 1], f"{p}")
        plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'g.')
        plt.text(orig_keypoint[p, 0], orig_keypoint[p, 1], f"{p}")
    plt.savefig(f"{config.OUTPUT_PATH}/val_epoch_{epoch}.png")
    plt.close()


def test_keypoints_plot(images_list, outputs_list):

    plt.figure(figsize=(10, 10))
    for i in range(len(images_list)):
        outputs = outputs_list[i]
        image = images_list[i]
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(-1, 2)
        plt.subplot(3, 3, i+1)
        plt.imshow(image, cmap='gray')
        for p in range(outputs.shape[0]):
                plt.plot(outputs[p, 0], outputs[p, 1], 'r.')
                plt.text(outputs[p, 0], outputs[p, 1], f"{p}")
        plt.axis('off')
    plt.savefig(f"{config.OUTPUT_PATH}/test_output.png")
    plt.show()
    plt.close()


def dataset_keypoints_plot(data):
    plt.figure(figsize=(20, 40))
    for i in range(30):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))
        img = img.reshape(96, 96)
        plt.subplot(5, 6, i + 1)
        plt.imshow(img, cmap='gray')
        keypoints = sample['keypoints']
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], 'r.')
    plt.show()
    plt.close()

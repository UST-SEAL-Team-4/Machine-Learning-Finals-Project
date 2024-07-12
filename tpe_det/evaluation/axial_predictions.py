import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def make_predictions_axial(net_axial, images, score_threshold=0.07):
    device = torch.device('cuda')
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        det = net_axial(images)
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:, :4]
            scores = det[i].detach().cpu().numpy()[:, 4]
            indexes = np.where(scores > score_threshold)[0]
            # print(indexes)
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    torch.cuda.empty_cache()
    return predictions

def prediction_list(net_axial, dataset):
    prediction_list = []
    for j, (images_axial, _, image_ids_axial, _) in enumerate(dataset):
        predictions = make_predictions_axial(net_axial, images_axial)
        prediction_list.append({"predictions": predictions, "id": image_ids_axial})
        print(f'Batch {j} prediction done')
        
    return prediction_list


def predicted_boxes(pred_list, case_num=0):
    predicted_boxes = []
    temp = pred_list[case_num]['predictions']
    for i in range(len(temp)):
        predicted_boxes.append(temp[i]['boxes'])
        
    return predicted_boxes

def plot_predictions(dataset, pred_list, case_num=0):
    slices, targets, _, _ = dataset[case_num]
    num_slices = len(slices)
    num_cols = 5
    num_rows = (num_slices + num_cols - 1) // num_cols

    _, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    print(targets[8]['boxes'])
    
    pred_boxes = predicted_boxes(pred_list)
    
    for idx, (slice_base, target) in enumerate(zip(slices, targets)):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        heatmap_data = torch.mean(slice_base.float(), dim=0)
        heatmap_data_np = heatmap_data.numpy()
        sns.heatmap(heatmap_data_np, ax=ax)

        print(idx)
        boxes = pred_boxes[idx]
        for box in boxes:
            box = torch.from_numpy(box)
            x_min, y_min, x_max, y_max = box
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor='g', facecolor='none'))

        boxes = target['boxes']
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor='b', facecolor='none'))
    plt.tight_layout()
    plt.show()
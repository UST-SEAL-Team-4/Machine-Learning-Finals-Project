import torch
import gc
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
from dataset.valdo import VALDODataset

def load_net(checkpoint_path):
    device = torch.device('cuda')
    # Not sure if gagamitin dapat yung d7
    config = get_efficientdet_config('tf_efficientdet_d7')

    config.update({'num_classes': 1})
    config.update({'image_size': (256, 256)})
    config.update({"norm_kwargs": dict(eps=.001, momentum=.01)})

    net = EfficientDet(config, pretrained_backbone=False)
    # Configures the classification head of the model
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchPredict(net)
    net.eval()
    device = torch.device(device)
    return net.to(device)

def make_predictions_axial(images, net_axial, score_threshold=0.07):
    device = torch.device('cuda')
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        # det = net_axial(images, torch.tensor([1]*images.shape[0]).float().to(device))
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

def get_test_dataset(df, transform):
    test_dataset = VALDODataset(
        img_paths=df['MRI Scans'].tolist(),
        ann_paths=df['Segmented Masks'].tolist(),
        transform=transform
    )

    return test_dataset




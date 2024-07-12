import torch
import gc
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

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

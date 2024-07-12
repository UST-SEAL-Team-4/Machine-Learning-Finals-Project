import pandas as pd
from torch.utils.data import DataLoader
from dataset import Dataset, VALDODataset
from preprocessing import get_transform
from utils import load_net, collate_fn, get_both_markings
from testing import get_test_dataset
from training import split_train_val_datasets, run_training
from evaluation import prediction_list, predicted_boxes, plot_predictions, count_metrics, performance_metric

# test if working
all_labels, all_ids = Dataset().get_all_labels_and_ids()
transform = get_transform(height=256, width=256, p=1.0)

dataset = VALDODataset(img_paths=all_ids, 
                       ann_paths=all_labels, 
                       transform=transform)
print(f'\n\ndataset loaded\n\n')

# create dataframe
has_cmb = [1 if count > 0 else 0 for count in dataset.cmb_counts]
df_dataset = pd.DataFrame({
    'MRI Scans': dataset.img_paths,
    'Segmented Masks': dataset.ann_paths,
    'CMB Count': dataset.cmb_counts,
    'Has CMB': has_cmb
})
print(f'\n\ndataset turned into dataframe\n\n')

# training and validation split
train_dataset, val_dataset = split_train_val_datasets(df=df_dataset,
                                                      transform=transform)
print(f'\n\ntraining and validation split done\n\n')

# training
best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val, history = run_training(train_dataset, val_dataset)
print(f'\n\ntraining done\n\n')

# load the checkpoint to use in the prediction
net_axial = load_net('Model_Save(Axial)_D7/best-checkpoint-008epoch.bin')
print(f'net axial loaded')

# testing
dataloader= DataLoader(
    dataset,
    batch_size=1,
    drop_last=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f'\n\ndataloader done\n\n')

# see the progress of prediction per batch
pred_list = prediction_list(net_axial, dataset)
pred_boxes = predicted_boxes(pred_list, 0)
print(f'\n\npredlist and pred boxes done\n\n')

# plot prediction
plot_predictions(dataset, pred_list, 0)
print(f'\n\npredictions plotted\n\n')

# set df for testing
df_cmb = df_dataset[df_dataset['Has CMB'] == 1]
df_no_cmb = df_dataset[df_dataset['Has CMB'] == 0]

dataset_cmb = get_test_dataset(df_cmb, transform)
dataset_no_cmb = get_test_dataset(df_no_cmb, transform)
print(f'\n\ntest dataset done\n\n')

# get markings
all_marking_cmb, predicted_marking_cmb = get_both_markings(dataset, pred_list)
all_marking_no_cmb, predicted_marking_no_cmb = get_both_markings(dataset, pred_list)
print(f'\n\nget markings done\n\n')

# get metric results
results_cmb = count_metrics(all_marking_cmb, predicted_marking_cmb)
results_no_cmb = count_metrics(all_marking_no_cmb, predicted_marking_no_cmb)

cmb_precision, cmb_recall, cmb_fp_ave = performance_metric(results_cmb, len(dataset_cmb))

print(f'\n\nmetrics result done\n\n')
print('With CMB Precision:', cmb_precision)
print('With CMB Recall:', cmb_recall)
print('With CMB FP Average:', cmb_fp_ave)

_, _, cmb_no_fp_ave = performance_metric(results_cmb, len(dataset_cmb))
print('Without CMB FP Average:', cmb_no_fp_ave)
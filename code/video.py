import h5py
from overrides import overrides
import torch
import torch.nn
import torch.utils.data
import torchvision
from tqdm import tqdm

from dataset import SarcasmDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def save_resnet_features():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = SarcasmDataset(transform=transforms)

    resnet = pretrained_resnet152().to(DEVICE)

    class Identity(torch.nn.Module):
        @overrides
        def forward(self, input_):
            return input_

    resnet.fc = Identity()  # Trick to avoid computing the fc1000 layer, as we don't need it here.

    with h5py.File(SarcasmDataset.features_file_path('resnet', 'pool5'), 'w') as pool5_features_file:
    # h5py.File(SarcasmDataset.features_file_path('resnet', 'res5c'), 'w') as res5c_features_file:

        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            # res5c_features_file.create_dataset(video_id, shape=(video_frame_count, 2048, 7, 7))
            pool5_features_file.create_dataset(video_id, shape=(video_frame_count, 2048))

        res5c_output = None

        def avg_pool_hook(_module, input_, _output):
            nonlocal res5c_output
            res5c_output = input_[0]

        resnet.avgpool.register_forward_hook(avg_pool_hook)

        total_frame_count = sum(dataset.frame_count_by_video_id[video_id] for video_id in dataset.video_ids)
        with tqdm(total=total_frame_count, desc="Extracting ResNet features") as progress_bar:
            for instance in torch.utils.data.DataLoader(dataset):
                video_id = instance['id'][0]
                frames = instance['frames'][0].to(DEVICE)

                batch_size = 1024
                for start_index in range(0, len(frames), batch_size):
                    end_index = min(start_index + batch_size, len(frames))
                    frame_ids_range = range(start_index, end_index)
                    frame_batch = frames[frame_ids_range]

                    avg_pool_value = resnet(frame_batch)

                    # res5c_features_file[video_id][frame_ids_range] = res5c_output.cpu()
                    pool5_features_file[video_id][frame_ids_range] = avg_pool_value.cpu()

                    progress_bar.update(len(frame_ids_range))



if __name__ == '__main__':
    save_resnet_features()
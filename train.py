import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocessing import FacialKeypointsDataset, Rescale, Normalize, ToTensor
from model import FacialKeypointsDetection
from tqdm import tqdm
from torch import nn
from torch import optim


def load_dataset(batch_size=16):
    transform = transforms.Compose([
        Rescale((224, 224)),
        Normalize(),
        ToTensor()
    ])

    train_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                     root_dir='data/training',
                                     transform=transform)

    valid_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                             root_dir='data/test/',
                                             transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, valid_dataloader


def train(epoch=20, batch_size=16):

    train_dataloader, test_dataloader = load_dataset(batch_size=batch_size)

    net = FacialKeypointsDetection()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00001)

    if torch.cuda.is_available():
         device = torch.device("cuda")
    elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Using MPS")
    else:
         device = torch.device("cpu")


    net.to(device)

    for epoch in range(epoch):
        running_loss = 0.0
        net.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, train_data in pbar:
            images, keypoints = train_data['image'], train_data['keypoints']
            keypoints = keypoints.view(keypoints.size(0), -1)

            keypoints = keypoints.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            keypoints = keypoints.to(device)
            images = images.to(device)

            output_keypoints = net(images)
            loss = criterion(output_keypoints, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/(batch + 1))

        val_loss = 0.0
        net.eval()
        with torch.no_grad():
            for valid_data in test_dataloader:
                images, keypoints = valid_data['image'], valid_data['keypoints']
                keypoints = keypoints.view(keypoints.size(0), -1)

                keypoints = keypoints.type(torch.FloatTensor).to(device)
                images = images.type(torch.FloatTensor).to(device)

                output_keypoints = net(images)
                loss = criterion(output_keypoints, keypoints)

                val_loss += loss.item()

        print('Epoch: {}, Val. Loss: {}'.format(epoch + 1, val_loss/len(test_dataloader)))


    torch.save(net.state_dict(), f"{epoch}.pth")


if __name__ == "__main__":
    train()

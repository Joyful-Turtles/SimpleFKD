import argparse
import torch
import cv2
from matplotlib import pyplot as plt
from model import FacialKeypointsDetection


def run(input_path, model_path):
    model = FacialKeypointsDetection()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.is_available():
         device = torch.device("cuda")
    elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Using MPS")
    else:
         device = torch.device("cpu")

    model.to(device)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224)) / 255.0
    image = image.reshape(1, 1, 224, 224)

    tensor_image = torch.from_numpy(image).float()
    tensor_image = tensor_image.to("mps")
    with torch.no_grad():
        output_pts = model(tensor_image).cpu().detach().numpy()

    output_pts = output_pts * 50.0 + 100
    output_pts = output_pts.reshape(68, 2)

    tensor_image = tensor_image.cpu().numpy().squeeze()
    plt.scatter(output_pts[:, 0], output_pts[:, 1], s=20, marker='.', c='m')
    plt.imshow(tensor_image, cmap='gray')
    plt.savefig("result.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/test/James_Carville_00.jpg")
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    run(args.input, args.model)

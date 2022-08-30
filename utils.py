import torch
from torchvision import transforms

class Predict():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])

        self.model = torch.load("model.pt",map_location=torch.device('cpu'))
        self.class_dict = {0: 'COVID', 1: 'Normal', 2: 'Pneumonia'}

    def predict(self, image):
        test_image_tensor = self.transform(image)
        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(test_image_tensor)
            ps = torch.exp(out)
            topk, topclass = ps.topk(1, dim=1)
        
        return self.class_dict[topclass.cpu().numpy()[0][0]], topk.numpy()[0][0]
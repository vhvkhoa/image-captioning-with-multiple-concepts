from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from .utils import load_json

class CocoImageDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_paths = []
        for image_root, dirs, image_names in os.walk(self.root):
            image_root = os.path.relpath(image_root, self.root)
            for image_name in image_names:
                if image_name[-4:].lower() in ['.jpg', '.png']:
                    self.image_paths.append(os.path.join(image_root, image_name))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([transforms.Resize((224,224)),
                                                    transforms.ToTensor(), normalize])
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transformation is not None:
            image = self.transformation(image)
        return path, image

    def __len__(self, ):
        return len(self.image_paths)


class CocoCaptionDataset(Dataset):
    def __init__(self, caption_file, concept_file, action_file, scene_dir, split='train', use_id=True):
        self.split = split
        self.use_id = use_id
        dataset = load_json(caption_file)

        if split == 'train':
            self.dataset = dataset['annotations']
            self.word_to_idx = load_json('data/word_to_idx.json')
        else:
            self.dataset = dataset['images']

        self.concepts = load_json(concept_file)
        self.actions = load_json(action_file)
        self.scene_dir = scene_dir

    def __getitem__(self, index):
        item = self.dataset[index]
        image_id = os.path.splitext(item['file_name'])[0]
        feature_path = os.path.join('data', self.split, 'feats', image_id + '.npy')
        feature = np.load(feature_path)
        concept = np.array(self.concepts[image_id])
        action = np.array(self.actions[image_id])
        scene_feature = np.load(os.path.join(self.scene_dir, self.split, image_id + '.npy'))

        if self.split == 'train':
            caption = item['caption']
            cap_vec = item['vector']
            return feature, concept, action, scene_feature, cap_vec, caption
        
        if self.use_id:
            idx = item['id']
        else:
            idx = item['file_name']
        return feature, concept, action, scene_feature, idx

    def __len__(self, ):
        return len(self.dataset)

    def get_vocab_dict(self):
        return self.word_to_idx

from torchvision.datasets import VisionDataset
from PIL import Image
import os.path
import torch
try:
    import torch_npu
except ImportError:
    pass
import json
import numpy as np


class FEMNIST(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, selected_client_ids=None):
        super(FEMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.selected_client_ids = (
            list(selected_client_ids) if selected_client_ids is not None else None
        )

        self.train_data_dir = os.path.join(root, 'train')
        self.test_data_dir = os.path.join(root, 'test')

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data_dir = self.train_data_dir
        else:
            data_dir = self.test_data_dir

        # ------------------------------------------------------------------
        # Fast path: load from .pt binary (100x faster than JSON parsing)
        # ------------------------------------------------------------------
        pt_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.pt'))
        json_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))

        if pt_files:
            self._load_from_pt(data_dir, pt_files)
        elif json_files:
            selected = set(self.selected_client_ids) if self.selected_client_ids is not None else None
            self._load_from_json(data_dir, json_files, selected)
        else:
            raise RuntimeError(f'No .pt or .json data files found in {data_dir}')

        print('')

    def _load_from_pt(self, data_dir, pt_files):
        """Load dataset from pre-converted .pt binary files."""
        selected = set(self.selected_client_ids) if self.selected_client_ids is not None else None

        all_images = []
        all_targets = []
        all_clients = []
        client_id_list = []
        client_offset = 0

        for f in pt_files:
            file_path = os.path.join(data_dir, f)
            d = torch.load(file_path, weights_only=True)

            file_client_ids = d["client_ids"]  # list[str]
            file_num_samples = d["num_samples"]  # list[int]
            file_images = d["images"]            # (N, 784) float32
            file_targets = d["targets"]          # (N,) int64
            file_clients = d["clients"]          # (N,) int64

            if selected is None:
                # Take all clients — remap indices to global space
                mask = torch.ones(len(file_images), dtype=torch.bool)
                client_id_list.extend(file_client_ids)
                index_shift = file_clients + client_offset
            else:
                # Filter by selected_client_ids
                mask = torch.zeros(len(file_images), dtype=torch.bool)
                keep_indices = []
                new_client_idx = len(client_id_list)
                for i, cid in enumerate(file_client_ids):
                    if cid in selected:
                        client_mask = (file_clients == i)
                        mask |= client_mask
                        client_id_list.append(cid)
                        keep_indices.append((i, new_client_idx))
                        new_client_idx += 1
                # Remap client indices
                index_shift = file_clients.clone()
                for old_idx, new_idx in keep_indices:
                    index_shift[file_clients == old_idx] = new_idx

            all_images.append(file_images[mask])
            all_targets.append(file_targets[mask])
            all_clients.append(index_shift[mask])
            client_offset += len(client_id_list)

        self.images = torch.cat(all_images, dim=0)   # (N, 784) float32
        self.targets = torch.cat(all_targets, dim=0)  # (N,) int64
        self.clients = torch.cat(all_clients, dim=0)  # (N,) int64
        self.client_ids = client_id_list

    def _load_from_json(self, data_dir, json_files, selected):
        """Original JSON loading path (slow, kept for backward compatibility)."""
        data = {}
        for f in json_files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            user_data = cdata['user_data']
            if selected is None:
                data.update(user_data)
            else:
                for user_id in selected:
                    if user_id in user_data:
                        data[user_id] = user_data[user_id]

        if self.selected_client_ids is None:
            list_keys = sorted(data.keys())
        else:
            list_keys = [user_id for user_id in self.selected_client_ids if user_id in data]
        self.client_ids = list(list_keys)
        self.images = []
        self.targets = []
        self.clients = []

        for i in range(0, len(list_keys)):
            self.images += data[list_keys[i]]["x"]
            self.targets += data[list_keys[i]]["y"]
            for j in range(0, len(data[list_keys[i]]["x"])):
                self.clients.append(i)

    def __getitem__(self, index):
        img, target = self.images[index], int(self.targets[index])
        # Support both tensor (.pt path) and list (JSON path) storage
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.array(img, dtype=np.float32).reshape(28, 28)
        img = Image.fromarray(img, mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_dict_clients(self):
        dict_clients = {}
        clients = self.clients
        # Support both tensor (.pt path) and list (JSON path)
        if isinstance(clients, torch.Tensor):
            clients = clients.tolist()
        for i in range(0, len(clients)):
            cid = clients[i]
            if cid not in dict_clients:
                dict_clients[cid] = []
            dict_clients[cid].append(i)

        for i in dict_clients.keys():
            dict_clients[i] = set(dict_clients[i])

        return dict_clients

    def download(self):
        raise Exception('Download currently not supported')

    def _check_exists(self):
        return os.path.exists(self.train_data_dir) and os.path.exists(self.test_data_dir)

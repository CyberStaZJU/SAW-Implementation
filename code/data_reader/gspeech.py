import hashlib
import json
import math
import os
import time
import wave

import numpy as np
import torch
from torch.utils.data import Dataset


def _json_files(root, split):
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return []
    return [
        os.path.join(split_dir, name)
        for name in sorted(os.listdir(split_dir))
        if name.endswith('.json')
    ]


def _load_split(root, split, selected_client_ids=None):
    selected = set(selected_client_ids) if selected_client_ids is not None else None
    rows = []
    for file_path in _json_files(root, split):
        with open(file_path, 'r') as handle:
            payload = json.load(handle)
        for client_id, user_payload in payload.get('user_data', {}).items():
            if selected is not None and client_id not in selected:
                continue
            xs = user_payload.get('x', [])
            ys = user_payload.get('y', [])
            for x_value, y_value in zip(xs, ys):
                rows.append((client_id, x_value, y_value))
    return rows


def _build_label_map(rows):
    labels = sorted({row[2] for row in rows}, key=lambda value: str(value))
    if all(isinstance(label, (int, np.integer)) for label in labels):
        int_labels = sorted(int(label) for label in labels)
        if int_labels == list(range(len(int_labels))):
            return {label: int(label) for label in labels}
    return {label: idx for idx, label in enumerate(labels)}


def _resolve_audio_path(root, value):
    if not isinstance(value, str):
        return None
    path = os.path.expanduser(os.path.expandvars(value))
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(root, path),
        os.path.join(root, 'raw', path),
        os.path.join(root, 'audio', path),
        os.path.join(root, 'clips', path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _load_wav(path, target_sample_rate):
    with wave.open(path, 'rb') as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Only 16-bit PCM wav is supported: {path}")

    waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_sample_rate:
        source_x = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False)
        target_len = int(round(waveform.shape[0] * float(target_sample_rate) / float(sample_rate)))
        target_x = np.linspace(0.0, 1.0, num=max(1, target_len), endpoint=False)
        waveform = np.interp(target_x, source_x, waveform).astype(np.float32)

    return torch.from_numpy(waveform)


def _hz_to_mel(freq):
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate, n_fft, n_mels):
    freq_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sample_rate / 2.0)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([_mel_to_hz(float(value)) for value in mel_points])
    bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    bin_points = torch.clamp(bin_points, 0, freq_bins - 1)

    filters = torch.zeros(n_mels, freq_bins)
    for mel_idx in range(n_mels):
        left = int(bin_points[mel_idx])
        center = int(bin_points[mel_idx + 1])
        right = int(bin_points[mel_idx + 2])
        if center > left:
            filters[mel_idx, left:center] = torch.arange(left, center).float().sub(left).div(center - left)
        if right > center:
            filters[mel_idx, center:right] = torch.arange(center, right).float().sub(right).div(center - right)
    return filters


class FedScaleGSpeech(Dataset):
    def __init__(
        self,
        root,
        train=True,
        sample_rate=16000,
        clip_seconds=1.0,
        n_mels=40,
        n_fft=512,
        win_length=400,
        hop_length=160,
        label_map=None,
        use_feature_cache=True,
        cache_dir=None,
        selected_client_ids=None,
    ):
        self.root = os.path.abspath(root)
        self.train = train
        self.selected_client_ids = (
            list(selected_client_ids) if selected_client_ids is not None else None
        )
        self.sample_rate = int(sample_rate)
        self.target_len = int(round(float(clip_seconds) * self.sample_rate))
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.mel_filters = _mel_filterbank(self.sample_rate, self.n_fft, self.n_mels)
        self.window = torch.hann_window(self.win_length)
        self.cache_dir = cache_dir or os.path.join(self.root, 'feature_cache')
        if cache_dir is None and self.selected_client_ids is not None:
            digest = hashlib.sha256()
            for client_id in self.selected_client_ids:
                digest.update(str(client_id).encode('utf-8'))
                digest.update(b'\n')
            self.cache_dir = os.path.join(self.cache_dir, f"clients_{digest.hexdigest()[:12]}")
        self.use_feature_cache = (
            bool(use_feature_cache)
            and os.environ.get('GSPEECH_FEATURE_CACHE', '1') != '0'
        )
        self.features = None

        if not (_json_files(self.root, 'train') and _json_files(self.root, 'test')):
            raise RuntimeError(
                "GSpeech dataset not found. Expected FedScale/LEAF-style JSON files "
                f"under {os.path.join(self.root, 'train')} and {os.path.join(self.root, 'test')}, "
                "with user_data[client_id] = {'x': [wav paths], 'y': [labels]}."
            )

        all_rows = _load_split(self.root, 'train') + _load_split(self.root, 'test')
        self.label_map = label_map or _build_label_map(all_rows)
        split_rows = _load_split(
            self.root,
            'train' if train else 'test',
            selected_client_ids=self.selected_client_ids,
        )

        self.examples = []
        self.clients = []
        client_to_index = {}
        for client_id, x_value, y_value in split_rows:
            if client_id not in client_to_index:
                client_to_index[client_id] = len(client_to_index)
            self.examples.append((x_value, self.label_map[y_value]))
            self.clients.append(client_to_index[client_id])

        if self.use_feature_cache:
            self.features = self._load_or_build_feature_cache()

    @property
    def num_classes(self):
        return len(self.label_map)

    @property
    def split_name(self):
        return 'train' if self.train else 'test'

    def _cache_config(self):
        return {
            'split': self.split_name,
            'sample_rate': self.sample_rate,
            'target_len': self.target_len,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
        }

    def _examples_signature(self):
        digest = hashlib.sha256()
        digest.update(json.dumps(self._cache_config(), sort_keys=True).encode('utf-8'))
        for x_value, label in self.examples:
            digest.update(repr(x_value).encode('utf-8'))
            digest.update(b'\0')
            digest.update(str(int(label)).encode('utf-8'))
            digest.update(b'\n')
        return digest.hexdigest()

    def _cache_path(self):
        name = (
            f"{self.split_name}"
            f"_sr{self.sample_rate}"
            f"_len{self.target_len}"
            f"_mel{self.n_mels}"
            f"_fft{self.n_fft}"
            f"_win{self.win_length}"
            f"_hop{self.hop_length}.pt"
        )
        return os.path.join(self.cache_dir, name)

    def _load_feature_cache(self, cache_path, signature):
        if not os.path.exists(cache_path):
            return None
        try:
            payload = torch.load(cache_path, map_location='cpu')
        except Exception as exc:
            print(f"Ignoring unreadable GSpeech feature cache {cache_path}: {exc}", flush=True)
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get('signature') != signature:
            return None
        features = payload.get('features')
        if not torch.is_tensor(features) or int(features.shape[0]) != len(self.examples):
            return None
        print(f"Loaded GSpeech feature cache: {cache_path}", flush=True)
        return features

    def _build_feature_cache(self, cache_path, signature):
        print(f"Building GSpeech feature cache: {cache_path}", flush=True)
        features = []
        for idx, (x_value, _) in enumerate(self.examples, start=1):
            waveform = self._waveform_from_value(x_value)
            features.append(self._log_mel(waveform))
            if idx % 5000 == 0:
                print(f"  cached {idx} / {len(self.examples)} GSpeech samples", flush=True)

        feature_tensor = torch.stack(features, dim=0).contiguous()
        payload = {
            'signature': signature,
            'config': self._cache_config(),
            'features': feature_tensor,
        }
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
        print(f"Saved GSpeech feature cache: {cache_path}", flush=True)
        return feature_tensor

    def _load_or_build_feature_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._cache_path()
        signature = self._examples_signature()
        cached = self._load_feature_cache(cache_path, signature)
        if cached is not None:
            return cached

        lock_dir = f"{cache_path}.lock"
        while True:
            try:
                os.mkdir(lock_dir)
                break
            except FileExistsError:
                if time.time() - os.path.getmtime(lock_dir) > 6 * 60 * 60:
                    os.rmdir(lock_dir)
                    continue
                print(f"Waiting for GSpeech feature cache: {cache_path}", flush=True)
                time.sleep(10)
                cached = self._load_feature_cache(cache_path, signature)
                if cached is not None:
                    return cached

        try:
            cached = self._load_feature_cache(cache_path, signature)
            if cached is not None:
                return cached
            return self._build_feature_cache(cache_path, signature)
        finally:
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass

    def _waveform_from_value(self, value):
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)
        path = _resolve_audio_path(self.root, value)
        if path is None or not os.path.exists(path):
            raise RuntimeError(f"GSpeech audio file not found: {value}")
        return _load_wav(path, self.sample_rate)

    def _fix_length(self, waveform):
        waveform = waveform.float().flatten()
        if waveform.numel() > self.target_len:
            return waveform[:self.target_len]
        if waveform.numel() < self.target_len:
            return torch.nn.functional.pad(waveform, (0, self.target_len - waveform.numel()))
        return waveform

    def _log_mel(self, waveform):
        waveform = self._fix_length(waveform)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = self.mel_filters.matmul(power)
        return torch.log1p(mel).unsqueeze(0)

    def __getitem__(self, index):
        x_value, label = self.examples[index]
        if self.features is not None:
            return self.features[index], int(label)
        waveform = self._waveform_from_value(x_value)
        return self._log_mel(waveform), int(label)

    def __len__(self):
        return len(self.examples)

    def get_dict_clients(self):
        dict_clients = {}
        for idx, client_id in enumerate(self.clients):
            dict_clients.setdefault(client_id, set()).add(idx)
        return dict_clients

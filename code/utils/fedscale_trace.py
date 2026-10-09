import csv
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


PROFILE_CONTAINER_KEYS = (
    'client_profiles',
    'client_profile',
    'client_device_capacity',
    'device_capacity',
    'profiles',
    'clients',
)

AVAILABILITY_CONTAINER_KEYS = (
    'client_availability',
    'availability',
    'availabilities',
    'device_availability',
    'traces',
    'trace',
)

CLIENT_ID_KEYS = (
    'client_id',
    'clientId',
    'client',
    'id',
    'user_id',
    'user',
)

START_KEYS = ('start', 'start_time', 'begin', 'begin_time', 'online_time')
END_KEYS = ('end', 'end_time', 'finish', 'finish_time', 'offline_time')
FEDSCALE_ACTIVE_KEYS = ('active',)
FEDSCALE_INACTIVE_KEYS = ('inactive',)
FEDSCALE_FINISH_TIME_KEYS = ('finish_time',)
DURATION_KEYS = (
    'duration',
    'completion_time',
    'execution_time',
    'exec_time',
    'train_time',
    'round_time',
)
STATUS_KEYS = ('status', 'available', 'online', 'active', 'is_available')

COMPUTE_TIME_KEYS = (
    'computation',
    'compute',
    'compute_time',
    'computation_time',
    'training_time',
)
COMM_TIME_KEYS = (
    'communication',
    'comm',
    'communication_time',
    'network_time',
    'upload_time',
)
BANDWIDTH_KEYS = (
    'bandwidth',
    'upload_bandwidth',
    'communication_speed',
    'comm_speed',
    'uplink',
)
COMPUTE_SPEED_KEYS = (
    'compute_speed',
    'computation_speed',
    'device_speed',
    'speed',
)


def _canonical_client_id(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith('.0'):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _numeric_sort_key(value):
    text = _canonical_client_id(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _to_float(value, default=None):
    if value is None or value == '':
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _first_value(record, keys, default=None):
    if not isinstance(record, dict):
        return default
    for key in keys:
        if key in record:
            return record[key]
    lower = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return default


def _load_csv(path):
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_fedscale_payload(path):
    """Load a FedScale-style profile/trace payload.

    FedScale deployments commonly serialize profiles with pickle, while
    downstream experiment scripts often export the same data to JSON/CSV.
    This loader accepts all three so the simulator can consume the official
    profile directly or a lossless converted copy.
    """
    if not path:
        return None
    path = os.path.expanduser(os.path.expandvars(str(path)))
    suffix = ''.join(Path(path).suffixes).lower()
    if suffix.endswith(('.jsonl', '.ndjson')):
        return _load_jsonl(path)
    if suffix.endswith('.json'):
        with open(path) as handle:
            return json.load(handle)
    if suffix.endswith('.csv'):
        return _load_csv(path)

    with open(path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            data = handle.read().decode('utf-8')

    if suffix.endswith(('.jsonl', '.ndjson')):
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return list(csv.DictReader(data.splitlines()))


def _unwrap_container(payload, keys):
    current = payload
    while isinstance(current, dict):
        for key in keys:
            if key in current:
                current = current[key]
                break
        else:
            return current
    return current


def _as_profile_mapping(payload):
    payload = _unwrap_container(payload, PROFILE_CONTAINER_KEYS)
    if payload is None:
        return {}

    if isinstance(payload, dict):
        rows_have_client_id = all(
            isinstance(v, dict) and any(k in v for k in CLIENT_ID_KEYS)
            for v in payload.values()
        )
        if not rows_have_client_id:
            return {
                _canonical_client_id(client_id): record
                for client_id, record in payload.items()
            }
        payload = list(payload.values())

    records = {}
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            client_id = _first_value(row, CLIENT_ID_KEYS)
            if client_id is None:
                continue
            records[_canonical_client_id(client_id)] = row
    return records


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer, float)):
        return float(value) > 0
    return str(value).strip().lower() in {'1', 'true', 'yes', 'online', 'available', 'active'}


def _parse_interval_record(record):
    if isinstance(record, dict):
        start = _to_float(_first_value(record, START_KEYS))
        end = _to_float(_first_value(record, END_KEYS))
        duration = _to_float(_first_value(record, DURATION_KEYS))
        if start is not None and end is None and duration is not None:
            end = start + max(0.0, duration)
        if start is not None and end is not None and end > start:
            return (start, end)
        return None

    if isinstance(record, (list, tuple)) and len(record) >= 2:
        start = _to_float(record[0])
        end = _to_float(record[1])
        if start is not None and end is not None and end > start:
            return (start, end)
    return None


def _parse_fedscale_behavior(record):
    if not isinstance(record, dict):
        return None

    active = _first_value(record, FEDSCALE_ACTIVE_KEYS)
    inactive = _first_value(record, FEDSCALE_INACTIVE_KEYS)
    if not isinstance(active, (list, tuple)) or not isinstance(inactive, (list, tuple)):
        return None

    intervals = []
    for start, end in zip(active, inactive):
        start = _to_float(start)
        end = _to_float(end)
        if start is not None and end is not None and end > start:
            intervals.append((start, end))

    intervals = _merge_intervals(intervals)
    if not intervals:
        return None

    finish_time = _to_float(_first_value(record, FEDSCALE_FINISH_TIME_KEYS))
    if finish_time is None:
        finish_time = max(end for _, end in intervals)

    return {
        'intervals': intervals,
        'finish_time': max(finish_time, max(end for _, end in intervals)),
    }


def _parse_interval_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        fedscale_behavior = _parse_fedscale_behavior(value)
        if fedscale_behavior is not None:
            return fedscale_behavior['intervals']
        nested = _first_value(value, AVAILABILITY_CONTAINER_KEYS)
        if nested is not None:
            return _parse_interval_list(nested)
        interval = _parse_interval_record(value)
        return [interval] if interval else []
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and not isinstance(value[0], (list, tuple, dict)):
            interval = _parse_interval_record(value)
            return [interval] if interval else []
        intervals = []
        for item in value:
            intervals.extend(_parse_interval_list(item))
        return intervals
    return []


def _merge_intervals(intervals):
    cleaned = sorted(
        (float(start), float(end))
        for start, end in intervals
        if end > start and math.isfinite(start) and math.isfinite(end)
    )
    if not cleaned:
        return []

    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _availability_from_rows(rows):
    interval_rows = defaultdict(list)
    point_rows = defaultdict(list)

    for row in rows:
        if not isinstance(row, dict):
            continue
        client_id = _first_value(row, CLIENT_ID_KEYS)
        if client_id is None:
            continue
        client_id = _canonical_client_id(client_id)

        interval = _parse_interval_record(row)
        if interval is not None:
            interval_rows[client_id].append(interval)
            continue

        time = _to_float(_first_value(row, ('time', 'timestamp', 'ts')))
        status = _first_value(row, STATUS_KEYS)
        if time is not None and status is not None:
            point_rows[client_id].append((time, _parse_bool(status)))

    for client_id, points in point_rows.items():
        points.sort(key=lambda item: item[0])
        open_start = None
        for time, active in points:
            if active and open_start is None:
                open_start = time
            elif not active and open_start is not None:
                if time > open_start:
                    interval_rows[client_id].append((open_start, time))
                open_start = None

    return {client_id: _merge_intervals(intervals) for client_id, intervals in interval_rows.items()}


def _as_availability_mapping(payload):
    payload = _unwrap_container(payload, AVAILABILITY_CONTAINER_KEYS)
    if payload is None:
        return {}
    if isinstance(payload, dict):
        availability_by_client = {}
        for client_id, value in payload.items():
            fedscale_behavior = _parse_fedscale_behavior(value)
            if fedscale_behavior is not None:
                availability_by_client[_canonical_client_id(client_id)] = fedscale_behavior
                continue

            if isinstance(value, dict):
                has_availability_shape = (
                    any(key in value for key in AVAILABILITY_CONTAINER_KEYS)
                    or any(key in value for key in FEDSCALE_ACTIVE_KEYS)
                    or any(key in value for key in FEDSCALE_INACTIVE_KEYS)
                    or any(key in value for key in START_KEYS)
                    or any(key in value for key in END_KEYS)
                )
                if not has_availability_shape:
                    continue
            elif isinstance(value, (list, tuple)):
                if not value or not isinstance(value[0], (list, tuple, dict)):
                    continue
            intervals = _merge_intervals(_parse_interval_list(value))
            if intervals:
                availability_by_client[_canonical_client_id(client_id)] = {
                    'intervals': intervals,
                    'finish_time': None,
                }
        return availability_by_client
    if isinstance(payload, list):
        return {
            client_id: {
                'intervals': intervals,
                'finish_time': None,
            }
            for client_id, intervals in _availability_from_rows(payload).items()
        }
    return {}


def _profile_duration(record, args):
    default_duration = float(getattr(args, 'fedscale_default_duration', 1.0))
    upload_size = float(getattr(args, 'fedscale_upload_size_mb', 1.0))
    download_size_arg = getattr(args, 'fedscale_download_size_mb', None)
    download_size = upload_size if download_size_arg is None else float(download_size_arg)
    batch_size = int(getattr(args, 'fedscale_batch_size', 0) or getattr(args, 'local_bs', 1))
    local_steps = int(getattr(args, 'fedscale_local_steps', 0) or getattr(args, 'local_period', 1))
    augmentation_factor = float(getattr(args, 'fedscale_augmentation_factor', 3.0))

    if isinstance(record, (int, float, np.integer, np.floating)):
        return max(0.0, float(record))

    if isinstance(record, (list, tuple)):
        values = [_to_float(value) for value in record[:2]]
        values = [value for value in values if value is not None]
        return max(0.0, sum(values)) if values else default_duration

    if not isinstance(record, dict):
        return default_duration

    explicit = _to_float(_first_value(record, DURATION_KEYS))
    if explicit is not None:
        return max(0.0, explicit)

    # FedScale's official client_device_capacity stores inference latency
    # in ms/sample under "computation" and bandwidth under "communication".
    if 'computation' in record and 'communication' in record:
        compute_speed = _to_float(record.get('computation'))
        bandwidth = _to_float(record.get('communication'))
        if compute_speed is not None and bandwidth is not None and bandwidth > 0:
            compute_time = augmentation_factor * batch_size * local_steps * compute_speed / 1000.0
            comm_time = (upload_size + download_size) / bandwidth
            return max(0.0, compute_time + comm_time)

    compute_time = _to_float(_first_value(record, COMPUTE_TIME_KEYS), 0.0)
    comm_time = _to_float(_first_value(record, COMM_TIME_KEYS), 0.0)

    compute_speed = _to_float(_first_value(record, COMPUTE_SPEED_KEYS))
    if compute_speed is not None and compute_time <= 0.0:
        compute_time = 1.0 / max(compute_speed, 1e-12)

    bandwidth = _to_float(_first_value(record, BANDWIDTH_KEYS))
    if bandwidth is not None and comm_time <= 0.0:
        comm_time = upload_size / max(bandwidth, 1e-12)

    duration = compute_time + comm_time
    if duration <= 0.0:
        duration = default_duration
    return max(0.0, duration)


class FedScaleTraceSampler:
    def __init__(self, args):
        profile_path = getattr(args, 'fedscale_client_profile_path', '')
        if not profile_path:
            raise ValueError(
                "fedscale_trace requires --fedscale_client_profile_path "
                "pointing to a FedScale client profile file"
            )

        raw_profile = load_fedscale_payload(profile_path)
        self.profile_records = _as_profile_mapping(raw_profile)
        if not self.profile_records:
            raise ValueError(f"No FedScale client profiles found in {profile_path}")

        availability = _as_availability_mapping(raw_profile)
        availability_path = getattr(args, 'fedscale_availability_trace_path', '')
        if availability_path:
            raw_availability = load_fedscale_payload(availability_path)
            availability.update(_as_availability_mapping(raw_availability))

        self.availability = availability
        self.time_scale = float(getattr(args, 'fedscale_time_scale', 1.0))
        self.min_duration = float(getattr(args, 'fedscale_min_duration', 1e-6))
        self.trace_wrap = not bool(getattr(args, 'fedscale_no_trace_wrap', False))
        self.trace_exhausted_penalty = float(
            getattr(args, 'fedscale_trace_exhausted_penalty', 3600.0)
        )

        seed = getattr(args, 'delay_seed', None)
        if seed is None:
            seed = int(getattr(args, 'seed', 0))
        self.client_ids = self._map_clients(
            num_users=int(getattr(args, 'num_users', 0)),
            sample_mode=str(getattr(args, 'fedscale_profile_sample', 'random')),
            seed=int(seed),
        )
        self.availability_client_ids = self._map_availability_clients(len(self.client_ids))

        self.base_durations = {
            client_id: max(
                self.min_duration,
                self.time_scale * _profile_duration(record, args),
            )
            for client_id, record in self.profile_records.items()
        }

    def _map_clients(self, num_users, sample_mode, seed):
        available_ids = sorted(self.profile_records.keys(), key=_numeric_sort_key)
        if num_users <= 0:
            return []

        zero_based = [_canonical_client_id(i) for i in range(num_users)]
        one_based = [_canonical_client_id(i + 1) for i in range(num_users)]
        profile_id_set = set(available_ids)
        if all(client_id in profile_id_set for client_id in zero_based):
            return zero_based
        if all(client_id in profile_id_set for client_id in one_based):
            return one_based

        if sample_mode == 'sorted':
            selected = available_ids[:num_users]
        else:
            rng = np.random.default_rng(seed + 7919)
            replace = len(available_ids) < num_users
            selected = rng.choice(available_ids, size=num_users, replace=replace).tolist()

        if len(selected) < num_users:
            repeats = int(math.ceil(num_users / max(1, len(selected))))
            selected = (selected * repeats)[:num_users]
        return [_canonical_client_id(client_id) for client_id in selected]

    def _map_availability_clients(self, num_users):
        if not self.availability:
            return [None] * num_users
        availability_keys = list(self.availability.keys())
        return [
            _canonical_client_id(availability_keys[(idx + 1) % len(availability_keys)])
            for idx in range(num_users)
        ]

    def _availability_spec(self, client_id):
        spec = self.availability.get(client_id)
        if spec is None:
            return [], None
        if isinstance(spec, dict):
            return spec.get('intervals', []), spec.get('finish_time')
        return spec, None

    def _next_online_window(self, intervals, time, finish_time=None):
        if not intervals:
            return (time, math.inf)

        if self.trace_wrap:
            if finish_time is not None and finish_time > 0:
                origin = 0.0
                period = max(self.min_duration, float(finish_time))
            else:
                origin = intervals[0][0]
                end = max(interval[1] for interval in intervals)
                period = max(self.min_duration, end - origin)
                if time < origin:
                    start, end = intervals[0]
                    return (max(time, start), end)

            cycle = math.floor((time - origin) / period) if time >= origin else -1

            for cycle_offset in range(3):
                current_cycle = cycle + cycle_offset
                base = origin + current_cycle * period
                for start, end in intervals:
                    abs_start = base + start
                    abs_end = base + end
                    if abs_end <= time:
                        continue
                    return (max(time, abs_start), abs_end)

        for start, end in intervals:
            if end <= time:
                continue
            return (max(time, start), end)

        return (time + self.trace_exhausted_penalty, time + self.trace_exhausted_penalty + self.min_duration)

    def _duration_with_availability(self, client_id, start_time, service_duration):
        intervals, finish_time = self._availability_spec(client_id)
        if not intervals:
            return service_duration

        remaining = max(self.min_duration, service_duration)
        current_time = float(start_time)
        guard = 0
        while remaining > self.min_duration and guard < 10000:
            guard += 1
            window_start, window_end = self._next_online_window(
                intervals,
                current_time,
                finish_time=finish_time,
            )
            if not math.isfinite(window_start) or not math.isfinite(window_end):
                return service_duration
            current_time = max(current_time, window_start)
            available_span = max(0.0, window_end - current_time)
            if available_span >= remaining:
                return max(self.min_duration, (current_time + remaining) - start_time)
            remaining -= available_span
            current_time = window_end + self.min_duration

        return max(self.min_duration, current_time - start_time)

    def sample_client_delay(self, idx, start_time):
        if idx is None:
            raise ValueError("fedscale_trace requires a client index")
        mapped_index = int(idx) % len(self.client_ids)
        client_id = self.client_ids[mapped_index]
        availability_client_id = self.availability_client_ids[mapped_index]
        base_duration = self.base_durations.get(client_id, self.min_duration)
        if availability_client_id is None:
            return base_duration
        return self._duration_with_availability(
            availability_client_id,
            float(start_time),
            base_duration,
        )

    def metadata(self):
        return {
            'fedscale_trace_num_profiles': len(self.profile_records),
            'fedscale_trace_num_availability_clients': len(self.availability),
            'fedscale_trace_client_ids': list(self.client_ids),
            'fedscale_trace_availability_client_ids': list(self.availability_client_ids),
            'fedscale_trace_wrap': self.trace_wrap,
            'fedscale_trace_time_scale': self.time_scale,
        }


def ensure_fedscale_trace_sampler(args, state=None):
    sampler = getattr(args, 'fedscale_trace_sampler', None)
    if sampler is None:
        sampler = FedScaleTraceSampler(args)
        args.fedscale_trace_sampler = sampler
    if state is not None:
        state['fedscale_trace_metadata'] = sampler.metadata()
    return sampler

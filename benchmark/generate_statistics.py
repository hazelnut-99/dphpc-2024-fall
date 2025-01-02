import sqlite3
import pandas as pd
import json
import os
import sys
import argparse
from get_comm_vol import get_communication_volume


def load_sqlite(db_path):
    connection = sqlite3.connect(db_path)
    query = "SELECT * FROM NVTX_EVENTS"
    return pd.read_sql_query(query, connection)


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False


def extract_statistics(db_paths):
    stats_summary = []

    for db_path in db_paths:
        df = load_sqlite(db_path)
        df = df[df['text'].apply(is_valid_json)]
        events = df['text'].tolist()
        events = [json.loads(e) for e in events]

        for event in events:
            if 'event' in event and event['event'] == '_core_forward_time_span':
                record = {
                    'metric_name': 'forward_time_ms',
                    'rank': event['rank'],
                    'measurement': event['time_elapsed_ms']
                }
                stats_summary.append(record)
            elif 'event' in event and event['event'] == 'after_core_forward':
                record = {
                    'metric_name': 'forward_peak_allocated_mib',
                    'rank': event['rank'],
                    'measurement': event['memory_info']['peak_allocated_mib']
                }
                stats_summary.append(record)
                record = {
                    'metric_name': 'forward_peak_reserved_mib',
                    'rank': event['rank'],
                    'measurement': event['memory_info']['peak_reserved_mib']
                }
                stats_summary.append(record)
            elif 'train_step' in event:
                record = {
                    'metric_name': 'tokens_per_sec',
                    'rank': event['rank'],
                    'measurement': event['train_step']['tokens_per_sec']
                }
                stats_summary.append(record)
                record = {
                    'metric_name': 'hardware_tflops',
                    'rank': event['rank'],
                    'measurement': event['train_step']['hardware_tflops']
                }
                stats_summary.append(record)

    return stats_summary


def check_out_of_memory_error(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return "OutOfMemoryError" in content


def check_return_code(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.strip() != '0':
            return False
    return True


def extract_communication_details(dir_path):
    comm_details, total_send, total_recv = get_communication_volume(dir_path)
    return [
        {
            'metric_name': 'comm_details',
            'measurement': json.dumps(comm_details)
        },
        {
            'metric_name': 'total_send_bytes',
            'measurement': total_send
        },
        {
            'metric_name': 'total_recv_bytes',
            'measurement': total_recv
        },
        {
            'metric_name': 'comm_per_link',
            'measurement': comm_details['0']['0']['send']
        }
    ]


def run(top_dir):
    if not os.path.isdir(top_dir):
        print(f"Error: The directory {top_dir} does not exist or is not a valid directory.")
        sys.exit(1)

    pairs = []
    for subdir in os.listdir(top_dir):
        subdir_path = os.path.join(top_dir, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            sqlite_paths, json_path, result_path, c, stderr_path = [], None, None, None, None
            for file_name in os.listdir(subdir_path):
                full_path = os.path.join(subdir_path, file_name)
                if file_name.endswith(".sqlite"):
                    sqlite_paths.append(full_path)
                elif file_name.endswith("parameters.json"):
                    json_path = full_path
                elif file_name.endswith("return_code"):
                    result_path = full_path
                elif file_name.endswith('stdout.o'):
                    stdout_path = full_path
                elif file_name.endswith('stderr.e'):
                    stderr_path = full_path

            pairs.append((sqlite_paths, json_path, result_path, stdout_path, stderr_path))

    results = []
    stats = []
    for db_paths, conf_path, rc_path, stdout_path, stderr_path in pairs:
        with open(conf_path, 'r') as file:
            conf = json.load(file)

        success = check_return_code(rc_path)
        result_item = dict(conf)
        result_item['success'] = success
        if not success:
            result_item['error_msg'] = 'OOM' if check_out_of_memory_error(stderr_path) else 'Unknown'
        result_item['path'] = os.path.dirname(rc_path)

        results.append(result_item)
        # if success, get details
        if result_item['success']:
            stat_items = extract_statistics(db_paths)
            if 'dp' not in conf and 'tp' not in conf and 'pp' not in conf:
                communication_items = extract_communication_details(os.path.dirname(db_paths[0]))
                stat_items.extend(communication_items)
            for stat_item in stat_items:
                stat_item.update(conf)

            stats.extend(stat_items)

    result_df = pd.DataFrame(results)
    stat_df = pd.DataFrame(stats)

    result_df.to_csv(f"{top_dir}/result.csv", index=False)
    stat_df.to_csv(f"{top_dir}/stat.csv", index=False)


parser = argparse.ArgumentParser()
parser.add_argument(
    'top_dir',
    type=str,
    help="top_dir"
)
args = parser.parse_args()
top_dir = args.top_dir
run(top_dir)
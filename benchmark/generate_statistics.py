import sqlite3
import pandas as pd
import json
import os
import sys


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


def extract_statistics(db_path):
    df = load_sqlite(db_path)
    df = df[df['text'].apply(is_valid_json)]
    events = df['text'].tolist()
    events = [json.loads(e) for e in events]

    stats_summary = []
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


def run(top_dir):
    if not os.path.isdir(top_dir):
        print(f"Error: The directory {top_dir} does not exist or is not a valid directory.")
        sys.exit(1)

    pairs = []
    for subdir in os.listdir(top_dir):
        subdir_path = os.path.join(top_dir, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Find .sqlite and .json files in the subdirectory
            sqlite_path, json_path, result_path = None, None, None
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(".sqlite"):
                    sqlite_path = file_name
                elif file_name.endswith(".json"):
                    json_path = file_name
                elif file_name.endswith("return_code"):
                    result_path = result_path
            pairs.append((sqlite_path, json_path, result_path))

    results = []
    stats = []
    for db_path, conf_path, rc_path in pairs:
        with open(conf_path, 'r') as file:
            conf = json.load(file)
        with open(rc_path, 'r') as file:
            content = file.read().strip()
            number = int(content)
            result_item = dict(conf)
            result_item['success'] = number == 0
        stat_item = extract_statistics(db_path)
        stat_item.update(conf)
        results.append(result_item)
        stats.append(stat_item)

    result_df = pd.DataFrame(results)
    stat_df = pd.DataFrame(stats)

    result_df.to_csv(f"{top_dir}/result.csv", index=False)
    stat_df.to_csv(f"{top_dir}/stat.csv", index=False)

run()
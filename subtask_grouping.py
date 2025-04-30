import json
from pathlib import Path
import argparse


def parse_time_range(time_range):
    """Convert time range string to start and end seconds."""
    start_str, end_str = time_range.split(" - ")
    
    def time_to_seconds(time_str):
        minutes, seconds = map(lambda x: int(float(x)), time_str.split(":"))
        return minutes * 60 + seconds
    
    return time_to_seconds(start_str), time_to_seconds(end_str)


def combine_time_ranges(ranges):
    """Combine multiple time ranges into a single range."""
    if not ranges:
        return None
    
    start_times, end_times = zip(*[parse_time_range(r) for r in ranges])
    min_start = min(start_times)
    max_end = max(end_times)
    min_start_minutes = min_start // 60
    min_start_seconds = min_start % 60
    max_end_minutes = max_end // 60
    max_end_seconds = max_end % 60
    return f"{min_start_minutes:02d}:{min_start_seconds:02d} - {max_end_minutes:02d}:{max_end_seconds:02d}"

def group_subtasks(motion_labels):
    """Create groups of subtasks from 0 to i for all possible i."""
    groups = []
    
    for i in range(len(motion_labels)-1):
        # Get subtasks from 0 to i
        subtasks = motion_labels[:i+1]
        
        # Extract time ranges and the final subtask label
        time_ranges = [task['time_range'] for task in subtasks]
        final_label = subtasks[-1]['sub_task']
        
        # Combine time ranges
        combined_range = combine_time_ranges(time_ranges)
        
        groups.append({
            'sub_task': final_label,
            'time_range': combined_range,
            'num_subtasks': i + 1
        })
    
    return groups


def main(args):
    input_file = args.input_dir / "dataset_motion_labels.json"
    output_file = args.input_dir / "grouped_subtasks.json"
    
    # Read input JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each trajectory
    grouped_data = {}
    for traj_id, traj_data in data.items():
        motion_labels = traj_data.get('motion_labels', [])

        if traj_id == "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_demo_3":
            print("foo")
        
        # Skip if motion_labels is empty or not a list
        if not motion_labels or not isinstance(motion_labels, list):
            continue
            
        # Create groups for this trajectory
        grouped_data[traj_id] = {
            'demo_id': traj_data['demo_id'],
            'motion_labels': group_subtasks(motion_labels),
        }
    
    # Save grouped data
    with open(output_file, 'w') as f:
        json.dump(grouped_data, f, indent=4)
    
    print(f"Grouped subtasks saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment motion labels with paraphrases.")
    parser.add_argument("--input_dir", type=Path, help="Path to the input dir of labels", required=True)
    args = parser.parse_args()
    main(args)
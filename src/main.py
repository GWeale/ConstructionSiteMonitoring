import os
from src.video_processing import read_video_frames, equirectangular_to_cartesian
from src.mapping import create_map
from src.safety_compliance import ensure_safety
from src.data_storage import save_map, save_reports

def main():
    video_path = 'data/raw/footage360.mp4'
    frames = read_video_frames(video_path)
    cartesian_frames = [equirectangular_to_cartesian(frame)[0] for frame in frames]
    map_points = create_map(frames)
    save_map(map_points, 'results/maps/map.npy')
    safety_reports = ensure_safety(frames)
    save_reports(safety_reports, 'results/reports/safety_reports.txt')

if __name__ == "__main__":
    main()

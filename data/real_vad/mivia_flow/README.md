Source: Mivia-IWDD illegal waste dumping videos, https://github.com/MiviaLab/IllegalWasteDumping (cite greco2026iwdd; bouwmans2026overview). No video is redistributed here.
Scores: Farneback optical-flow magnitude (64x64 gray, INTER_AREA) per frame, 3-frame per-video median filter; frames sampled at 10 fps by time (all frames if native fps < 10); sampled native indices in sampled_indices.json.
Labels: a sampled frame is 1 if its timestamp (k/10 s) is at or after the video's onset time in seconds, for positive videos; negative videos are all 0.
Calibration: training videos labelled "false" (sorted by video_id); test: all 100 test videos (sorted by video_id).

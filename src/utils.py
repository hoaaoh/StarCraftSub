import math
import subprocess
import re


def extract_audio(input_video, output_audio):
    command = ['ffmpeg', '-i', input_video, '-q:a', '0', '-map', 'a', output_audio]
    subprocess.run(command)


def detect_silence(audio_file):
    command = ['ffmpeg', '-i', audio_file, '-af', 'silencedetect=noise=-20dB:d=0.5', '-f', 'null', '-']
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    return result.stderr


def parse_silence_output(output):
    silence_times = []
    for line in output.split("\n"):
        if "silence_end" in line:
            match = re.search(r"silence_end: (\d+\.\d+)", line)
            if match:
                silence_times.append(float(match.group(1)))
    return silence_times


def find_closest_time(silence_times, target_time):
    closest_time = min(silence_times, key=lambda x: abs(x - target_time))
    return closest_time


def get_audio_duration(file_path):
    command = ['ffmpeg', '-i', file_path]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    duration_line = [line for line in result.stderr.split('\n') if 'Duration' in line]
    if duration_line:
        match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', duration_line[0])
        if match:
            hours, minutes, seconds = map(float, match.groups())
            duration = hours * 3600 + minutes * 60 + seconds
            return duration
    return None


def second_to_time_format(second: float, with_ms: bool = False) -> str:
    ms = divmod(int(math.floor(second * 1000)), 1000)
    second = int(math.floor(second))
    hour, second = divmod(second, 3600)
    minute, second = divmod(second, 60)
    result = f"{hour:02}:{minute:02}:{second:02}"
    if with_ms:
        result += f"{ms:03}"
    return result


def cut_audio(input_file, output_prefix, cut_points):
    formatted_cut_points = [f"{pt}:00" if len(pt.split(':')) == 2 else pt for pt in cut_points]
    start_time = "00:00:00"
    part_number = 1

    for end_time in formatted_cut_points:
        output_file = f"{output_prefix}_part{part_number}.mp3"
        command = [
            'ffmpeg', '-i', input_file, '-ss',
            start_time, '-to', end_time, '-c',
            'copy', output_file,
        ]
        subprocess.run(command)
        start_time = end_time
        part_number += 1

    # Handle the last segment
    output_file = f"{output_prefix}_part{part_number}.mp3"
    command = [
        'ffmpeg', '-i', input_file, '-ss', start_time, '-c',
        'copy', output_file,
    ]
    subprocess.run(command)

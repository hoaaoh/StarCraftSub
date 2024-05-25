import math
import json
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

import src.utils as utils


load_dotenv()


class Whisper:
    def __init__(self, system_prompt_file: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
        self.prompt = self.set_prompt(system_prompt_file)
        self.transcript = []
        ...

    def set_prompt(self, fn: str) -> str:
        with open(fn) as f:
            lines = f.readlines()
        prompt = "".join(lines)
        return prompt

    def get_transcripts(
        self,
        audio_prefix: str,
        output_prefix: str,
        cut_points: List[float],
    ) -> None:
        cut_points = [0] + cut_points
        count_of_part = len(cut_points)
        for i in range(1, count_of_part+1):
            audio_file = f"{audio_prefix}_part{i}.mp3"
            transcript_file = f"{output_prefix}_unprocessed_{i}.json"
            if os.path.isfile(transcript_file):
                self.transcript.append(json.load(open(transcript_file)))
                continue
            transcript = self._get_transcript(audio_file, transcript_file)
            self.transcript.append(transcript)

        for i in range(1, count_of_part+1):
            transcript_file = f"{output_prefix}_unprocessed_{i}.json"
            transcript = json.load(open(transcript_file))
            for segment in transcript["segments"]:
                segment["start"] += cut_points[i-1]
                segment["end"] += cut_points[i-1]
            processed_file = f"{output_prefix}_processed_{i}.json"
            json.dump(transcript, open(processed_file, "w"))

    def _get_transcript(self, audio_file: str, output_file: str) -> None:
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=open(audio_file, "rb"),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="zh",
            prompt=self.prompt
        )
        return json.dump(
            transcript.to_dict(),
            open(output_file, "w")
        )

    def gen_audios(self, full_audio_file: str, audio_prefix: str) -> List[str]:
        def check_contains_file(directory):
            items = os.listdir(directory)
            for item in items:
                if os.path.isfile(os.path.join(directory, item)):
                    return True
            return False

        duration = utils.get_audio_duration(full_audio_file)
        time_span = 300  # 5 minutes
        silences = utils.parse_silence_output(utils.detect_silence(full_audio_file))
        current_time = 0
        cut_points = []
        cut_timestamps = []
        while 1.5 * time_span + current_time < duration:
            target_time = current_time + time_span
            closest_time = utils.find_closest_time(silences, target_time)
            if closest_time <= current_time:
                closest_time = min(current_time + time_span, duration)
            cut_points.append(utils.second_to_time_format(closest_time))
            cut_timestamps.append(closest_time)
            current_time = closest_time

        if not check_contains_file(os.path.dirname(audio_prefix)):
            utils.cut_audio(full_audio_file, audio_prefix, cut_points)
        return cut_timestamps


class Parser:
    def __init__(self, file_dir: str):
        self.texts = None
        self.start = None
        self.end = None
        self.segments = []
        self.read(file_dir)

    def get_text_segments(self):
        self.texts = [segment["text"] for segment in self.segments]
        self.start = [segment["start"] for segment in self.segments]
        self.end = [segment["end"] for segment in self.segments]

    def read(self, file_dir: str):
        files = os.listdir(file_dir)
        json_files = [
            f for f in files if f.endswith(".json") and "_processed_" in f]
        for json_file in json_files:
            dic = json.load(open(file_dir + "/" + json_file))
            self.segments.extend(dic["segments"])
        self.segments.sort(key=lambda x: x["start"])
        self.get_text_segments()

    def generate_srt(self, fn: str, to_file: bool = False) -> str:
        def to_srt(timestamp):
            second = int(math.floor(timestamp))
            millisecond = int(math.floor(timestamp*1000)) % 1000
            hour, second = divmod(second, 3600)
            minute, second = divmod(second, 60)
            return f"{hour:02}:{minute:02}:{second:02},{millisecond:03}"

        # test srt formatter
        result = ""
        for idx, (text, start, end) in enumerate(zip(self.texts, self.start, self.end)):
            result += f"{idx+1}\n"
            result += f"{to_srt(start)} --> {to_srt(end)}\n"
            result += f"{text}\n"
            result += "\n"

        with open(fn, "w") as f:
            f.write(result)
        return result


class Refiner:
    def __init__(self):
        self.system_prompt = ""
        self.prompt = ""
        self.client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
        self.completion = None
        ...

    def add_system_prompt(self, fn: str) -> None:
        with open(fn) as f:
            lines = f.readlines()
        self.system_prompt += "\n".join(lines)

    def set_prompt(self, prompt: str) -> None:
        # we have text sequences here
        # pass in those keywords into the prompts
        self.prompt = prompt
        ...

    def refine_sequence(self):
        self.completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt},
            ],
            response_format={"type": "json_object"},
        )
        return self.completion.to_dict()["choices"][0]["message"]["content"]

import json

import src.gen_srt as gen


if __name__ == "__main__":
    audio_source = "./samples/starcraft/rex2.mp3"
    audio_prefix = "./experiment/audio_cuts/rex"
    whisper_prefix = "./experiment/whisper/rex"
    whisper = gen.Whisper("./dictionary/starcraft/whisper/prompt.txt")
    cut_points = whisper.gen_audios(audio_source, audio_prefix)
    whisper.get_transcripts(audio_prefix, whisper_prefix, cut_points)

    srt_prefix = "./experiment/srt/rex"
    parser = gen.Parser("./experiment/whisper/")

    refiner = gen.Refiner()
    refiner.add_system_prompt("./dictionary/starcraft/llm_refine/system_prompt.txt")
    refiner.add_system_prompt("./dictionary/starcraft/llm_refine/info_match.txt")
    refiner.add_system_prompt("./dictionary/starcraft/llm_refine/characters_chinese.txt")

    # refiner.add_prompt("\n".join(parser.texts[:100]))
    print(len(parser.texts))
    result = []
    fp = open("./experiment/llm/testing.txt", "w")
    for start in range(0, len(parser.texts), 50):
        print(f"running {start} to {min(len(parser.texts), start+50)}")
        target = parser.texts[start:start+50]
        target = [f"{idx+1}. {text}" for idx, text in enumerate(target)]
        print(target)
        refiner.set_prompt("\n".join(target))
        refined = refiner.refine_sequence()
        refined_list = json.loads(refined)["transcripts"].strip().split("\n")
        refined_list = ["".join(sentence.split(".")[1:]).strip() for sentence in refined_list]
        print(len(refined_list), len(target))
        assert len(refined_list) == len(target)
        result.extend(refined_list)
        fp.write("\n".join(refined_list))

    # print("\n".join(parser.texts))
    # print(refined)
    json.dump(result, open("./experiment/llm/testing.json", "w"))
    # parser.generate_srt(srt_prefix + ".srt", True)

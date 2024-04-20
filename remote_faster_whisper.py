#!/usr/bin/env python3

# Remote Faster Whisper
# An API interface for Faster Whisper to parse audio over HTTP
#
#    Copyright (C) 2023 Joshua M. Boniface <joshua@boniface.me>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

from configargparse import ArgParser
from flask import Flask, Blueprint, request, Response, jsonify
from speech_recognition.audio import AudioData
from faster_whisper import WhisperModel
from io import BytesIO
from os.path import exists
from os import makedirs
from time import time
from yaml import safe_load
from speech_recognition import Recognizer, AudioFile
from numpy import float32
import numpy as np
from soundfile import read as sf_read
from re import sub, search

import time
import json

class FasterWhisperApi:
    def __init__(
        self,
        listen="127.0.0.1",
        port=9876,
        base_url="/api/v0",
        faster_whisper_config={},
        transformations={},
        audio_file_reqest_param = 'file',
    ):
        """
        Initialize the API and Faster Whisper configuration
        """
        self.app = Flask(__name__)
        self.blueprint = Blueprint("api", __name__, url_prefix=base_url)

        self.listen = listen
        self.port = port

        self.transformations = transformations

        self.model_cache_dir = faster_whisper_config.get(
            "model_cache_dir", "/tmp/whisper-cache"
        )
        self.model = faster_whisper_config.get("model", "base")
        self.device = faster_whisper_config.get("device", "auto")
        self.device_index = faster_whisper_config.get("device_index", 0)
        self.compute_type = faster_whisper_config.get("compute_type", "int8")
        self.beam_size = faster_whisper_config.get("beam_size", 5)
        self.translate = faster_whisper_config.get("translate", False)
        self.language = faster_whisper_config.get("language", None)
        self.local_files_only = faster_whisper_config.get("local_files_only", False)        
        if not self.language:
            self.language = None

        self.save_audio = faster_whisper_config.get("debug", {}).get("save_audio")
        if self.save_audio:
            self.save_path = faster_whisper_config.get("debug", {}).get("save_path")

        if self.save_audio:
            if not exists(self.save_path):
                makedirs(self.save_path)

        @self.blueprint.route("/transcribe", methods=["POST"])
        def transcribe():

            def result_generator( info, segments):

                # return json objects with newline
                with self.app.app_context():
                    attributes = json.dumps(info._asdict())
                    yield '{"TranscriptionInfo" : ' + attributes + ' }\n'

                    try:
                         for segment in segments:
                            attributes = json.dumps(segment._asdict())
                            yield '{"Segment" : ' + attributes + ' }\n'
                    except Exception as e:
                        exception_info = f"Exception Type: {type(e)}\n"
                        exception_info += f"Exception Arguments: {e.args}\n"
                        exception_info += f"Exception Message: {e}"
                        print(exception_info)
            try:
                if audio_file_reqest_param in request.files:
                    audio_input = request.files[audio_file_reqest_param]
                elif request.data:
                    audio_input = np.frombuffer(request.data, dtype=np.float32)
                else:
                    raise ValueError("Request did not contain any audio file or binary audio data")

            except Exception as e:
                return {
                    "message": e.args[0]
                }, 400

            try:

                # Parse JSON string into a Python list
                if request.args.get('temperature') is not None:
                    data_list = json.loads(request.args.get('temperature'))
                    # Convert list elements to floats and create a tuple
                    temperature = tuple(float(x) for x in data_list)
                else:
                    # passing temperature=None does not result in using paramter default in transcribe() -> define default here
                    temperature = [
                        0.0,
                        0.2,
                        0.4,
                        0.6,
                        0.8,
                        1.0,
                    ]
                    
                segments, info = self.whisper_model.transcribe(
                    audio=audio_input,
                    language=request.args.get('language'),
                    task=request.args.get('task'),
                    beam_size=int(request.args.get('beam_size')) if request.args.get('beam_size') is not None else None,
                    best_of=int(request.args.get('best_of')) if request.args.get('best_of') is not None else None,
                    patience=float(request.args.get('patience')) if request.args.get('patience') is not None else None,
                    length_penalty=float(request.args.get('length_penalty')) if request.args.get('length_penalty') is not None else None,
                    repetition_penalty=float(request.args.get('repetition_penalty')) if request.args.get('repetition_penalty') is not None else None,
                    no_repeat_ngram_size=int(request.args.get('no_repeat_ngram_size')) if request.args.get('no_repeat_ngram_size') is not None else None,
                    temperature=temperature,
                    compression_ratio_threshold=float(request.args.get('compression_ratio_threshold')) if request.args.get('compression_ratio_threshold') is not None else None,
                    log_prob_threshold=float(request.args.get('log_prob_threshold')) if request.args.get('log_prob_threshold') is not None else None,
                    no_speech_threshold=float(request.args.get('no_speech_threshold')) if request.args.get('no_speech_threshold') is not None else None,
                    condition_on_previous_text=bool(request.args.get('condition_on_previous_text')) if request.args.get('condition_on_previous_text') is not None else None,
                    prompt_reset_on_temperature=bool(request.args.get('prompt_reset_on_temperature')) if request.args.get('prompt_reset_on_temperature') is not None else None,
                    initial_prompt=request.args.get('initial_prompt'),
                    suppress_blank=bool(request.args.get('suppress_blank')) if request.args.get('suppress_blank') is not None else None,
                    suppress_tokens=json.loads(request.args.get('suppress_tokens')) if request.args.get('suppress_tokens') is not None else None,
                    word_timestamps=True if request.args.get('word_timestamps') == 'True' else False,
                    prepend_punctuations=bool(request.args.get('prepend_punctuations')) if request.args.get('prepend_punctuations') is not None else None,
                    append_punctuations=bool(request.args.get('append_punctuations')) if request.args.get('append_punctuations') is not None else None,
                    hallucination_silence_threshold=float(request.args.get('hallucination_silence_threshold')) if request.args.get('hallucination_silence_threshold') is not None else None,
                    vad_filter=bool(request.args.get('vad_filter') == 'True') if request.args.get('vad_filter') is not None else None,
                    vad_parameters=json.loads(request.args.get('vad_parameters')) if request.args.get('vad_parameters') is not None else None,
                )
            except Exception as e:
                exception_info = f"Exception Type: {type(e)}\n"
                exception_info += f"Exception Arguments: {e.args}\n"
                exception_info += f"Exception Message: {e}"
                return {
                    # Generate a string containing essential information about the exception
                    "message": "Calling model failed: " + exception_info
                }, 400
            
            # call generator function which processes each segment
            return Response(result_generator(info, segments))

        self.app.register_blueprint(self.blueprint)

    def start(self):
        """
        Initialize the WhisperModel (including downloading the model files) and start the API
        """
        print("Initializing WhisperModel instance")
        self.whisper_model = WhisperModel(
            self.model,
            device=self.device,
            device_index=self.device_index,
            compute_type=self.compute_type,
            download_root=self.model_cache_dir,
            local_files_only=self.local_files_only
        )

        print("Starting API")
        self.app.run(debug=False, host=self.listen, port=self.port)

    def perform_faster_whisper_recognition(self, audio_data):
        """
        Perform recognition on {audio_data} with model
        """
        print("Performing recognition on audio data")

        t_start = time()
        # wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        # wav_stream = BytesIO(wav_bytes)
        # audio_array, sampling_rate = sf_read(wav_stream)
        # audio_array = audio_array.astype(float32)

        segments, info = self.whisper_model.transcribe(
            audio_data,
            beam_size=self.beam_size,
            language=self.language,
            task="translate" if self.translate else "transcribe",
        )

        found_text = list()
        for segment in segments:
            found_text.append(segment.text)
        text = " ".join(found_text).strip()

        # Perform transformations on text
        if self.transformations:
            if 'lower' in self.transformations:
                text = text.lower()
            if 'casefold' in self.transformations:
                text = text.casefold()
            if 'upper' in self.transformations:
                text = text.upper()
            if 'title' in self.transformations:
                text = text.title()
            for tr in [tr for tr in self.transformations if isinstance(tr, list) and search(tr[0], text)]:
                _text = text
                text = sub(tr[0], tr[1], text)
                print(f'Transforming "{tr[0]}" -> "{tr[1]}": pre: "{_text}", post: "{text}"')

        t_end = time()
        t_run = t_end - t_start

        result = {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "sample_duration": info.duration,
            "runtime": t_run,
        }

        print(f"Result: {result}")
        return result


def parse_args():
    """
    Parse CLI arguments/environment variables (configuration file path)
    """
    p = ArgParser()
    p.add(
        "-c",
        "--config",
        env_var="RFW_CONFIG_FILE",
        help="Configuration file path",
        required=True,
    )
    options = p.parse_args()
    return options


def parse_config(configfile):
    """
    Parse YAML configuration into {config} dictionary
    """
    with open(configfile, "r") as fh:
        config = safe_load(fh)

    return config


def start_api():
    """
    Parse arguments, grab configuration, and initialize and start the API
    """
    options = parse_args()
    config = parse_config(options.config)
    api = FasterWhisperApi(
        **config["daemon"],
        faster_whisper_config=config["faster_whisper"],
        transformations=config["transformations"],
    )
    api.start()


# Main entrypoint
if __name__ == "__main__":
    start_api()

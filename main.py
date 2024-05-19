
from flask import Flask, request, jsonify
from flask import Response, stream_with_context


from langchain import LLMChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts import Prompt

from flask_cors import CORS

import re




import os

import trainer

import time
from pydub.playback import play
from deepgram import DeepgramClient, SpeakOptions


from pydub import AudioSegment



app = Flask(__name__)
cors = CORS(app)


openai_api_key = os.environ["OPENAI_API_KEY"]

API_SECRET = os.environ["API_SECRET"]

deepgram_api_key = os.environ["DEEPGRAM_API_KEY"]




last_api_call_time = 0
history = []
llmChain = None

# inserting data into ai advocate history
username = ""
password = ""
hostname = ""
database_name = ""


# Construct the connection URL
db_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"







searched = 0
previous_response = ""

def clean_response(response):
    # Define a regex pattern to match valid characters
    pattern = r'[^a-zA-Z0-9,. ]'

    # Use re.sub to replace invalid characters with an empty string
    cleaned_response = re.sub(pattern, '', response)

    return cleaned_response

# Function to synthesize audio from text using Deepgram
def synthesize_audio(text, deepgram_api_key=deepgram_api_key):
    deepgram_api_key = deepgram_api_key
    # Create a Deepgram client using the API key
    deepgram = DeepgramClient(api_key=deepgram_api_key)
    # Choose a model to use for synthesis
    options = SpeakOptions(
        model="aura-helios-en"  # Specify the desired audio format
    )
    speak_options = {"text": text}
    # Synthesize audio and stream the response
    response = deepgram.speak.v("1").stream(speak_options, options)
    # Get the audio stream from the response
    audio_buffer = response.stream

    # Read the audio data from the buffer
    audio_data = audio_buffer.read()

    return audio_data



@app.route("/", methods=["GET"])
def index():
    return "API Online"


# Dictionary to store unique histories for each caregiver_id
careteam_histories = {}

previous_response = {}

searched = {}
last_api_call_times = {}
agent = 'undefined'





def reset_history(careteam_id):
    careteam_histories[careteam_id] = []

@app.route("/voiceadvocate", methods=["POST"])
def askvoice():
    global last_api_call_time
    global llmChain
    global count1

    username = ""
    password = ""
    hostname = ""
    database_name = ""
    db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

    api_secret_from_frontend = request.headers.get('X-API-SECRET')
    if api_secret_from_frontend != API_SECRET:
        return jsonify({'error': 'Unauthorized access'}), 401

    careteam_id = request.headers.get('careteamid')
    caregiver_id = request.headers.get('userid')

    if careteam_id == "not implied" or caregiver_id == "not implied":
        return jsonify({'message': "Caregiver or careteam id not implied"})

    try:
        reqData = request.get_json()
        user_question = reqData['question']
        user_address = request.headers.get('userprimaddress')
        print(f"All Headers: {request.headers}")

        current_time = time.time()
        last_api_call_time_for_caregiver = last_api_call_times.get(careteam_id, 0)
        if current_time - last_api_call_time_for_caregiver > 600:
            reset_history(careteam_id)
            last_api_call_times[careteam_id] = current_time




            # Only confirm address if the question is related to a search
            user_location = trainer.get_coordinates(user_address)
            count1 = trainer.train(user_location)  # Train based on user location for the first call of a session
            print(count1)


        if not llmChain:

            # Initialize llmChain if it's not initialized yet
            with open("training/mastervoice.txt", "r" ,encoding = 'utf-8') as f:
                promptTemplate = f.read()

            prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])
            llmChain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0.5 ,max_tokens = 100,
                                                              model_name="gpt-4o",
                                                              openai_api_key=openai_api_key, streaming = True))

        careteam_history = careteam_histories.setdefault(careteam_id, [])

        @stream_with_context
        def generate_response():
            # Record the start time
            start_time = time.time()

            # Stream response from LLMChain
            response_stream = llmChain.predict(question=user_question, context="\n\n".join(careteam_history),
                                               history=careteam_history)

            # Initialize a buffer to store the sentence
            sentence_buffer = ""
            sentence_index = 1  # Initialize sentence index

            # Iterate over the stream of chunks
            for chunk in response_stream:
                # Append the chunk to the buffer
                sentence_buffer += chunk

                # Check if the sentence is complete
                if sentence_buffer.endswith(('.', '!', '?')):
                    # Record the end time
                    end_time = time.time()

                    # Calculate the elapsed time
                    elapsed_time = end_time - start_time

                    # Log the complete sentence
                    print(f"Sentence {sentence_index}: {sentence_buffer.strip()}")
                    sentence_index += 1

                    print(f"Elapsed time for LLMChain: {elapsed_time} seconds")

                    # Synthesize audio from the sentence
                    #audio_data = synthesize_audio(sentence_buffer.strip())

                    # Play the synthesized audio
                    #play(audio_data)
                    # Yield the sentence
                    yield f"{sentence_buffer.strip()}\n"
                    sentence_buffer = ""

            # Yield any remaining buffered content
            if sentence_buffer:
                print(f"Sentence {sentence_index}: {sentence_buffer.strip()}")
                yield f"{sentence_buffer.strip()}\n"

            careteam_history.append(f"Bot: {sentence_buffer.strip()}")
            careteam_history.append(f"Human: {user_question}")


        return Response(generate_response(), content_type='text/plain')



    except Exception as e:
        return jsonify({"answer": None, "success": False, "message": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)


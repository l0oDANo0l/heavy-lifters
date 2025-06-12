from dotenv import load_dotenv
load_dotenv() 

import asyncio
import random

import numpy as np
import sounddevice as sd

from agents import (
    Agent,
    function_tool,
)
from agents.voice import (
    StreamedAudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)

IS_RECORDING = False
CHUNK_LENGTH_S = 0.05  # 100ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


async def start_voice_pipeline(pipeline, audio_player, audio_input) -> None:
        try:
            audio_player.start()
            result = await pipeline.run(audio_input)

            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    print(f"event_audio")
                    audio_player.write(event.data)
                elif event.type == "voice_stream_event_lifecycle":
                    print(f"Lifecycle event: {event.event}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            audio_player.close()


async def send_mic_audio(audio_input) -> None:
        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue
                
                data, _ = stream.read(read_size)

                await audio_input.add_audio(data)
                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()    


async def main():
    
    # Use the 'VoicePipeline' which is best for realtime convos. (https://openai.github.io/openai-agents-python/voice/quickstart/)
    agent = Agent(
        name="Assistant",
        instructions=
            "You're speaking to a human, so be polite and concise. The conversation should always be in English.",
        model="gpt-4o-mini",
        tools=[get_weather],
    )
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    audio_player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    audio_input = StreamedAudioInput()

    await asyncio.gather(
        start_voice_pipeline(pipeline, audio_player, audio_input),
        send_mic_audio(audio_input)
    );

if __name__ == "__main__":
    asyncio.run(main())

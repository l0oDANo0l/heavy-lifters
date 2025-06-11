from dotenv import load_dotenv
load_dotenv() 

import asyncio
import random

import numpy as np
import sounddevice as sd

from agents import (
    Agent,
    function_tool,
    set_tracing_disabled,
)
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."

agent = Agent(
    name="Assistant",
    instructions=
        "You're speaking to a human, so be polite and concise. The conversation should always be in English.",
    model="gpt-4o-mini",
    tools=[get_weather],
)


async def main():
    
    # Use the 'VoicePipeline' Which is best for realtime convos. (https://openai.github.io/openai-agents-python/voice/quickstart/)
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    duration = 3  # seconds
    samplerate = 24000

    # TODO: Switch over to detecting audio, and start / stop of conversation!
    print("Speak now... say something like, what's the weather in portland oregon today?")
    buffer = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print(f"[debug] event: got audio")
    audio_input = AudioInput(buffer)

    result = await pipeline.run(audio_input)

    async for event in result.stream():
        print(f"[debug] event: {event.type}")
        if event.type == "voice_stream_event_audio":
            # play audio
            player.write(event.data)


if __name__ == "__main__":
    asyncio.run(main())

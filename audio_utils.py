```python
import base64
from io import BytesIO
from gtts import gTTS

def toggle_audio_player(text, key):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_base64 = base64.b64encode(mp3_fp.read()).decode()

    audio_id = f"audio_{key}"

    audio_html = rf"""
    <audio id="{audio_id}" src="data:audio/mp3;base64,{audio_base64}"></audio>
    <script>
    if (!window.activeAudio) {{
        window.activeAudio = null;
        window.activeButton = null;
    }}

    const btn_{key} = document.getElementById("btn_{key}");
    const audio_{key} = document.getElementById("{audio_id}");

    function toggleAudio_{key}() {{
        if (window.activeAudio && window.activeAudio !== audio_{key}) {{
            window.activeAudio.pause();
            window.activeAudio.currentTime = 0;
            if (window.activeButton) {{
                window.activeButton.innerText = "🔊";
            }}
        }}

        if (audio_{key}.paused) {{
            audio_{key}.play();
            btn_{key}.innerText = "❌";
            window.activeAudio = audio_{key};
            window.activeButton = btn_{key};
        }} else {{
            audio_{key}.pause();
            audio_{key}.currentTime = 0;
            btn_{key}.innerText = "🔊";
            window.activeAudio = null;
            window.activeButton = null;
        }}
    }}

    if (btn_{key}) {{
        btn_{key}.onclick = toggleAudio_{key};
    }}

    audio_{key}.addEventListener("ended", function() {{
        btn_{key}.innerText = "🔊";
        if (window.activeAudio === audio_{key}) {{
            window.activeAudio = null;
            window.activeButton = null;
        }}
    }});
    </script>
    """
    return audio_html

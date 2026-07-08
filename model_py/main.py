import json

from config.settings import VIDEO_PATH
from models.state import AgentState
from services.vision_service import analyze_video
from services.brain_service import analyze_state


def main():

    print(f"{VIDEO_PATH} okunuyor...")

    world_state = analyze_video(VIDEO_PATH)

    print("\n===== WORLD STATE =====")
    print(json.dumps(world_state, indent=2, ensure_ascii=False))

    state = AgentState(
        video_path=VIDEO_PATH,
        vision=world_state
    )

    print("\n===== STATE =====")
    print(json.dumps(state.__dict__, indent=2, ensure_ascii=False))

    print("\n🧠 AŞAMA 2: Beyin modeli çalışıyor...\n")

    message = analyze_state(world_state)

    print("--- 🎯 NİHAİ SİSTEM ÇIKTISI ---")

    if message.tool_calls:
        print("🚨 SİSTEM BİR AKSİYON TETİKLEDİ 🚨")

        for tool in message.tool_calls:
            print(f"Fonksiyon: {tool.function.name}")
            print(f"Parametre: {tool.function.arguments}")

    else:
        print(message.content)


if __name__ == "__main__":
    main()

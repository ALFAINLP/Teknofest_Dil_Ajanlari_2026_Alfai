import json
import os
from typing import List, Dict, Any


class MemoryManager:
    """
    Agent hafızasını yöneten sınıf.

    Desteklenen hafızalar:
    - Event Memory
    - Risk Memory
    - Tool Memory
    - Persistent Memory (JSON)
    - Temporal Context

    İleride eklenebilir:
    - Semantic Memory
    """

    @staticmethod
    def add_events(
        memory: List[Dict[str, Any]],
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Yeni olayları hafızaya ekler.
        """
        if memory is None:
            memory = []

        memory.extend(events)
        return memory

    @staticmethod
    def add_risk(
        memory: List[str],
        risk: str
    ) -> List[str]:
        """
        Yeni risk seviyesini hafızaya ekler.
        """
        if memory is None:
            memory = []

        memory.append(risk)
        return memory

    @staticmethod
    def count_similar_events(
        memory: List[Dict[str, Any]],
        keyword: str
    ) -> int:
        """
        Hafızada belirtilen anahtar kelimeyi içeren olayların sayısını döndürür.
        """
        if memory is None:
            return 0

        count = 0

        for event in memory:
            event_text = str(event.get("event", "")).lower()

            if keyword.lower() in event_text:
                count += 1

        return count

    @staticmethod
    def add_tool(
        memory: List[str],
        tool_name: str
    ) -> List[str]:
        """
        Kullanılan aracı hafızaya ekler.
        """
        if memory is None:
            memory = []

        memory.append(tool_name)
        return memory

    @staticmethod
    def was_tool_used(
        memory: List[str],
        tool_name: str
    ) -> bool:
        """
        Belirtilen aracın daha önce kullanılıp kullanılmadığını kontrol eder.
        """
        if memory is None:
            return False

        return tool_name in memory

    @staticmethod
    def recent_risk_count(
        memory: List[str],
        risk_level: str,
        limit: int = 3
    ) -> int:
        """
        Son N risk kaydı içinde belirtilen risk seviyesinin kaç kez geçtiğini döndürür.
        """
        if memory is None:
            return 0

        recent = memory[-limit:]

        return sum(
            1 for risk in recent
            if risk == risk_level
        )

    @staticmethod
    
    def analyze_temporal_context(
        event_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Geçmiş olayları analiz ederek tekrar eden örüntüleri
        ve sistematik riskleri tespit eder.
        """

        summary = {
            "total_events": len(event_history),
            "repeated_events": 0,
            "same_zone": False,
            "dominant_zone": "",
            "dominant_event": "",
            "systematic_risk": False,
            "recommended_action": "Normal İzleme"
        }

        if not event_history:
            return summary

        # Son 10 olayı analiz et
        recent_events = event_history[-10:]

        # ==========================
        # Bölge Analizi
        # ==========================
        zone_counter = {}

        for event in recent_events:

            zone = event.get("zone")

            if zone:
                zone_counter[zone] = zone_counter.get(zone, 0) + 1

        if zone_counter:

            dominant_zone = max(
                zone_counter,
                key=zone_counter.get
            )

            summary["dominant_zone"] = dominant_zone

            # Aynı bölge en az 3 kez görülmüşse
            if zone_counter[dominant_zone] >= 3:
                summary["same_zone"] = True

        # ==========================
        # Olay Analizi
        # ==========================
        event_counter = {}

        for event in recent_events:

            event_info = event.get("event", "")

            # event bazen dict bazen string olabilir
            if isinstance(event_info, dict):
                description = event_info.get("description", "")
            else:
                description = str(event_info)

            description = (
                description.strip()
                .lower()
                .replace(".", "")
                .replace(",", "")
                .replace("!", "")
                .replace("?", "")
            )

            if not description:
                continue

            # İlk 5 kelimeyi olay imzası olarak kullan
            signature = " ".join(description.split()[:5])

            event_counter[signature] = (
                event_counter.get(signature, 0) + 1
            )

        if event_counter:

            dominant_event = max(
                event_counter,
                key=event_counter.get
            )

            repeated = event_counter[dominant_event]

            summary["dominant_event"] = dominant_event
            summary["repeated_events"] = repeated

            # Aynı olay aynı bölgede en az 3 kez görülmüşse
            if repeated >= 3 and summary["same_zone"]:

                summary["systematic_risk"] = True

                summary["recommended_action"] = (
                    "Aynı bölgede tekrar eden benzer olaylar tespit edildi. "
                    "Risk seviyesini artır ve bakım ekibini bilgilendir."
                )

        return summary

    @staticmethod
    def save_memory(
        event_history: List[Dict[str, Any]],
        risk_history: List[str],
        tool_history: List[str]
    ) -> None:
        """
        Hafızayı JSON dosyasına kaydeder.
        """

        os.makedirs("memory", exist_ok=True)

        memory = {
            "event_history": event_history,
            "risk_history": risk_history,
            "tool_history": tool_history
        }

        with open(
            "memory/memory.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                memory,
                f,
                ensure_ascii=False,
                indent=4
            )

    @staticmethod
    def load_memory() -> Dict[str, Any]:
        """
        Hafızayı JSON dosyasından yükler.
        """

        path = "memory/memory.json"

        if not os.path.exists(path):
            return {
                "event_history": [],
                "risk_history": [],
                "tool_history": []
            }

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "trigger_operator_alert",
            "description": "Tespit edilen kritik olayda operatöre acil uyarı mesajı göndermek için kullanılır. Sağlık veya güvenlik müdahalesi gerektiren durumlarda çağrılır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alert_level": {
                        "type": "string",
                        "description": "Uyarı seviyesi: 'Yüksek' veya 'Kritik'"
                    },
                    "raw_incident_context": {
                        "type": "string",
                        "description": "Olayın kısa açıklaması"
                    }
                },
                "required": ["alert_level", "raw_incident_context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_previous_incidents",
            "description": "Geçmişte benzer olayların yaşanıp yaşanmadığını sorgular. Örneğin 'forklift devrilmesi' gibi anahtar kelimelerle geçmiş vakaları arar. Eğer benzer vaka varsa, bu bilgi risk değerlendirmesini etkiler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_keywords": {
                        "type": "string",
                        "description": "Sorgulanacak olay türü veya anahtar kelimeler (örn: 'forklift devrilmesi', 'kimyasal sızıntı')"
                    }
                },
                "required": ["incident_keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_personnel_on_duty",
            "description": "Belirli bir zaman damgasında ve bölgede görevli personel listesini getirir. Bu bilgi, acil durumda kimi arayacağını bilmek için kullanılır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "integer",
                        "description": "Sorgulanacak zaman damgası (saniye cinsinden TAM SAYI, örn: 15)"
                    },
                    "zone": {
                        "type": "string",
                        "description": "Bölge adı (örn: 'Lojistik Saha - Peron 3')"
                    }
                },
                "required": ["timestamp", "zone"]
            }
        }
    }
]
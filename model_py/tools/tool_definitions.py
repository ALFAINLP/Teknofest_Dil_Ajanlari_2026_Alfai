TOOLS = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "saglik_ekibini_cagir",
    #         "description": "Sahada acil tıbbi müdahale gerektiren bir durum tespit edildiğinde sağlık ekibini çağırmak için kullanılır.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "sebep": {
    #                     "type": "string",
    #                     "description": "Sağlık ekibinin çağrılma nedeni"
    #                 }
    #             },
    #             "required": ["sebep"]
    #         }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "guvenligi_uyar",
    #         "description": "Sahada güvenlik personelinin müdahale etmesi gereken durumlarda kullanılır.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "mesaj": {
    #                     "type": "string",
    #                     "description": "Güvenlik personeline iletilecek uyarı mesajı"
    #                 }
    #             },
    #             "required": ["mesaj"]
    #         }
    #     }
    # },
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
    }
]

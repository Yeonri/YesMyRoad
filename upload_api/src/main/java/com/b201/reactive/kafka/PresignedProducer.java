package com.b201.reactive.kafka;

import com.b201.reactive.dto.PresignedMessage;
import lombok.RequiredArgsConstructor;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class PresignedProducer {

    private final KafkaTemplate<String, PresignedMessage> kafkaTemplate;

    public void send(PresignedMessage message) {

        kafkaTemplate.send("presigned-topic", message);
    }
}

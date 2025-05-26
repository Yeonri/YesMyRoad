package com.b201.reactive.kafka;

import com.b201.reactive.dto.RawMessage;
import lombok.RequiredArgsConstructor;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class RawProducer {

    private final KafkaTemplate<String, RawMessage> kafkaTemplate;

    public void send(RawMessage message) {
        kafkaTemplate.send("raw-topic", message);
    }
}
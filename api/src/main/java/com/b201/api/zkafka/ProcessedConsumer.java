package com.b201.api.zkafka;

import com.b201.api.dto.AiResultDto;
import com.b201.api.service.AiResultService;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;


@Slf4j
@Component
@AllArgsConstructor
public class ProcessedConsumer {

    private AiResultService aiResultService;

    @KafkaListener(
            topics = "processed-topic",
            groupId = "processed-group",
            containerFactory = "processedMessageKafkaListenerContainerFactory"
    )
    public void consume(ConsumerRecord<String, ProcessedMessage> record) {
        ProcessedMessage message = record.value();

        log.info("Consumed processed message: {}", message);

        AiResultDto aiResultDto = AiResultDto.builder()
                .captureTimestampUtc(message.getCaptureTimestampUtc()) // 문자열 → LocalDateTime
                .location(
                        new AiResultDto.Location(
                                message.getLocation().getLatitude(),
                                message.getLocation().getLongitude(),
                                message.getLocation().getAccuracyMeters()
                        )
                )
                .imageInfo(
                        new AiResultDto.ImageInfo(
                                message.getImageInfo().getImageUrl(),
                                message.getImageInfo().getRisk()
                        )
                )
                .detections(
                        message.getDetections().stream()
                                .map(d -> new AiResultDto.Detection(d.getCategoryName()))
                                .toList()
                )
                .build();
        log.info(aiResultDto.toString());

        aiResultService.addAiResult(aiResultDto);

    }
}

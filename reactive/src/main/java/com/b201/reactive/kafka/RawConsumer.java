package com.b201.reactive.kafka;

import com.b201.reactive.dto.ImageInfo;
import com.b201.reactive.dto.Location;
import com.b201.reactive.dto.PresignedMessage;
import com.b201.reactive.dto.RawMessage;
import com.b201.reactive.util.S3Util;
import lombok.RequiredArgsConstructor;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class RawConsumer {

    private final S3Util s3Util;
    private final PresignedProducer presignedMessageProducer;

    @KafkaListener(
            topics = "raw-topic",
            groupId = "presigned-group",
            containerFactory = "rawMessageKafkaListenerContainerFactory"
    )
    public void consume(ConsumerRecord<String, RawMessage> record) {
        RawMessage rawMessage = record.value();
        String uuid = rawMessage.getImageInfo().getUuid();
        String objectKey = uuid + ".jpg";

        s3Util.createPresignedUrlAsync(objectKey)
                .doOnNext(presignedUrl -> {
                    Location location = rawMessage.getLocation();
                    ImageInfo imageInfo = new ImageInfo(
                            uuid,
                            presignedUrl,  // presignedUrl 추가됨
                            objectKey,          // imageUrl은 아직 null
                            null           // risk는 아직 null
                    );
                    PresignedMessage presignedMessage = new PresignedMessage(
                            rawMessage.getCaptureTimestampUtc(),
                            location,
                            imageInfo
                    );
                    presignedMessageProducer.send(presignedMessage);
                })
                .subscribe();

    }
}

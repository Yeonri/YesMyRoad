package com.b201.reactive.service;

import com.b201.reactive.dto.ImageInfo;
import com.b201.reactive.dto.Location;
import com.b201.reactive.dto.RawMessage;
import com.b201.reactive.kafka.RawProducer;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.io.File;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class DeviceUploadService {

    private static final Logger log = LoggerFactory.getLogger(DeviceUploadService.class);
    private final RawProducer rawMessageProducer;

    private static final String IMAGE_DIR = "/mnt/image-buffer/";

    public Mono<String> processUpload(FilePart image, String latitude, String longitude) {
        LocalDateTime captureTimestampUtc = LocalDateTime.now();
        String uuid = UUID.randomUUID().toString();
        String localPath = IMAGE_DIR + uuid + ".jpg";

        log.info("processUpload");

        return image.transferTo(new File(localPath))
                .then(Mono.fromSupplier(() -> {
                    Location location = new Location(Double.parseDouble(latitude), Double.parseDouble(longitude), null);
                    ImageInfo imageInfo = new ImageInfo(uuid, null, null, null);
                    RawMessage rawMessage = new RawMessage(captureTimestampUtc, location, imageInfo);

                    rawMessageProducer.send(rawMessage);
                    return uuid;
                }));
    }
}


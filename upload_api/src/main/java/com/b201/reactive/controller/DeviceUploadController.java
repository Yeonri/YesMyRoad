package com.b201.reactive.controller;

import com.b201.reactive.service.DeviceUploadService;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api")
public class DeviceUploadController {

    private static final Logger log = LoggerFactory.getLogger(DeviceUploadController.class);
    private final DeviceUploadService deviceUploadService;

    @PostMapping(value= "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<ResponseEntity<String>> handleUpload(
            @RequestPart("image") FilePart image,
            @RequestPart("latitude") String latitude,
            @RequestPart("longitude") String longitude
    ) {
        log.info("handleUpload controller");
        return deviceUploadService.processUpload(image, latitude, longitude)
                .map(uuid -> ResponseEntity.ok("uploaded: " + uuid));
    }
}
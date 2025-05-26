package com.b201.image_api.controller;

import com.b201.image_api.service.S3Service;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/")
public class S3Controller {

    private final S3Service s3Service;

    @GetMapping("/{imagename}")
    public ResponseEntity<?> downloadFromS3(@PathVariable String imagename, HttpServletRequest request) {

        return s3Service.downloadFromS3(imagename, request);
    }
}

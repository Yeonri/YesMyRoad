package com.b201.api.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.CapturePointResponseDto;
import com.b201.api.dto.DamageDetailResponseDto;
import com.b201.api.dto.FeatureDto;
import com.b201.api.service.CapturePointService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/capture-points")
@RequiredArgsConstructor
public class CapturePointController {

	private final CapturePointService capturePointService;

	@GetMapping
	public ResponseEntity<CapturePointResponseDto> getCapturePoints() {
		List<FeatureDto> features = capturePointService.findAllFeatures();
		if (features.isEmpty()) {
			return ResponseEntity.noContent().build();
		}

		CapturePointResponseDto body = CapturePointResponseDto.builder().features(features).build();
		return ResponseEntity.ok(body);
	}

	@GetMapping("/{publicId}")
	public ResponseEntity<DamageDetailResponseDto> getDamageDetails(@PathVariable("publicId") String publicId) {
		DamageDetailResponseDto body = capturePointService.findDamageDetail(publicId);
		return ResponseEntity.ok(body);
	}

}

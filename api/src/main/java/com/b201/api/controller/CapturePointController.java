package com.b201.api.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.CapturePointResponseDto;
import com.b201.api.dto.DamageDetailResponseDto;
import com.b201.api.service.CapturePointService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/capture-points")
@RequiredArgsConstructor
public class CapturePointController {

	private final CapturePointService capturePointService;

	@GetMapping
	public ResponseEntity<CapturePointResponseDto> getCapturePoints() {
		CapturePointResponseDto body = capturePointService.findAllFeatures();
		if (body.getFeatures().isEmpty()) {
			// 데이터가 없으면 204 No Content
			return ResponseEntity.noContent().build();
		}
		// 데이터가 있으면 200 + body
		return ResponseEntity.ok(body);
	}

	@GetMapping("/{publicId}")
	public ResponseEntity<DamageDetailResponseDto> getDamageDetails(
		@PathVariable String publicId
	) {
		return capturePointService
			.findDamageDetail(publicId)
			.map(ResponseEntity::ok)
			.orElseGet(() -> ResponseEntity.notFound().build());
	}

}

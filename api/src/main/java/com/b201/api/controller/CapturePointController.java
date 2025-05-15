package com.b201.api.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.damage.DamageDetailResponseDto;
import com.b201.api.dto.point.CapturePointResponseDto;
import com.b201.api.security.CustomUserDetails;
import com.b201.api.service.CapturePointService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RestController
@RequestMapping("/api/capture-points")
@RequiredArgsConstructor
public class CapturePointController {

	private final CapturePointService capturePointService;

	@GetMapping
	public ResponseEntity<CapturePointResponseDto> getCapturePoints(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		CapturePointResponseDto body = capturePointService.findAllFeatures(regionName);
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
		log.info("api/capture-points/{}", publicId);
		return ResponseEntity.of(capturePointService.findDamageDetail(publicId));
	}

}

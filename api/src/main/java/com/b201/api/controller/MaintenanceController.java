package com.b201.api.controller;

import java.util.List;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.maintenance.CompletionStatsDto;
import com.b201.api.dto.maintenance.MaintenanceStatusDto;
import com.b201.api.dto.maintenance.MonthlyMaintenanceStatusDto;
import com.b201.api.dto.maintenance.RegionMaintenanceStatusDto;
import com.b201.api.security.CustomUserDetails;
import com.b201.api.service.MaintenanceService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RestController
@RequestMapping(value = "/api/maintenance", produces = MediaType.APPLICATION_JSON_VALUE)
@RequiredArgsConstructor
public class MaintenanceController {

	private final MaintenanceService maintenanceService;

	@GetMapping("/overview")
	public ResponseEntity<MaintenanceStatusDto> getAllMaintenance(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok().body(maintenanceService.getMaintenanceStatus(regionName));
	}

	@GetMapping("/completion-stats")
	public ResponseEntity<CompletionStatsDto> getCompletionStats(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok().body(maintenanceService.getCompletionStats(regionName));
	}

	@GetMapping("/monthly-status")
	public ResponseEntity<List<MonthlyMaintenanceStatusDto>> getMonthlyStats(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok().body(maintenanceService.getMonthlyMaintenanceStatus(regionName));
	}

	@GetMapping("/districts")
	public ResponseEntity<List<RegionMaintenanceStatusDto>> getDistricts(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok().body(maintenanceService.getRegionMaintenanceStatus(regionName));
	}
}

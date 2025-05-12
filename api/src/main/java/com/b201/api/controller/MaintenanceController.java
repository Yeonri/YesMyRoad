package com.b201.api.controller;

import java.util.List;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.maintenance.CompletionStatsDto;
import com.b201.api.dto.maintenance.MaintenanceStatusDto;
import com.b201.api.dto.maintenance.MonthlyMaintenanceStatusDto;
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
	public ResponseEntity<MaintenanceStatusDto> getAllMaintenance() {
		return ResponseEntity.ok().body(maintenanceService.getMaintenanceStatus());
	}

	@GetMapping("/completion-stats")
	public ResponseEntity<CompletionStatsDto> getCompletionStats() {
		log.info("result: {}", maintenanceService.getCompletionStats());
		return ResponseEntity.ok().body(maintenanceService.getCompletionStats());
	}

	@GetMapping("/monthly-status")
	public ResponseEntity<List<MonthlyMaintenanceStatusDto>> getMonthlyStats() {
		return ResponseEntity.ok().body(maintenanceService.getMonthlyMaintenanceStatus());
	}
}

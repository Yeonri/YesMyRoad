package com.b201.api.service;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.stereotype.Service;

import com.b201.api.dto.maintenance.CompletionStatsDto;
import com.b201.api.dto.maintenance.MaintenanceStatusDto;
import com.b201.api.dto.maintenance.MonthlyMaintenanceStatusDto;
import com.b201.api.repository.CaptureDamageRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MaintenanceService {

	private final CaptureDamageRepository captureDamageRepository;

	public MaintenanceStatusDto getMaintenanceStatus() {
		return captureDamageRepository.countByStauts();
	}

	public CompletionStatsDto getCompletionStats() {
		LocalDateTime now = LocalDateTime.now();
		return captureDamageRepository.getCompletionStatsByPeriod(
			now.minusDays(1),
			now.minusWeeks(1),
			now.minusMonths(1)
		);
	}

	public List<MonthlyMaintenanceStatusDto> getMonthlyMaintenanceStatus() {
		return captureDamageRepository.getMonthlyMaintenanceStatsByPeriod();
	}
}

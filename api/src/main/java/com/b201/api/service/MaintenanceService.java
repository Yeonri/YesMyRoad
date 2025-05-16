package com.b201.api.service;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.stereotype.Service;

import com.b201.api.dto.maintenance.CompletionStatsDto;
import com.b201.api.dto.maintenance.MaintenanceStatusDto;
import com.b201.api.dto.maintenance.MonthlyMaintenanceStatusDto;
import com.b201.api.dto.maintenance.RegionMaintenanceStatusDto;
import com.b201.api.repository.CaptureDamageRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class MaintenanceService {

	private final CaptureDamageRepository captureDamageRepository;

	public MaintenanceStatusDto getMaintenanceStatus(String regionName) {
		log.info("[getMaintenanceStatus] 호출됨");
		MaintenanceStatusDto status = captureDamageRepository.countByStauts(regionName);
		log.debug("[getMaintenanceStatus] result = {}", status);
		return status;
	}

	public CompletionStatsDto getCompletionStats(String regionName) {
		log.info("[getCompletionStats] 호출됨");
		LocalDateTime now = LocalDateTime.now();
		CompletionStatsDto stats = captureDamageRepository.getCompletionStatsByPeriod(
			regionName,
			now.minusDays(1),
			now.minusWeeks(1),
			now.minusMonths(1)
		);
		log.debug("[getCompletionStats] result = {}", stats);
		return stats;
	}

	public List<MonthlyMaintenanceStatusDto> getMonthlyMaintenanceStatus(String regionName) {
		log.info("[getMonthlyMaintenanceStatus] 호출됨");
		List<MonthlyMaintenanceStatusDto> list = captureDamageRepository.getMonthlyMaintenanceStatsByPeriod(regionName);
		log.debug("[getMonthlyMaintenanceStatus] entries = {}", list.size());
		return list;
	}

	public List<RegionMaintenanceStatusDto> getRegionMaintenanceStatus(String regionName) {
		log.info("[getRegionMaintenanceStatus] 호출됨");
		List<RegionMaintenanceStatusDto> list = captureDamageRepository.getRegionMaintenanceStatsByPeriod(regionName);
		log.debug("[getRegionMaintenanceStatus] entries = {}", list.size());
		return list;
	}
}

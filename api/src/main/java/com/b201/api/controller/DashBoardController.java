package com.b201.api.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.dashboard.CategoryCountDto;
import com.b201.api.dto.dashboard.DailyStatusDto;
import com.b201.api.dto.dashboard.MonthlyStatusDto;
import com.b201.api.dto.dashboard.RegionCountDto;
import com.b201.api.dto.dashboard.TopRegionDto;
import com.b201.api.dto.dashboard.WeeklyStatusDto;
import com.b201.api.security.CustomUserDetails;
import com.b201.api.service.DashboardService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/dashboard")
@RequiredArgsConstructor
public class DashBoardController {

	private final DashboardService dashboardService;

	@GetMapping("/type")
	public ResponseEntity<List<CategoryCountDto>> getCategoryDistribution(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok(dashboardService.getCategoryDistribution(regionName));
	}

	/** 오늘자 파손 현황 */
	@GetMapping("/daily")
	public ResponseEntity<DailyStatusDto> getDaily(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok(dashboardService.getDailyStatusWithChangeRate(regionName));
	}

	/** 이번 주 파손 현황 */
	@GetMapping("/weekly")
	public ResponseEntity<WeeklyStatusDto> getWeekly(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok(dashboardService.getWeeklyStatusWithChangeRate(regionName));
	}

	/** 이번 달 파손 현황 */
	@GetMapping("/monthly")
	public ResponseEntity<MonthlyStatusDto> getMonthly(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok(dashboardService.getMonthlyStatusWithChangeRate(regionName));
	}

	@GetMapping("/monthly-summary")
	public ResponseEntity<?> getMonthlySummary() {
		return ResponseEntity.ok(dashboardService.getMonthlyDamageSummary());
	}

	@GetMapping("/districts")
	public ResponseEntity<List<RegionCountDto>> getDistricts(
		@RequestParam("city") String cityName
	) {
		List<RegionCountDto> list = dashboardService.getDistrictDistribution(cityName);
		return ResponseEntity.ok(list);
	}

	@GetMapping("/top3")
	public List<TopRegionDto> top3Regions() {
		return dashboardService.getTop3Regions();
	}
}

package com.b201.api.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.dashboard.CategoryCountDto;
import com.b201.api.dto.dashboard.DailyStatusDto;
import com.b201.api.dto.dashboard.DistinctRegionCountDto;
import com.b201.api.dto.dashboard.MonthlyStatusDto;
import com.b201.api.dto.dashboard.RegionNameWithCountDto;
import com.b201.api.dto.dashboard.RiskStatusDto;
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
	public ResponseEntity<?> getMonthlySummary(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok(dashboardService.getMonthlyDamageSummary(regionName));
	}

	@GetMapping("/districts")
	public ResponseEntity<RegionNameWithCountDto> getDistricts(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		RegionNameWithCountDto result = dashboardService.getDistrictDistribution(regionName);
		return ResponseEntity.ok(result);
	}

	@GetMapping("/top3")
	public List<TopRegionDto> top3Regions(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return dashboardService.getTop3Regions(regionName);
	}

	@GetMapping("/risk")
	public ResponseEntity<RiskStatusDto> getRisk(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok(dashboardService.getRiskStatus(regionName));
	}

	@GetMapping("/region-count")
	public ResponseEntity<DistinctRegionCountDto> getDistinctRegionCount(
		@AuthenticationPrincipal CustomUserDetails user
	) {
		String regionName = user.getRegionName();
		return ResponseEntity.ok(dashboardService.getDistinctRegionCount(regionName));
	}

}

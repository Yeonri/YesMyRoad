package com.b201.api.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.CategoryCountDto;
import com.b201.api.dto.DailyStatusDto;
import com.b201.api.dto.MonthlyStatusDto;
import com.b201.api.dto.WeeklyStatusDto;
import com.b201.api.service.DashboardService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/dashboard")
@RequiredArgsConstructor
public class DashBoardController {

	private final DashboardService dashboardService;

	@GetMapping("/type")
	public ResponseEntity<List<CategoryCountDto>> getCategoryDistribution() {
		return ResponseEntity.ok(dashboardService.getCategoryDistribution());
	}

	/** 오늘자 파손 현황 */
	@GetMapping("/daily")
	public ResponseEntity<DailyStatusDto> getDaily() {
		return ResponseEntity.ok(dashboardService.getDailyStatusWithChangeRate());
	}

	/** 이번 주 파손 현황 */
	@GetMapping("/weekly")
	public ResponseEntity<WeeklyStatusDto> getWeekly() {
		return ResponseEntity.ok(dashboardService.getWeeklyStatusWithChangeRate());
	}

	/** 이번 달 파손 현황 */
	@GetMapping("/monthly")
	public ResponseEntity<MonthlyStatusDto> getMonthly() {
		return ResponseEntity.ok(dashboardService.getMonthlyStatusWithChangeRate());
	}

	@GetMapping("/monthly-summary")
	public ResponseEntity<?> getMonthlySummary() {
		return ResponseEntity.ok(dashboardService.getMonthlyDamageSummary());
	}
}

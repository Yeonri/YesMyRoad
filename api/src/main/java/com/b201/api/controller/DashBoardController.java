package com.b201.api.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.CategoryCountDto;
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
}

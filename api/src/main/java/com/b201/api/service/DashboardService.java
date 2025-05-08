package com.b201.api.service;

import java.util.List;

import org.springframework.stereotype.Service;

import com.b201.api.dto.CategoryCountDto;
import com.b201.api.repository.CaptureDamageRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class DashboardService {

	private final CaptureDamageRepository captureDamageRepository;

	//유형별 도로 파손 분포 수
	public List<CategoryCountDto> getCategoryDistribution() {
		return captureDamageRepository.countByCategoryName();
	}

}

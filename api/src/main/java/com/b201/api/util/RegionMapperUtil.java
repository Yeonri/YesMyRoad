package com.b201.api.util;

import java.util.List;

import org.springframework.stereotype.Component;

import com.b201.api.domain.Region;
import com.b201.api.repository.RegionRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Component
@RequiredArgsConstructor
public class RegionMapperUtil {

	private final RegionRepository regionRepository;

	public Region mapAddressToRegion(String fullAddress) {
		List<Region> regions = regionRepository.findAll();
		return regions.stream()
			.filter(region -> fullAddress.startsWith(region.getRegionName()))
			.findFirst()
			.orElseThrow(() -> new IllegalArgumentException("해당 주소에 매칭되는 지역이 없습니다: " + fullAddress));
	}
}


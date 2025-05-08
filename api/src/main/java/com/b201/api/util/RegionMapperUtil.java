package com.b201.api.util;

import java.util.List;

import org.springframework.stereotype.Component;

import com.b201.api.domain.Region;
import com.b201.api.repository.RegionRepository;

import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class RegionMapperUtil {

	private final RegionRepository regionRepository;

	public Region mapAddressToRegion(String fullAddress) {
		// 1) 자식(구)부터 모두 로드
		List<Region> children = regionRepository.findByParentRegionIsNotNull();

		// 2) 주소에 포함된 구 이름이 있는지 탐색
		for (Region child : children) {
			if (fullAddress.contains(child.getRegionName())) {
				return child;
			}
		}

		// 3) 없으면 “시/도”(parent) 레벨로 매핑
		List<Region> parents = regionRepository.findByParentRegionIsNull();
		for (Region parent : parents) {
			if (fullAddress.startsWith(parent.getRegionName())) {
				return parent;
			}
		}

		throw new IllegalArgumentException("해당 주소에 매칭되는 지역이 없습니다: " + fullAddress);
	}
}


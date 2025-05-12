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
		// 1) “시/도”(parent) 먼저 찾기
		Region parent = regionRepository.findByParentRegionIsNull().stream()
			.filter(p -> fullAddress.startsWith(p.getRegionName()))
			.findFirst()
			.orElseThrow(() -> new IllegalArgumentException(
				"주소에서 시/도를 찾을 수 없습니다: " + fullAddress));

		// 2) 해당 시/도에 속한 구들만 로드해서 매칭
		List<Region> children = regionRepository.findByParentRegion(parent);
		for (Region child : children) {
			if (fullAddress.contains(child.getRegionName())) {
				return child;   // “서구”가 대전광역시의 서구라면 이걸 리턴
			}
		}

		// 3) 구가 안 붙어있으면 시/도 리턴
		return parent;
	}
}

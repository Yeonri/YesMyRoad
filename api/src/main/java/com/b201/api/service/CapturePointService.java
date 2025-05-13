package com.b201.api.service;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.domain.CapturePoint;
import com.b201.api.dto.AddressDto;
import com.b201.api.dto.damage.DamageDetailResponseDto;
import com.b201.api.dto.damage.DamageDto;
import com.b201.api.dto.point.CapturePointResponseDto;
import com.b201.api.dto.point.FeatureDto;
import com.b201.api.dto.point.GeometryDto;
import com.b201.api.dto.point.PropertiesDto;
import com.b201.api.repository.CapturePointRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class CapturePointService {

	private final CapturePointRepository capturePointRepository;

	public CapturePointResponseDto findAllFeatures() {
		log.info("[findAllFeatures] 호출됨");

		List<FeatureDto> featureDtos = capturePointRepository.findAll().stream()
			.map(this::mapToFeatureDto)
			.toList();

		log.debug("[findAllFeatures] 매핑된 feature 개수 = {}", featureDtos.size());

		return CapturePointResponseDto.builder()
			.features(featureDtos)   // features가 비어 있으면 []으로 직렬화됨
			.build();
	}

	private FeatureDto mapToFeatureDto(CapturePoint capturePoint) {
		double longitude = capturePoint.getLocation().getY();
		double latitude = capturePoint.getLocation().getX();

		GeometryDto geometryDto = GeometryDto.builder()
			.coordinates(new double[] {longitude, latitude})
			.build();

		PropertiesDto propertiesDto = PropertiesDto.builder()
			.publicId(capturePoint.getPublicId())
			.address(new AddressDto(capturePoint.getStreetAddress()))
			.accuracyMeters(capturePoint.getAccuracyMeters())
			.build();

		log.trace("[mapToFeatureDto] publicId={} 변환 완료", capturePoint.getPublicId());
		return FeatureDto.builder()
			.geometry(geometryDto)
			.properties(propertiesDto)
			.build();
	}

	public Optional<DamageDetailResponseDto> findDamageDetail(String publicId) {
		log.info("[findDamageDetail] 호출됨, publicId={}", publicId);

		Optional<DamageDetailResponseDto> result = capturePointRepository.findByPublicId(publicId)
			.map(cp -> DamageDetailResponseDto.builder()
				.risk(cp.getRisk())
				.imageUrl(cp.getImageUrl())
				.damages(cp.getCaptureDamages().stream()
					.map(this::mapToDamageDto)
					.toList())
				.build());

		result.ifPresentOrElse(
			dto -> log.debug("[findDamageDetail] 조회된 damage 개수 = {}", dto.getDamages().size()),
			() -> log.warn("[findDamageDetail] 해당 publicId의 데이터 없음: {}", publicId)
		);

		return result;
	}

	private DamageDto mapToDamageDto(CaptureDamage captureDamage) {
		DamageDto dto = DamageDto.builder()
			.id(captureDamage.getDamageId())
			.category(captureDamage.getDamageCategory().getCategoryName())
			.status(captureDamage.getStatus().getNumber())
			.updatedAt(captureDamage.getUpdatedAt())
			.build();

		log.trace("[mapToDamageDto] damageId={} 변환 완료", captureDamage.getDamageId());
		return dto;
	}
}

package com.b201.api.service;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.domain.CapturePoint;
import com.b201.api.dto.AddressDto;
import com.b201.api.dto.CapturePointResponseDto;
import com.b201.api.dto.DamageDetailResponseDto;
import com.b201.api.dto.DamageDto;
import com.b201.api.dto.FeatureDto;
import com.b201.api.dto.GeometryDto;
import com.b201.api.dto.PropertiesDto;
import com.b201.api.repository.CapturePointRepository;

import lombok.RequiredArgsConstructor;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class CapturePointService {

	private final CapturePointRepository capturePointRepository;

	public CapturePointResponseDto findAllFeatures() {
		List<FeatureDto> featureDtos = capturePointRepository.findAll().stream()
			.map(this::mapToFeatureDto)
			.toList();

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

		return FeatureDto.builder().geometry(geometryDto).properties(propertiesDto).build();
	}

	public Optional<DamageDetailResponseDto> findDamageDetail(String publicId) {
		return capturePointRepository.findByPublicId(publicId)
			.map(cp -> DamageDetailResponseDto.builder()
				.risk(cp.getRisk())
				.imageUrl(cp.getImageUrl())
				.damages(cp.getCaptureDamages().stream()
					.map(this::mapToDamageDto)
					.toList())
				.build());
	}

	private DamageDto mapToDamageDto(CaptureDamage captureDamage) {

		return DamageDto.builder()
			.category(captureDamage.getDamageCategory().getCategoryName())
			.status(captureDamage.getStatus().getNumber())
			.updatedAt(captureDamage.getUpdatedAt())
			.build();
	}
}

package com.b201.api.service;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.domain.CapturePoint;
import com.b201.api.dto.AddressDto;
import com.b201.api.dto.DamageDetailResponseDto;
import com.b201.api.dto.DamageDto;
import com.b201.api.dto.FeatureDto;
import com.b201.api.dto.GeometryDto;
import com.b201.api.dto.PropertiesDto;
import com.b201.api.exception.ResourceNotFoundException;
import com.b201.api.repository.CaptureDamageRepository;
import com.b201.api.repository.CapturePointRepository;

import lombok.RequiredArgsConstructor;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class CapturePointService {

	private final CapturePointRepository capturePointRepository;
	private final CaptureDamageRepository captureDamageRepository;

	public List<FeatureDto> findAllFeatures() {
		return capturePointRepository.findAll().stream()
			.map(this::mapToFeatureDto)
			.toList();
	}

	private FeatureDto mapToFeatureDto(CapturePoint capturePoint) {

		double longitude = capturePoint.getLocation().getX();
		double latitude = capturePoint.getLocation().getY();

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

	public DamageDetailResponseDto findDamageDetail(String publicId) {

		CapturePoint cp = capturePointRepository.findByPublicId(publicId)
			.orElseThrow(() -> new ResourceNotFoundException("CapturePoint", publicId));

		List<DamageDto> damages = cp.getCaptureDamages()
			.stream()
			.map(this::mapToDamageDto)
			.toList();

		return DamageDetailResponseDto.builder()
			.risk(cp.getRisk())
			.imageUrl(cp.getImageUrl())
			.damages(damages)
			.build();
	}

	private DamageDto mapToDamageDto(CaptureDamage captureDamage) {

		return DamageDto.builder()
			.category(captureDamage.getDamageCategory().getCategoryName())
			.status(captureDamage.getStatus().getNumber())
			.updatedAt(captureDamage.getUpdatedAt())
			.build();
	}
}

package com.b201.api.service;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.b201.api.domain.CapturePoint;
import com.b201.api.dto.AddressDto;
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

	public List<FeatureDto> findAllFeatures() {
		return capturePointRepository.findAll().stream()
			.map(this::toFeatureDto)
			.toList();
	}

	private FeatureDto toFeatureDto(CapturePoint capturePoint) {

		double longitude = capturePoint.getLocation().getX();
		double latitude = capturePoint.getLocation().getY();

		GeometryDto geometryDto = GeometryDto.builder()
			.coordinates(new double[] {longitude, latitude})
			.build();

		PropertiesDto propertiesDto = PropertiesDto.builder()
			.publicId(capturePoint.getPublicId())
			.address(new AddressDto(capturePoint.getProvinceRegion().getName(), capturePoint.getCityRegion().getName(),
				capturePoint.getDistrictRegion().getName(), capturePoint.getStreetAddress()))
			.accuracyMeters(capturePoint.getAccuracyMeters())
			.build();

		return FeatureDto.builder().geometry(geometryDto).properties(propertiesDto).build();
	}
}

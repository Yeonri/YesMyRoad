package com.b201.api.service;

import java.util.List;

import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.domain.CapturePoint;
import com.b201.api.domain.DamageCategory;
import com.b201.api.domain.Region;
import com.b201.api.dto.AiResultDto;
import com.b201.api.exception.AddressLookupException;
import com.b201.api.repository.CapturePointRepository;
import com.b201.api.repository.DamageCategoryRepository;
import com.b201.api.util.RegionMapperUtil;
import com.b201.api.util.VworldAddressUtil;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class AiResultService {

	private final VworldAddressUtil addressUtil;
	private final CapturePointRepository capturePointRepository;
	private final DamageCategoryRepository damageCategoryRepository;
	private final GeometryFactory geometryFactory;
	private final RegionMapperUtil regionMapperUtil;

	@Transactional
	public void addAiResult(AiResultDto dto) {
		//  주소 조회
		String street = findAddress(dto);
		Region region = regionMapperUtil.mapAddressToRegion(street);

		Point pt = toPoint(dto);

		CapturePoint capturePoint = toCapturePoint(dto, street, pt);
		// capturepoint에 관할 지역 설정
		capturePoint.setRegion(region);

		List<CaptureDamage> damages = toCaptureDamages(dto, capturePoint);

		damages.forEach(capturePoint::addDamage);

		capturePointRepository.save(capturePoint);
	}

	//Vworld api를 통해 좌표를 주소값으로 변환.
	private String findAddress(AiResultDto dto) {
		try {
			return addressUtil.changePointToAddress(
				dto.getLocation().getLongitude(),
				dto.getLocation().getLatitude()
			);
		} catch (RestClientException e) {
			throw new AddressLookupException(
				"주소변환 실패 : [" + dto.getLocation().getLongitude() + ", " + dto.getLocation().getLatitude() + "]", e);
		}
	}

	//dto의 있는 위도,경도를 point 객체 생성
	private Point toPoint(AiResultDto dto) {
		return geometryFactory.createPoint(
			new Coordinate(dto.getLocation().getLongitude(), dto.getLocation().getLatitude()));
	}

	//dto 내용과 주소, point 객체를 통해 CapturePoint 객체 생성
	private CapturePoint toCapturePoint(AiResultDto dto, String address, Point point) {
		return CapturePoint.builder()
			.accuracyMeters(dto.getLocation().getAccuracyMeters())
			.risk(dto.getImageInfo().getRisk())
			.captureTimestamp(dto.getCaptureTimestampUtc())
			.imageUrl(dto.getImageInfo().getImageUrl())
			.location(point)
			.streetAddress(address)
			.build();
	}

	//CaptureDamage 리스트 매핑
	private List<CaptureDamage> toCaptureDamages(AiResultDto dto, CapturePoint capturePoint) {
		return dto.getDetections().stream()
			.map(detection -> {
				DamageCategory category = damageCategoryRepository.findByCategoryName(detection.getCategoryName())
					.orElseGet(() -> damageCategoryRepository.save(new DamageCategory(
						detection.getCategoryName())));
				return CaptureDamage.builder()
					.capturePoint(capturePoint)
					.damageCategory(category)
					.build();
			})
			.toList();
	}
}

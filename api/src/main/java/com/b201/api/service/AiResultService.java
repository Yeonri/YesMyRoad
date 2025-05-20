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
		log.info("[addAiResult] 호출됨, dto = {}", dto);

		// 주소 조회
		String street = findAddress(dto);
		log.debug("[addAiResult] 주소 조회 완료 => {}", street);

		Region region = regionMapperUtil.mapAddressToRegion(street);
		log.debug("[addAiResult] Region 매핑 완료 => {}", region);

		// dto의 있는 위도,경도를 point 객체 생성
		Point pt = toPoint(dto);
		log.debug("[addAiResult] Point 변환 완료 => {}", pt);

		// dto 내용과 주소, point 객체를 통해 CapturePoint 객체 생성
		CapturePoint capturePoint = toCapturePoint(dto, street, pt);
		// capturepoint에 관할 지역 설정
		capturePoint.setRegion(region);
		log.debug("[addAiResult] CapturePoint 객체 생성 및 Region 설정 => {}", capturePoint);

		// CaptureDamage 리스트 매핑
		List<CaptureDamage> damages = toCaptureDamages(dto, capturePoint);
		log.debug("[addAiResult] CaptureDamage 리스트 생성, 개수 = {}", damages.size());

		damages.forEach(capturePoint::addDamage);

		CapturePoint saved = capturePointRepository.save(capturePoint);
		log.info("[addAiResult] CapturePoint 저장 완료, id = {}", saved.getCapturePointId());
	}

	// Vworld api를 통해 좌표를 주소값으로 변환.
	private String findAddress(AiResultDto dto) {
		try {
			return addressUtil.changePointToAddress(
				dto.getLocation().getLongitude(),
				dto.getLocation().getLatitude()
			);
		} catch (RestClientException e) {
			log.error("[findAddress] 주소 변환 실패, 좌표 = [{}, {}], error = {}",
				dto.getLocation().getLongitude(),
				dto.getLocation().getLatitude(),
				e.getMessage(), e);
			throw new AddressLookupException(
				"주소변환 실패: [" + dto.getLocation().getLongitude() + ", "
					+ dto.getLocation().getLatitude() + "]", e);
		}
	}

	// dto의 있는 위도,경도를 point 객체 생성
	private Point toPoint(AiResultDto dto) {
		Point p = geometryFactory.createPoint(
			new Coordinate(dto.getLocation().getLongitude(), dto.getLocation().getLatitude())
		);
		log.debug("[toPoint] 생성된 Point => {}", p);
		return p;
	}

	// dto 내용과 주소, point 객체를 통해 CapturePoint 객체 생성
	private CapturePoint toCapturePoint(AiResultDto dto, String address, Point point) {
		CapturePoint cp = CapturePoint.builder()
			.accuracyMeters(dto.getLocation().getAccuracyMeters())
			.risk(dto.getImageInfo().getRisk())
			.captureTimestamp(dto.getCaptureTimestampUtc())
			.imageUrl("https://k12b201.p.ssafy.io/images/" + dto.getImageInfo().getImageUrl() + ".jpg")
			.location(point)
			.streetAddress(address)
			.build();
		log.debug("[toCapturePoint] 생성된 CapturePoint => {}", cp);
		return cp;
	}

	// CaptureDamage 리스트 매핑
	private List<CaptureDamage> toCaptureDamages(AiResultDto dto, CapturePoint capturePoint) {
		List<CaptureDamage> list = dto.getDetections().stream()
			.map(detection -> {
				DamageCategory category = damageCategoryRepository
					.findByCategoryName(detection.getCategoryName())
					.orElseGet(() -> damageCategoryRepository.save(
						new DamageCategory(detection.getCategoryName())
					));
				return CaptureDamage.builder()
					.capturePoint(capturePoint)
					.damageCategory(category)
					.createdAt(capturePoint.getCaptureTimestamp())
					.build();
			})
			.toList();
		log.debug("[toCaptureDamages] 매핑된 CaptureDamage 개수 => {}", list.size());
		return list;
	}
}

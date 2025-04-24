package com.b201.api.service;

import java.util.List;
import java.util.stream.Collectors;

import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.domain.CapturePoint;
import com.b201.api.domain.DamageCategory;
import com.b201.api.dto.AiResultDto;
import com.b201.api.repository.CaptureDamageRepository;
import com.b201.api.repository.CapturePointRepository;
import com.b201.api.repository.DamageCategoryRepository;
import com.b201.api.util.VworldAddressUtil;

import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
public class AiResultService {

	private final VworldAddressUtil addressUtil;
	private final CapturePointRepository capturePointRepository;
	private final CaptureDamageRepository captureDamageRepository;
	private final DamageCategoryRepository damageCategoryRepository;

	public AiResultService(
		VworldAddressUtil addressUtil,
		CapturePointRepository capturePointRepository,
		CaptureDamageRepository captureDamageRepository,
		DamageCategoryRepository damageCategoryRepository
	) {
		this.addressUtil = addressUtil;
		this.capturePointRepository = capturePointRepository;
		this.captureDamageRepository = captureDamageRepository;
		this.damageCategoryRepository = damageCategoryRepository;
	}

	@Transactional
	public void addAiResult(AiResultDto dto) {
		// 1) 주소 조회
		String street = null;
		try {
			street = addressUtil.changePointToAddress(
				dto.getLocation().getLongitude(),
				dto.getLocation().getLatitude()
			);
		} catch (RestClientException e) {
			log.error("주소 변환 실패 for point={},{}",
				dto.getLocation().getLongitude(), dto.getLocation().getLatitude(), e);
			throw e;  // 필요에 따라 커스텀 예외로 래핑해도 좋습니다
		}

		GeometryFactory gf = new GeometryFactory();
		Point pt = gf.createPoint(new Coordinate(
			dto.getLocation().getLongitude(),
			dto.getLocation().getLatitude()
		));
		// 2) CapturePoint 저장
		CapturePoint cp = CapturePoint.builder()
			.accuracyMeters(dto.getLocation().getAccuracyMeters())
			.captureTimestamp(dto.getCaptureTimestampUtc())
			.risk(dto.getImageInfo().getRisk())
			.imageUrl(dto.getImageInfo().getImageUrl())
			.streetAddress(street)
			.location(pt).build();
		capturePointRepository.save(cp);

		// 3) Detection → CaptureDamage 배치 저장
		List<CaptureDamage> damages = dto.getDetections().stream()
			.map(d -> {
				// 카테고리 조회 또는 저장
				DamageCategory cat = damageCategoryRepository
					.findByCategoryName(d.getCategoryName())
					.orElseGet(() -> damageCategoryRepository
						.save(new DamageCategory(d.getCategoryName()))
					);
				return CaptureDamage.builder()
					.capturePoint(cp)
					.damageCategory(cat)
					.build();
			})
			.collect(Collectors.toList());
		captureDamageRepository.saveAll(damages);

		log.info("AI 결과 저장 완료: pointId={}, damages={}", cp.getCapturePointId(), damages.size());
	}
}

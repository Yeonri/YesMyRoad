package com.b201.api.service;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.domain.CaptureDamage.DamageStatus;
import com.b201.api.repository.CaptureDamageRepository;

import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class CaptureDamageService {

	private final CaptureDamageRepository damageRepo;

	/**
	 * 특정 파손 건(damageId)의 상태를 변경한다.
	 *
	 * @param damageId   변경할 파손 ID
	 * @param newStatus  새 상태(REPORTED, RECEIVED, IN_PROGRESS, COMPLETED 중 하나)
	 * @throws EntityNotFoundException 해당 ID의 파손 기록이 없으면 예외
	 */
	@Transactional
	public CaptureDamage changeStatus(Integer damageId, DamageStatus newStatus) {
		log.info("[changeStatus] 호출됨, damageId={}, newStatus={}", damageId, newStatus);

		CaptureDamage damage = damageRepo.findById(damageId)
			.orElseThrow(() -> {
				log.error("[changeStatus] 해당 ID의 파손 기록 없음: {}", damageId);
				return new EntityNotFoundException("Damage not found: " + damageId);
			});

		damage.setStatus(newStatus);
		log.debug("[changeStatus] 상태 업데이트 완료, damageId={}, status={}", damageId, newStatus);

		return damage;
	}
}

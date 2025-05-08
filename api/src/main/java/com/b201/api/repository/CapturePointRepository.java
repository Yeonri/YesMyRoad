package com.b201.api.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.CapturePoint;

@Repository
public interface CapturePointRepository extends JpaRepository<CapturePoint, Integer> {

	// ① Optional 반환으로 null 체크를 안전하게
	// ② EntityGraph로 연관된 CaptureDamage까지 한 번에 페치
	@EntityGraph(attributePaths = {"captureDamages", "captureDamages.damageCategory"})
	Optional<CapturePoint> findByPublicId(String publicId);
}


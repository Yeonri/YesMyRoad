package com.b201.api.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.CapturePoint;
import com.b201.api.dto.dashboard.DistinctRegionCountDto;

@Repository
public interface CapturePointRepository extends JpaRepository<CapturePoint, Integer> {

	@Query("""
		select p from CapturePoint p join p.region r join r.parentRegion pr where pr.regionName = :regionName
		""")
	List<CapturePoint> findAllByRegionName(@Param("regionName") String regionName);

	// ① Optional 반환으로 null 체크를 안전하게
	// ② EntityGraph로 연관된 CaptureDamage까지 한 번에 페치
	@EntityGraph(attributePaths = {"captureDamages", "captureDamages.damageCategory"})
	Optional<CapturePoint> findByPublicId(String publicId);

	@Query("""
		select new com.b201.api.dto.dashboard.DistinctRegionCountDto(
			count(distinct cp.streetAddress)
		)
		from CapturePoint cp
			where cp.region.parentRegion.regionName = :regionName
		""")
	DistinctRegionCountDto getDistinctRegionCountDto(@Param("regionName") String regionName);
}


package com.b201.api.repository;

import java.util.List;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.dto.CategoryCountDto;
import com.b201.api.dto.DailyCountDto;
import com.b201.api.dto.MonthlyDamageSummaryDto;
import com.b201.api.dto.RegionCountDto;
import com.b201.api.dto.TopRegionDto;

@Repository
public interface CaptureDamageRepository extends JpaRepository<CaptureDamage, Integer> {

	@Query("""
		  SELECT new com.b201.api.dto.CategoryCountDto(
		           d.damageCategory.categoryName,
		           COUNT(d)
		         )
		    FROM CaptureDamage d
			WHERE d.status <> com.b201.api.domain.CaptureDamage.DamageStatus.COMPLETED
		   GROUP BY d.damageCategory.categoryName
		""")
	List<CategoryCountDto> countByCategoryName();

	/**
	 * 일자별 파손 건수 집계
	 */
	@Query("""
		  SELECT new com.b201.api.dto.DailyCountDto(
		           DATE(d.capturePoint.captureTimestamp),
		           COUNT(d)
		         )
		    FROM CaptureDamage d
		   GROUP BY DATE(d.capturePoint.captureTimestamp)
		   ORDER BY DATE(d.capturePoint.captureTimestamp)
		""")
	List<DailyCountDto> countDaily();

	/**
	 * 월별 도로균열, 도로홀 분류 건수와 총합계
	 */
	@Query("""
		  SELECT new com.b201.api.dto.MonthlyDamageSummaryDto(
		      YEAR(d.capturePoint.captureTimestamp),
		          MONTH(d.capturePoint.captureTimestamp),
		              SUM(CASE WHEN d.damageCategory.categoryName = '도로균열' THEN 1 ELSE 0 END),
				              SUM(CASE WHEN d.damageCategory.categoryName = '도로홀'    THEN 1 ELSE 0 END),
		    COUNT(d)
		  )
		  FROM CaptureDamage d
		  GROUP BY YEAR(d.capturePoint.captureTimestamp),
		           MONTH(d.capturePoint.captureTimestamp)
		  ORDER BY YEAR(d.capturePoint.captureTimestamp),
		           MONTH(d.capturePoint.captureTimestamp)
		""")
	List<MonthlyDamageSummaryDto> findMonthlyDamageSummary();

	/**
	 * cityName을 받아
	 * 해당 광역시 이름으로 시작하는 regionName(구 단위)의 파손 건수를 집계
	 */
	@Query("""
		  SELECT new com.b201.api.dto.RegionCountDto(
		    r.regionName,
		    COUNT(cd.damageId)
		  )
		  FROM Region r
		  LEFT JOIN CapturePoint cp
		    ON cp.region = r
		  LEFT JOIN CaptureDamage cd
		    ON cd.capturePoint = cp
		   AND cd.status <> com.b201.api.domain.CaptureDamage.DamageStatus.COMPLETED
		  WHERE r.parentRegion.regionName = :parentName
		  GROUP BY r.regionName
		  ORDER BY r.regionName
		""")
	List<RegionCountDto> countByCity(
		@Param("parentName") String cityName
	);

	/**
	 * 상위 N개 지역별 도로파손 통계 (완료된 건까지 모두 집계)
	 */
	@Query("""
		  SELECT new com.b201.api.dto.TopRegionDto(
		    r.regionName,
		    COUNT(cd), 
		    SUM(CASE WHEN cd.damageCategory.categoryName = '도로균열' THEN 1 ELSE 0 END),
		    SUM(CASE WHEN cd.damageCategory.categoryName = '도로홀'  THEN 1 ELSE 0 END)
		  )
		  FROM CaptureDamage cd
		  JOIN cd.capturePoint cp
		  JOIN cp.region r
		  GROUP BY r.regionName
		  ORDER BY COUNT(cd) DESC
		""")
	List<TopRegionDto> findTopRegions(Pageable pageable);
}

package com.b201.api.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.dto.CategoryCountDto;
import com.b201.api.dto.DailyCountDto;
import com.b201.api.dto.MonthlyDamageSummaryDto;

@Repository
public interface CaptureDamageRepository extends JpaRepository<CaptureDamage, Integer> {

	@Query("""
		  SELECT new com.b201.api.dto.CategoryCountDto(
		           d.damageCategory.categoryName,
		           COUNT(d)
		         )
		    FROM CaptureDamage d
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

}

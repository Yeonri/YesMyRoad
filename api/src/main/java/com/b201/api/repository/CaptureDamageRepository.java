package com.b201.api.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.dto.CategoryCountDto;
import com.b201.api.dto.DailyCountDto;

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

}

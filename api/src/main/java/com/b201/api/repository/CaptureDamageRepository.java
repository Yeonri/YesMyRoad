package com.b201.api.repository;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.CaptureDamage;
import com.b201.api.dto.dashboard.CategoryCountDto;
import com.b201.api.dto.dashboard.DailyCountDto;
import com.b201.api.dto.dashboard.MonthlyDamageSummaryDto;
import com.b201.api.dto.dashboard.RegionCountDto;
import com.b201.api.dto.dashboard.TopRegionDto;
import com.b201.api.dto.maintenance.CompletionStatsDto;
import com.b201.api.dto.maintenance.MaintenanceStatusDto;
import com.b201.api.dto.maintenance.MonthlyMaintenanceStatusDto;
import com.b201.api.dto.maintenance.RegionMaintenanceStatusDto;

@Repository
public interface CaptureDamageRepository extends JpaRepository<CaptureDamage, Integer> {

	// 카테고리별 도로 손상 집계 현황
	@Query("""
		  SELECT new com.b201.api.dto.dashboard.CategoryCountDto(
		           d.damageCategory.categoryName,
		           COUNT(d)
		         )
		    FROM CaptureDamage d
				    join d.capturePoint.region r
						    join r.parentRegion pr
								    where d.status = 'COMPLETED'
										    and pr.regionName = :regionName
		   GROUP BY d.damageCategory.categoryName
		""")
	List<CategoryCountDto> countByCategoryName(@Param("regionName") String regionName);

	// 일자별 파손 집계
	@Query("""
		  SELECT new com.b201.api.dto.dashboard.DailyCountDto(
		           DATE(d.capturePoint.captureTimestamp),
		           COUNT(d)
		         )
		    FROM CaptureDamage d
				    join d.capturePoint.region r
						    join r.parentRegion pr
								    where pr.regionName = :regionName
		   GROUP BY DATE(d.capturePoint.captureTimestamp)
		   ORDER BY DATE(d.capturePoint.captureTimestamp)
		""")
	List<DailyCountDto> countDaily(@Param("regionName") String regionName);

	// 월별 도로균열, 도로홀 분류 건수와 총합계
	@Query("""
		  SELECT new com.b201.api.dto.dashboard.MonthlyDamageSummaryDto(
		      YEAR(d.capturePoint.captureTimestamp),
		          MONTH(d.capturePoint.captureTimestamp),
		              SUM(CASE WHEN d.damageCategory.categoryName = '도로균열' THEN 1 ELSE 0 END),
				              SUM(CASE WHEN d.damageCategory.categoryName = '도로홀'    THEN 1 ELSE 0 END),
		    COUNT(d)
		  )
		  FROM CaptureDamage d
				  join d.capturePoint.region r
						  join r.parentRegion pr
								  where pr.regionName = :regionName
		  GROUP BY YEAR(d.capturePoint.captureTimestamp),
		           MONTH(d.capturePoint.captureTimestamp)
		  ORDER BY YEAR(d.capturePoint.captureTimestamp),
		           MONTH(d.capturePoint.captureTimestamp)
		""")
	List<MonthlyDamageSummaryDto> findMonthlyDamageSummary(@Param("regionName") String regionName);

	/**
	 * cityName을 받아
	 * 해당 광역시 이름으로 시작하는 regionName(구 단위)의 파손 건수를 집계
	 */
	@Query("""
		  SELECT new com.b201.api.dto.dashboard.RegionCountDto(
		    r.regionName,
		    COUNT(cd.damageId)
		  )
		  FROM Region r
				  left join r.capturePoints cp
						  left join cp.captureDamages cd
								  left join cd.damageCategory dc
										  where r.parentRegion.regionName = :regionName
		  GROUP BY r.regionName
		""")
	List<RegionCountDto> countByCity(
		@Param("regionName") String cityName
	);

	/**
	 * 상위 N개 지역별 도로파손 통계 (완료된 건까지 모두 집계)
	 */
	@Query("""
		  SELECT new com.b201.api.dto.dashboard.TopRegionDto(
		    r.regionName,
				    count(cd),
						    coalesce(sum(case when cd.damageCategory.categoryName = '도로균열' then 1 ELSE 0 END),0),
								    coalesce(sum(case when cd.damageCategory.categoryName = '도로홀' then 1 ELSE 0 END),0)
		  )
		  from Region r
				  join r.parentRegion pr
						  left join r.capturePoints cp
								  left join cp.captureDamages cd
										  left join cd.damageCategory dc
										  where pr.regionName = :regionName
		  GROUP BY r.regionName
		  ORDER BY COUNT(cd) DESC
		""")
	List<TopRegionDto> findTopRegions(@Param("regionName") String regionName, Pageable pageable);

	@Query("""
			select new com.b201.api.dto.maintenance.MaintenanceStatusDto(
					sum(case when cd.status = 'REPORTED' then 1 else 0 end),
							sum(case when cd.status = 'RECEIVED' then 1 else 0 end),
									sum(case when cd.status = 'IN_PROGRESS' then 1 else 0 end),
											sum(case when cd.status = 'COMPLETED' then 1 else 0 end)
			)
			from CaptureDamage cd
		""")
	MaintenanceStatusDto countByStauts();

	@Query("""
			select new com.b201.api.dto.maintenance.CompletionStatsDto(
					coalesce(sum(case when cd.updatedAt >= :dailyAgo then 1 else 0 end), 0),
							coalesce(sum(case when cd.updatedAt >= :weeklyAgo then 1 else 0 end), 0),
									coalesce(sum(case when cd.updatedAt >= :monthlyAgo then 1 else 0 end), 0)
			)
					from CaptureDamage cd
							where cd.status = 'COMPLETED'
		""")
	CompletionStatsDto getCompletionStatsByPeriod(@Param("dailyAgo") LocalDateTime dailyAgo,
		@Param("weeklyAgo") LocalDateTime weeklyAgo,
		@Param("monthlyAgo") LocalDateTime monthlyAgo);

	@Query("""
		select new com.b201.api.dto.maintenance.MonthlyMaintenanceStatusDto(
				year(cd.createdAt),
						month(cd.createdAt),
								sum(case when cd.status = 'REPORTED' then 1 else 0 end),
										sum(case when cd.status = 'RECEIVED' then 1 else 0 end),
												sum(case when cd.status = 'IN_PROGRESS' then 1 else 0 end),
														sum(case when cd.status = 'COMPLETED' then 1 else 0 end)
		)
				from CaptureDamage cd
						group by year(cd.createdAt),
								month(cd.createdAt)
										order by year(cd.createdAt),
												month(cd.createdAt)
		""")
	List<MonthlyMaintenanceStatusDto> getMonthlyMaintenanceStatsByPeriod();

	@Query("""
				select new com.b201.api.dto.maintenance.RegionMaintenanceStatusDto(
					r.regionName,
							new com.b201.api.dto.maintenance.MaintenanceStatusDto(
									coalesce(sum(case when cd.status = 'REPORTED' then 1 else 0 end),0),
											coalesce(sum(case when cd.status = 'RECEIVED' then 1 else 0 end),0),
													coalesce(sum(case when cd.status = 'IN_PROGRESS' then 1 else 0 end),0),
															coalesce(sum(case when cd.status = 'COMPLETED' then 1 else 0 end),0))
				)
				from Region r
						left join CapturePoint cp
								on cp.region = r
										left join CaptureDamage cd
												on cd.capturePoint = cp
														where r.parentRegion.regionName = :city
																group by r.regionName
		""")
	List<RegionMaintenanceStatusDto> getRegionMaintenanceStatsByPeriod(@Param("city") String city);
}

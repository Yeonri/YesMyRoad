package com.b201.api.service;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.YearMonth;
import java.util.List;

import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

import com.b201.api.dto.dashboard.CategoryCountDto;
import com.b201.api.dto.dashboard.DailyCountDto;
import com.b201.api.dto.dashboard.DailyStatusDto;
import com.b201.api.dto.dashboard.MonthlyDamageSummaryDto;
import com.b201.api.dto.dashboard.MonthlyStatusDto;
import com.b201.api.dto.dashboard.RegionCountDto;
import com.b201.api.dto.dashboard.TopRegionDto;
import com.b201.api.dto.dashboard.WeeklyStatusDto;
import com.b201.api.repository.CaptureDamageRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class DashboardService {

	private final CaptureDamageRepository damageRepo;

	// 유형별 도로 파손 분포 수
	public List<CategoryCountDto> getCategoryDistribution(String regionName) {
		log.info("[getCategoryDistribution] 호출됨, regionName={}", regionName);
		List<CategoryCountDto> list = damageRepo.countByCategoryName(regionName);
		log.debug("[getCategoryDistribution] 분류 개수={}", list.size());
		return list;
	}

	// 오늘자 파손 건수 + 전일 대비 증감율
	public DailyStatusDto getDailyStatusWithChangeRate(String regionName) {
		log.info("[getDailyStatusWithChangeRate] 호출됨");
		LocalDate today = LocalDate.now();
		LocalDate yesterday = today.minusDays(1);

		List<DailyCountDto> raw = damageRepo.countDaily(regionName);

		long todayCount = raw.stream()
			.filter(d -> d.getDay().equals(today))
			.mapToLong(DailyCountDto::getCount)
			.findFirst().orElse(0L);
		long yesterdayCount = raw.stream()
			.filter(d -> d.getDay().equals(yesterday))
			.mapToLong(DailyCountDto::getCount)
			.findFirst().orElse(0L);

		double rate = 0.0;
		if (yesterdayCount > 0) {
			rate = (todayCount - yesterdayCount) * 100.0 / yesterdayCount;
		}

		log.debug("[getDailyStatusWithChangeRate] todayCount={}, yesterdayCount={}, rate={}",
			todayCount, yesterdayCount, rate);
		return new DailyStatusDto(today, todayCount, rate);
	}

	// 이번 주(월요일~) 파손 건수 합계 + 전주 대비 증감율
	public WeeklyStatusDto getWeeklyStatusWithChangeRate(String regionName) {
		log.info("[getWeeklyStatusWithChangeRate] 호출됨");
		LocalDate today = LocalDate.now();
		LocalDate thisMon = today.with(DayOfWeek.MONDAY);
		LocalDate lastMon = thisMon.minusWeeks(1);

		List<DailyCountDto> raw = damageRepo.countDaily(regionName);

		long thisWeekSum = raw.stream()
			.filter(d -> !d.getDay().isBefore(thisMon) && !d.getDay().isAfter(today))
			.mapToLong(DailyCountDto::getCount)
			.sum();
		long lastWeekSum = raw.stream()
			.filter(d -> !d.getDay().isBefore(lastMon) && d.getDay().isBefore(thisMon))
			.mapToLong(DailyCountDto::getCount)
			.sum();

		double rate = 0.0;
		if (lastWeekSum > 0) {
			rate = (thisWeekSum - lastWeekSum) * 100.0 / lastWeekSum;
		}

		log.debug("[getWeeklyStatusWithChangeRate] thisWeekSum={}, lastWeekSum={}, rate={}",
			thisWeekSum, lastWeekSum, rate);
		return new WeeklyStatusDto(thisMon, thisWeekSum, rate);
	}

	// 이번 달 파손 건수 합계 + 전월 대비 증감율
	public MonthlyStatusDto getMonthlyStatusWithChangeRate(String regionName) {
		log.info("[getMonthlyStatusWithChangeRate] 호출됨");
		LocalDate today = LocalDate.now();
		YearMonth thisYm = YearMonth.from(today);
		YearMonth lastYm = thisYm.minusMonths(1);

		List<DailyCountDto> raw = damageRepo.countDaily(regionName);

		long thisMonthSum = raw.stream()
			.filter(d -> YearMonth.from(d.getDay()).equals(thisYm))
			.mapToLong(DailyCountDto::getCount)
			.sum();
		long lastMonthSum = raw.stream()
			.filter(d -> YearMonth.from(d.getDay()).equals(lastYm))
			.mapToLong(DailyCountDto::getCount)
			.sum();

		double rate = 0.0;
		if (lastMonthSum > 0) {
			rate = (thisMonthSum - lastMonthSum) * 100.0 / lastMonthSum;
		}

		log.debug("[getMonthlyStatusWithChangeRate] thisMonthSum={}, lastMonthSum={}, rate={}",
			thisMonthSum, lastMonthSum, rate);
		return new MonthlyStatusDto(thisYm, thisMonthSum, rate);
	}

	/**
	 * 월별 도로파손 누적 탐지 현황
	 */
	public List<MonthlyDamageSummaryDto> getMonthlyDamageSummary(String regionName) {
		log.info("[getMonthlyDamageSummary] 호출됨");
		List<MonthlyDamageSummaryDto> list = damageRepo.findMonthlyDamageSummary(regionName);
		log.debug("[getMonthlyDamageSummary] summary 개수 = {}", list.size());
		return list;
	}

	/**
	 * 특정 광역시의 구 단위 파손 분포
	 */
	public List<RegionCountDto> getDistrictDistribution(String cityName) {
		log.info("[getDistrictDistribution] 호출됨, cityName={}", cityName);
		List<RegionCountDto> list = damageRepo.countByCity(cityName);
		log.debug("[getDistrictDistribution] 구 단위 분포 개수 = {}", list.size());
		return list;
	}

	/**
	 * 상위 3개 지역의 도로파손 통계
	 */
	public List<TopRegionDto> getTop3Regions(String regionName) {
		log.info("[getTop3Regions] 호출됨");
		List<TopRegionDto> list = damageRepo.findTopRegions(regionName, PageRequest.of(0, 3));
		log.debug("[getTop3Regions] top regions 개수 = {}", list.size());
		return list;
	}
}

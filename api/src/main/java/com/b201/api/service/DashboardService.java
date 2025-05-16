package com.b201.api.service;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.YearMonth;
import java.util.List;

import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.b201.api.dto.dashboard.CategoryCountDto;
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
	@Transactional(readOnly = true)
	public List<CategoryCountDto> getCategoryDistribution(String regionName) {
		log.info("[getCategoryDistribution] 호출됨, regionName={}", regionName);
		List<CategoryCountDto> list = damageRepo.countByCategoryName(regionName);
		log.debug("[getCategoryDistribution] 분류 개수={}", list.size());
		return list;
	}

	private double calculateRate(long current, long previous) {
		if (previous > 0) {
			return (current - previous) * 100.0 / previous;
		}
		if (current == previous) {
			return 0.0;
		}
		throw new ArithmeticException("기준 값(이전 기간) 데이터가 없습니다.");
	}


	// 오늘자 파손 건수 + 전일 대비 증감율
	@Transactional(readOnly = true)
	public DailyStatusDto getDailyStatusWithChangeRate(String regionName) {
		log.info("[getDailyStatusWithChangeRate] 호출됨, regionName={}", regionName);

		LocalDate today = LocalDate.now();
		LocalDate yesterday = today.minusDays(1);

		// 1) 오늘 0시부터 지금까지
		LocalDateTime todayStart = today.atStartOfDay();
		LocalDateTime now = LocalDateTime.now();
		long todayCount = damageRepo.countBetween(
			regionName,
			todayStart,
			now
		);

		// 2) 어제 0시부터 오늘 0시까지
		LocalDateTime yesterdayStart = yesterday.atStartOfDay();
		long yesterdayCount = damageRepo.countBetween(
			regionName,
			yesterdayStart,
			todayStart
		);

		// 3) 증감율 계산
		double rate = calculateRate(todayCount, yesterdayCount);

		log.debug("[getDailyStatusWithChangeRate] todayCount={}, yesterdayCount={}, rate={}",
			todayCount, yesterdayCount, rate);
		return new DailyStatusDto(today, todayCount, rate);
	}

	// 이번 주(월요일~) 파손 건수 합계 + 전주 대비 증감율
	@Transactional(readOnly = true)
	public WeeklyStatusDto getWeeklyStatusWithChangeRate(String regionName) {
		log.info("[getWeeklyStatusWithChangeRate] 호출됨, regionName={}", regionName);
		LocalDate today = LocalDate.now();
		LocalDate thisMon = today.with(DayOfWeek.MONDAY);
		LocalDate lastMon = thisMon.minusWeeks(1);

		long thisWeekSum = damageRepo.countBetween(regionName, thisMon.atStartOfDay(), LocalDateTime.now());

		long lastWeekSum = damageRepo.countBetween(regionName, lastMon.atStartOfDay(), thisMon.atStartOfDay());

		double rate = calculateRate(thisWeekSum, lastWeekSum);

		log.debug("[getWeeklyStatusWithChangeRate] thisWeekSum={}, lastWeekSum={}, rate={}",
			thisWeekSum, lastWeekSum, rate);
		return new WeeklyStatusDto(thisMon, thisWeekSum, rate);
	}

	// 이번 달 파손 건수 합계 + 전월 대비 증감율
	@Transactional(readOnly = true)
	public MonthlyStatusDto getMonthlyStatusWithChangeRate(String regionName) {
		log.info("[getMonthlyStatusWithChangeRate] 호출됨, regionName={}", regionName);

		LocalDate today = LocalDate.now();
		LocalDate firstDayThisMon = today.withDayOfMonth(1);
		LocalDate firstDayLastMon = firstDayThisMon.minusMonths(1);

		// 이번 달: 1일 0시부터 지금까지
		LocalDateTime thisStart = firstDayThisMon.atStartOfDay();
		LocalDateTime now = LocalDateTime.now();
		long thisMonthCount = damageRepo.countBetween(
			regionName,
			thisStart,
			now
		);

		// 지난 달: 지난달 1일 0시부터 이번 달 1일 0시까지
		LocalDateTime lastStart = firstDayLastMon.atStartOfDay();
		LocalDateTime lastEnd = firstDayThisMon.atStartOfDay();
		long lastMonthCount = damageRepo.countBetween(
			regionName,
			lastStart,
			lastEnd
		);

		// 증감율 계산
		double rate = calculateRate(thisMonthCount, lastMonthCount);

		log.debug("[getMonthlyStatusWithChangeRate] thisMonthCount={}, lastMonthCount={}, rate={}",
			thisMonthCount, lastMonthCount, rate);

		return new MonthlyStatusDto(YearMonth.from(firstDayThisMon), thisMonthCount, rate);
	}

	/**
	 * 월별 도로파손 누적 탐지 현황
	 */
	@Transactional(readOnly = true)
	public List<MonthlyDamageSummaryDto> getMonthlyDamageSummary(String regionName) {
		log.info("[getMonthlyDamageSummary] 호출됨, regionName={}", regionName);
		List<MonthlyDamageSummaryDto> list = damageRepo.findMonthlyDamageSummary(regionName);
		log.debug("[getMonthlyDamageSummary] summary 개수 = {}", list.size());
		return list;
	}

	/**
	 * 특정 광역시의 구 단위 파손 분포
	 */
	@Transactional(readOnly = true)
	public List<RegionCountDto> getDistrictDistribution(String cityName) {
		log.info("[getDistrictDistribution] 호출됨, regionName={}", cityName);
		List<RegionCountDto> list = damageRepo.countByCity(cityName);
		log.debug("[getDistrictDistribution] 구 단위 분포 개수 = {}", list.size());
		return list;
	}

	/**
	 * 상위 3개 지역의 도로파손 통계
	 */
	@Transactional(readOnly = true)
	public List<TopRegionDto> getTop3Regions(String regionName) {
		log.info("[getTop3Regions] 호출됨, regionName={}", regionName);
		List<TopRegionDto> list = damageRepo.findTopRegions(regionName, PageRequest.of(0, 3));
		log.debug("[getTop3Regions] top regions 개수 = {}", list.size());
		return list;
	}
}

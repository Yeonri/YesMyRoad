package com.b201.api.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.Region;

@Repository
public interface RegionRepository extends JpaRepository<Region, Integer> {
	List<Region> findByParentRegionIsNull();

	List<Region> findByParentRegionIsNotNull();
}

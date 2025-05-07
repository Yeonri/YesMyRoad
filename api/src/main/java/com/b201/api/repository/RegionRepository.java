package com.b201.api.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.b201.api.domain.Region;

public interface RegionRepository extends JpaRepository<Region, Integer> {
	Optional<Region> findByRegionName(String name);
}

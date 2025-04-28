package com.b201.api.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.b201.api.domain.DamageCategory;

public interface DamageCategoryRepository extends JpaRepository<DamageCategory, Integer> {

	Optional<DamageCategory> findByCategoryName(String categoryName);

	boolean existsByCategoryName(String categoryName);
}

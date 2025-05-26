package com.b201.api.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.DamageCategory;

@Repository
public interface DamageCategoryRepository extends JpaRepository<DamageCategory, Integer> {

	Optional<DamageCategory> findByCategoryName(String categoryName);

	boolean existsByCategoryName(String categoryName);
}

package com.b201.api.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.CaptureDamage;

@Repository
public interface CaptureDamageRepository extends JpaRepository<CaptureDamage, Integer> {
}

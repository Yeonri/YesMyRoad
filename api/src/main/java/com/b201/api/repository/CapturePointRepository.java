package com.b201.api.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.b201.api.domain.CapturePoint;

public interface CapturePointRepository extends JpaRepository<CapturePoint, Integer> {

}

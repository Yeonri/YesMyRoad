package com.b201.api.domain;

import java.time.LocalDateTime;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import org.locationtech.jts.geom.Point;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 지도상의 캡처 포인트 엔티티 (위치 정보를 POINT 타입으로 저장)
 */
@Entity
@Getter
@NoArgsConstructor
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
@Table(name = "capture_point")
public class CapturePoint {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@EqualsAndHashCode.Include
	@Column(name = "capture_point_id", nullable = false, unique = true)
	private Integer capturePointId;

	@Column(name = "capture_timestamp", nullable = false)
	private LocalDateTime captureTimestamp;

	@JdbcTypeCode(SqlTypes.GEOMETRY)
	@Column(name = "location", nullable = false)
	private Point location;

	@Column(name = "accuracy_meters")
	private Double accuracyMeters;

	@Column(name = "image_url")
	private String imageUrl;

	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "province_id", nullable = false)
	private Region provinceRegion;

	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "city_id", nullable = false)
	private Region cityRegion;

	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "district_id")
	private Region districtRegion;

	@Column(name = "street_address")
	private String streetAddress;

	@Builder(toBuilder = true)
	public CapturePoint(
		LocalDateTime captureTimestamp,
		Point location,
		Double accuracyMeters,
		String imageUrl,
		Region provinceRegion,
		Region cityRegion,
		Region districtRegion,
		String streetAddress
	) {
		this.captureTimestamp = captureTimestamp;
		this.location = location;
		this.accuracyMeters = accuracyMeters;
		this.imageUrl = imageUrl;
		this.provinceRegion = provinceRegion;
		this.cityRegion = cityRegion;
		this.districtRegion = districtRegion;
		this.streetAddress = streetAddress;
	}

}

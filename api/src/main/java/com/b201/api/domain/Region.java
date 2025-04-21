package com.b201.api.domain;

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
 * 행정구역(Region) 엔티티
 */
@Entity
@Getter
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
@NoArgsConstructor
@Table(name = "region")
public class Region {

	/**
	 * 고유 식별자
	 */
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@EqualsAndHashCode.Include
	@Column(name = "region_id", nullable = false, unique = true)
	private Integer regionId;

	/**
	 * 상위 행정구역 (1단계 상위), 도/광역시는 null
	 */
	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "parent_id")
	private Region parent;

	/**
	 * 계층 레벨 (1=도/광역시, 2=시/군/구, 3=읍/면/동)
	 */
	@Column(name = "level", nullable = false)
	private Integer level;

	/**
	 * 행정구역명
	 */
	@Column(name = "name", nullable = false, length = 100)
	private String name;

	/**
	 * Builder 생성자 (ID 제외)
	 * @param parent 상위 행정구역 엔티티
	 * @param level  계층 레벨 (1=도/광역시, 2=시/군/구, 3=읍/면/동)
	 * @param name   행정구역명
	 */
	@Builder
	public Region(Region parent, Integer level, String name) {
		this.parent = parent;
		this.level = level;
		this.name = name;
	}
}

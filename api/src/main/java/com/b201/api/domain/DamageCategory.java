package com.b201.api.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
@Table(name = "damage_category")
public class DamageCategory {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@EqualsAndHashCode.Include
	@Column(unique = true, nullable = false, name = "category_id")
	private Integer categoryId;

	@Column(unique = true, nullable = false, name = "category_name")
	private String categoryName;

	/**
	 * builder 패턴 생성자
	 * @param categoryName 파손 유형 이름
	 */
	@Builder
	public DamageCategory(String categoryName) {
		this.categoryName = categoryName;
	}
}

package com.b201.api.domain;

import java.util.List;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@NoArgsConstructor
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
@Getter
@Table(name = "region")
public class Region {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@EqualsAndHashCode.Include
	@Column(name = "region_id")
	private Integer id;

	@Column(name = "region_name", nullable = false)
	private String regionName;

	//상위 광역시/도가 없는 경우(null), 있으면 부모 Region
	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "parent_region_id")
	private Region parentRegion;

	@OneToMany(mappedBy = "region", fetch = FetchType.LAZY)
	private List<CapturePoint> capturePoints;
}
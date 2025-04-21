package com.b201.api.domain;

import java.time.LocalDateTime;

import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import com.fasterxml.jackson.annotation.JsonFormat;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EntityListeners;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@EntityListeners(AuditingEntityListener.class)
@NoArgsConstructor
@Getter
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
@Table(name = "capture_damage")
public class CaptureDamage {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@EqualsAndHashCode.Include
	@Column(unique = true, nullable = false, name = "damage_id")
	private Integer damageId;

	@ManyToOne
	@JoinColumn(name = "capture_point_id", nullable = false)
	private CapturePoint capturePoint;

	@ManyToOne
	@JoinColumn(name = "category_id", nullable = false)
	private DamageCategory damageCategory;

	@Enumerated(EnumType.STRING)
	@JsonFormat(shape = JsonFormat.Shape.NUMBER)
	@Column(name = "status", nullable = false, length = 20)
	private DamageStatus status;

	@Column(name = "description")
	private String description;

	@Enumerated(EnumType.STRING)
	@JsonFormat(shape = JsonFormat.Shape.NUMBER)
	@Column(name = "severity", length = 10)
	private Severity severity;

	@CreatedDate
	@Column(name = "created_at", nullable = false, updatable = false)
	private LocalDateTime createdAt;

	@LastModifiedDate
	@Column(name = "updated_at")
	private LocalDateTime updatedAt;

	public enum DamageStatus {
		REPORTED,     // 보고됨
		RECEIVED,     // 수리접수
		IN_PROGRESS,  // 수리중
		COMPLETED     // 수리완료
	}

	public enum Severity {
		SAFE,    // 안전
		WARNING, // 주의
		DANGER   // 위험
	}

	@PrePersist
	public void prePersist() {
		if (this.status == null) {
			this.status = DamageStatus.REPORTED;
		}
	}

	/**
	 * 자동생성 PK_ID 를 제외한 빌더 패턴 생성자
	 * @param capturePoint         이미지 정보
	 * @param damageCategory       파손 유형
	 * @param status               수리 상태
	 * @param description          설명
	 * @param severity             심각도(optional)
	 */
	@Builder
	public CaptureDamage(
		CapturePoint capturePoint,
		DamageCategory damageCategory,
		DamageStatus status,
		String description,
		Severity severity
	) {
		this.capturePoint = capturePoint;
		this.damageCategory = damageCategory;
		this.status = status;
		this.description = description;
		this.severity = severity;
	}

}

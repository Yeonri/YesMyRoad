// com.b201.api.controller.CaptureDamageController.java
package com.b201.api.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.domain.CaptureDamage.DamageStatus;
import com.b201.api.dto.damage.StatusUpdateRequest;
import com.b201.api.service.CaptureDamageService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/damages")
@RequiredArgsConstructor
public class CaptureDamageController {

	private final CaptureDamageService damageService;

	/**
	 * 요청 바디로 받은 새 상태로 해당 파손 건의 상태를 변경합니다.
	 */
	@PatchMapping("/{damageId}")
	public ResponseEntity<Void> updateStatus(
		@PathVariable("damageId") Integer id,
		@RequestBody StatusUpdateRequest req
	) {
		damageService.changeStatus(id, DamageStatus.valueOf(req.getStatus()));
		return ResponseEntity.ok().build();    // 200 OK, 빈 바디
	}
}

// com.b201.api.controller.CaptureDamageController.java
package com.b201.api.controller;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.domain.CaptureDamage.DamageStatus;
import com.b201.api.dto.StatusUpdateRequest;
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
	@PatchMapping("/status/{id}")
	@ResponseStatus(HttpStatus.NO_CONTENT)
	public void updateStatus(
		@PathVariable("id") Integer damageId,
		@RequestBody StatusUpdateRequest req
	) {
		// req.getStatus() 는 "COMPLETED" 같은 enum 이름
		damageService.changeStatus(damageId, DamageStatus.valueOf(req.getStatus()));
	}
}

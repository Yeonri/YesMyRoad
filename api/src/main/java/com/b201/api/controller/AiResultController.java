package com.b201.api.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.AiResultDto;
import com.b201.api.service.AiResultService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/detect")
public class AiResultController {

	private final AiResultService aiResultService;

	@PostMapping
	public ResponseEntity<?> addPoint(@RequestBody AiResultDto aiResultDto) {
		aiResultService.addAiResult(aiResultDto);
		return new ResponseEntity<>(HttpStatus.OK);
	}
}

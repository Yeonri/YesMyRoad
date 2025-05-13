package com.b201.api.util;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import com.fasterxml.jackson.databind.JsonNode;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class VworldAddressUtil {

	private final String apiKey;
	private final RestClient addressRestClient;

	public VworldAddressUtil(@Value("${address.key}") String apiKey, RestClient addressRestClient) {
		this.addressRestClient = addressRestClient;
		this.apiKey = apiKey;
	}

	/**
	 * 위도(lat), 경도(lng)를 받아 도로명주소 변환 API 호출 후
	 * JSON 문자열을 리턴합니다.
	 */
	public String changePointToAddress(double lng, double lat) {
		log.info("[changePointToAddress] 호출됨, longitude={}, latitude={}", lng, lat);
		String point = lng + "," + lat;

		JsonNode root = addressRestClient.get()
			.uri(uriBuilder -> uriBuilder
				.path("/req/address")
				.queryParam("service", "address")
				.queryParam("request", "getAddress")
				.queryParam("key", apiKey)
				.queryParam("errorFormat", "json")
				.queryParam("point", point)
				.queryParam("type", "PARCEL")
				.queryParam("simple", "true")
				.build()
			)
			.retrieve()
			.body(JsonNode.class);

		if (root == null) {
			log.error("[changePointToAddress] API 반환값이 null입니다.");
			throw new RestClientException("API returned null response");
		}

		// 2) response.result[0] 으로 내려가서
		JsonNode result = root
			.path("response")
			.path("result");
		if (result.isEmpty()) {
			log.warn("[changePointToAddress] 잘못된 좌표값, result가 비어있음: {}", point);
			throw new RestClientException("잘못된 좌표값입니다.");
		}
		JsonNode first = result.get(0);

		String address = first.path("text").asText("");
		log.debug("[changePointToAddress] 변환된 주소: {}", address);
		return address;
	}
}

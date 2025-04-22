package com.b201.api.dto;

import lombok.Builder;

@Builder
public record AddressDto(String province, String city, String district, String street) {

}

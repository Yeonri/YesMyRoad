package com.b201.api.dto;

import lombok.Builder;

@Builder
public record PropertiesDto(String publicId, AddressDto address) {
}
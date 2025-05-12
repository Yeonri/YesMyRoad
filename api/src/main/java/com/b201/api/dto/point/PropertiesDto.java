package com.b201.api.dto.point;

import com.b201.api.dto.AddressDto;

import lombok.Builder;

@Builder
public record PropertiesDto(String publicId, AddressDto address, Double accuracyMeters) {
}
